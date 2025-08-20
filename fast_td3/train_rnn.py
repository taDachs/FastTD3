import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import random
import time
import math

import tqdm
import wandb
import numpy as np

try:
    # Required for avoiding IsaacGym import error
    import isaacgym
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict, merge_tensordicts, is_tensor_collection

from fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    PerTaskRewardNormalizer,
    SequenceReplayBuffer,
    save_params,
    mark_step,
)
from hyperparams import get_args

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass


def main():
    args = get_args()
    print(args)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    if args.env_name.startswith("h1hand-") or args.env_name.startswith("h1-"):
        from environments.humanoid_bench_env import HumanoidBenchEnv

        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
        eval_envs = envs
        render_env = HumanoidBenchEnv(
            args.env_name, 1, render_mode="rgb_array", device=device
        )
    elif args.env_name.startswith("Isaac-") or args.env_name.startswith("Unitree-"):
        from environments.isaaclab_env import IsaacLabEnv

        env_type = "isaaclab"
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds if args.squash else None,
        )
        eval_envs = envs
        render_env = envs
    elif args.env_name.startswith("MTBench-"):
        from environments.mtbench_env import MTBenchEnv

        env_name = "-".join(args.env_name.split("-")[1:])
        env_type = "mtbench"
        envs = MTBenchEnv(env_name, args.device_rank, args.num_envs, args.seed)
        eval_envs = envs
        render_env = envs
    else:
        from environments.mujoco_playground_env import make_env

        # TODO: Check if re-using same envs for eval could reduce memory usage
        env_type = "mujoco_playground"
        envs, eval_envs, render_env = make_env(
            args.env_name,
            args.seed,
            args.num_envs,
            args.num_eval_envs,
            args.device_rank,
            use_tuned_reward=args.use_tuned_reward,
            use_domain_randomization=args.use_domain_randomization,
            use_push_randomization=args.use_push_randomization,
        )

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    action_low, action_high = args.action_low, args.action_high
    print(action_low)
    # exit()

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        if env_type in ["mtbench"]:
            reward_normalizer = PerTaskRewardNormalizer(
                num_tasks=envs.num_tasks,
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
        else:
            reward_normalizer = RewardNormalizer(
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
        "squash": args.squash,
        "noise_scheduling": args.noise_scheduling,
        "memory_type": args.memory_type,
        "memory_hidden_dim": args.memory_hidden_dim
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    if env_type == "mtbench":
        actor_kwargs["n_obs"] = n_obs - envs.num_tasks + args.task_embedding_dim
        critic_kwargs["n_obs"] = n_critic_obs - envs.num_tasks + args.task_embedding_dim
        actor_kwargs["num_tasks"] = envs.num_tasks
        actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
        critic_kwargs["num_tasks"] = envs.num_tasks
        critic_kwargs["task_embedding_dim"] = args.task_embedding_dim

    if args.agent == "fasttd3":
        if env_type in ["mtbench"]:
            raise NotImplementedError
            from fast_td3 import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3 import RNNActor, Critic

            actor_cls = RNNActor
            critic_cls = Critic

        print("Using FastTD3")
    elif args.agent == "fasttd3_simbav2":
        raise NotImplementedError
        if env_type in ["mtbench"]:
            from fast_td3_simbav2 import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3_simbav2 import Actor, Critic

            actor_cls = Actor
            critic_cls = Critic

        print("Using FastTD3 + SimbaV2")
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
                "alpha_init": 1.0 / (args.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": args.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
                "alpha_init": 1.0 / (args.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
                "num_blocks": args.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
    else:
        raise ValueError(f"Agent {args.agent} not supported")

    actor = actor_cls(**actor_kwargs)

    if env_type in ["mtbench"]:
        # Python 3.8 doesn't support 'from_module' in tensordict
        policy = actor.explore
    else:
        from tensordict import from_module

        actor_detach = actor_cls(**actor_kwargs)
        # Copy params to actor_detach without grad
        from_module(actor).data.to_module(actor_detach)
        policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs)
    qnet_target = critic_cls(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    # Add learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    rb = SequenceReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        playground_mode=env_type == "mujoco_playground",
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    if args.teacher_obs_path:
        teacher_dict: TensorDict = TensorDict.load_memmap(args.teacher_obs_path)
        teacher_buffer_size = len(teacher_dict["observations"][0])
        teacher_rb = SequenceReplayBuffer(
            n_env=args.num_envs,
            buffer_size=teacher_buffer_size,
            n_obs=n_obs,
            n_act=n_act,
            n_critic_obs=n_critic_obs,
            asymmetric_obs=envs.asymmetric_obs,
            playground_mode=env_type == "mujoco_playground",
            n_steps=args.num_steps,
            gamma=args.gamma,
            device=device,
        )

        for i in range(teacher_buffer_size):
            def get_slice(val):
                if is_tensor_collection(val):
                    # Recurse
                    return val.apply(get_slice, call_on_nested=True)
                return val[:, i]
            slice = teacher_dict.apply(get_slice)
            teacher_rb.extend(slice)


    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate():
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        if env_type == "isaaclab":
            obs = eval_envs.reset(random_start_init=False)
        else:
            obs = eval_envs.reset()

        # Run for a fixed number of steps
        if args.memory_type == "gru":
            hidden_in = torch.zeros(1, args.num_envs, actor.memory_hidden_dim, device=device)
        elif args.memory_type == "lstm":
            hidden_in = (
                    torch.zeros(1, args.num_envs, actor.memory_hidden_dim, device=device),
                    torch.zeros(1, args.num_envs, actor.memory_hidden_dim, device=device)
            )
        else:
            raise NotImplementedError
        for i in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs, update=False)
                obs = obs.unsqueeze(1)
                actions, hidden_in = actor(obs, hidden_in=hidden_in)
                actions = actions.squeeze(1)

            next_obs, rewards, dones, infos = eval_envs.step(actions.float())

            if env_type == "mtbench":
                # We only report success rate in MTBench evaluation
                rewards = (
                    infos["episode"]["success"].float() if "episode" in infos else 0.0
                )
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            if env_type == "mtbench" and "episode" in infos:
                dones = dones | infos["episode"]["success"]
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout():
        # Quick rollout for rendering
        if env_type == "humanoid_bench":
            obs = render_env.reset()
            renders = [render_env.render()]
        elif env_type in ["isaaclab", "mtbench"]:
            raise NotImplementedError(
                "We don't support rendering for IsaacLab and MTBench environments"
            )
        else:
            obs = render_env.reset()
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            renders = [render_env.state]
        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)
            next_obs, _, done, _ = render_env.step(actions.float())
            if env_type == "mujoco_playground":
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            if i % 2 == 0:
                if env_type == "humanoid_bench":
                    renders.append(render_env.render())
                else:
                    renders.append(render_env.state)
            if done.any():
                break
            obs = next_obs

        if env_type == "mujoco_playground":
            renders = render_env.render_trajectory(renders)
        return renders

    def update_main(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            if args.memory_type == "gru":
                hidden_in = torch.zeros(1, actions.shape[0], actor.memory_hidden_dim, device=device)
            elif args.memory_type == "lstm":
                hidden_in = (
                        torch.zeros(1, actions.shape[0], actor.memory_hidden_dim, device=device),
                        torch.zeros(1, actions.shape[0], actor.memory_hidden_dim, device=device)
                )
            else:
                raise NotImplementedError
            next_state_actions = (actor(next_observations, hidden_in)[0] + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]

            flat_next_state_actions = next_state_actions.reshape(-1, n_act)
            flat_next_critic_observations = next_critic_observations.reshape(-1, n_critic_obs)
            flat_rewards = rewards.reshape(-1)
            flat_bootstrap = bootstrap.reshape(-1)
            flat_discount = discount.reshape(-1)

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        flat_next_critic_observations,
                        flat_next_state_actions,
                        flat_rewards,
                        flat_bootstrap,
                        flat_discount,
                    )
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations.reshape(-1, n_critic_obs), actions.reshape(-1, n_act))
            done_mask = data["mask"].reshape(-1)
            # done_mask = torch.ones_like(done_mask, device=device)
            num_valid = max(1, done_mask.sum())
            qf1_loss = -(torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ) * done_mask).sum() / num_valid
            qf2_loss = -(torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ) * done_mask).sum() / num_valid
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )

            if args.memory_type == "gru":
                hidden_in = torch.zeros(1, critic_observations.shape[0], actor.memory_hidden_dim, device=device)
            elif args.memory_type == "lstm":
                hidden_in = (
                        torch.zeros(1, critic_observations.shape[0], actor.memory_hidden_dim, device=device),
                        torch.zeros(1, critic_observations.shape[0], actor.memory_hidden_dim, device=device)
                )
            else:
                raise NotImplementedError
            qf1, qf2 = qnet(
                critic_observations.reshape(-1, n_critic_obs),
                actor(data["observations"], hidden_in=hidden_in)[0].reshape(-1, n_act),
            )
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            done_mask = data["mask"].reshape(-1)
            # done_mask = torch.ones_like(done_mask, device=device)
            num_valid = max(1, done_mask.sum())
            actor_loss = -(qf_value * done_mask).sum() / num_valid

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    def sample_batch(batch_size: int, seq_len: int):
        if args.teacher_obs_path:
            teacher_data = teacher_rb.sample(max(1, batch_size // 2), seq_len=seq_len)
            student_data = rb.sample(max(1, batch_size // 2), seq_len=seq_len)
            data = TensorDict({
                "observations": torch.cat([student_data["observations"], teacher_data["observations"]], dim=0),
                "critic_observations": torch.cat([student_data["critic_observations"], teacher_data["critic_observations"]], dim=0),
                "actions": torch.cat([student_data["actions"], teacher_data["actions"]], dim=0),
                "next": {
                    "observations": torch.cat([student_data["next"]["observations"], teacher_data["next"]["observations"]], dim=0),
                    "critic_observations": torch.cat([student_data["next"]["critic_observations"], teacher_data["next"]["critic_observations"]], dim=0),
                    "rewards": torch.cat([student_data["next"]["rewards"], teacher_data["next"]["rewards"]], dim=0),
                    "truncations": torch.cat([student_data["next"]["truncations"], teacher_data["next"]["truncations"]], dim=0),
                    "dones": torch.cat([student_data["next"]["dones"], teacher_data["next"]["dones"]], dim=0),
                    "effective_n_steps": torch.cat([student_data["next"]["effective_n_steps"], teacher_data["next"]["effective_n_steps"]], dim=0),
                },
            })
        else:
            data = rb.sample(max(1, batch_size), seq_len=seq_len)
        data["observations"] = normalize_obs(data["observations"].reshape(-1, n_obs)).reshape(-1, seq_len, n_obs)
        data["next"]["observations"] = normalize_obs(
            data["next"]["observations"].reshape(-1, n_obs)
        ).reshape(-1, seq_len, n_obs)
        if envs.asymmetric_obs:
            data["critic_observations"] = normalize_critic_obs(
                data["critic_observations"].reshape(-1, n_critic_obs)
            ).reshape(-1, seq_len, n_critic_obs)
            data["next"]["critic_observations"] = normalize_critic_obs(
                data["next"]["critic_observations"].reshape(-1, n_critic_obs)
            ).reshape(-1, seq_len, n_critic_obs)
        raw_rewards = data["next"]["rewards"]
        if env_type in ["mtbench"] and args.reward_normalization:
            # Multi-task reward normalization
            task_ids_one_hot = data["observations"][..., -envs.num_tasks :]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            data["next"]["rewards"] = normalize_reward(
                raw_rewards, task_ids=task_indices
            )
        else:
            data["next"]["rewards"] = normalize_reward(raw_rewards)

        return data, raw_rewards


    if args.compile:
        compile_mode = args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        update_pol = torch.compile(update_pol, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    desc = ""

    cumulative_rewards = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, device=device)
    episode_returns = torch.zeros(args.num_envs, device=device)

    if args.memory_type == "gru":
        explore_hidden_in = torch.zeros(1, args.num_envs, actor.memory_hidden_dim, device=device)
    elif args.memory_type == "lstm":
        explore_hidden_in = (
                torch.zeros(1, args.num_envs, actor.memory_hidden_dim, device=device),
                torch.zeros(1, args.num_envs, actor.memory_hidden_dim, device=device)
        )
    else:
        raise NotImplementedError

    while global_step < args.total_timesteps:
        mark_step()
        logs_dict = TensorDict()
        if (
            start_time is None
            and global_step >= args.measure_burnin + args.learning_starts
        ):
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions, explore_hidden_out = policy(obs=norm_obs, hidden_in=explore_hidden_in, dones=dones)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]

        dones_mask = dones.bool()
        episode_returns = torch.where(dones_mask, cumulative_rewards, episode_returns)
        cumulative_rewards = torch.where(dones_mask, 0, cumulative_rewards + rewards)
        episode_lengths = torch.where(dones_mask, 0, episode_lengths + 1)

        if args.reward_normalization:
            if env_type == "mtbench":
                task_ids_one_hot = obs[..., -envs.num_tasks :]
                task_indices = torch.argmax(task_ids_one_hot, dim=1)
                update_stats(rewards, dones.float(), task_ids=task_indices)
            else:
                update_stats(rewards, dones.float())

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]
        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs
        rb.extend(transition)

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                if args.use_seq_critic:
                    batch_size = max(args.batch_size // (args.num_envs * args.seq_len), 1)
                    data, raw_rewards = sample_batch(batch_size, args.seq_len)
                else:
                    batch_size = max(args.batch_size // args.num_envs, 1)
                    data, raw_rewards = sample_batch(batch_size, 1)

                logs_dict = update_main(data, logs_dict)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1 or args.policy_frequency == 1:
                        policy_batch_size = max(args.batch_size // (args.num_envs * args.seq_len), 1)
                        data, _ = sample_batch(policy_batch_size, args.seq_len)
                        logs_dict = update_pol(data, logs_dict)
                else:
                    if global_step % args.policy_frequency == 0:
                        policy_batch_size = max(args.batch_size // (args.num_envs * args.seq_len), 1)
                        data, _ = sample_batch(policy_batch_size, args.seq_len)
                        logs_dict = update_pol(data, logs_dict)

                soft_update(qnet, qnet_target, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "Train/actor_loss": logs_dict["actor_loss"].mean(),
                        "Train/qf_loss": logs_dict["qf_loss"].mean(),
                        "Train/qf_max": logs_dict["qf_max"].mean(),
                        "Train/qf_min": logs_dict["qf_min"].mean(),
                        "Train/actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "Train/critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "Train/env_rewards": rewards.mean(),
                        "Train/buffer_rewards": raw_rewards.mean(),
                        "Train/mean_reward": episode_returns.mean(),
                        "Train/mean_episode_length": episode_lengths.mean(),
                    }
                    if "isaaclab" in infos:
                        for k, v in infos["isaaclab"]["log"].items():
                            logs[k] = v


                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        if env_type in ["humanoid_bench", "isaaclab", "mtbench"]:
                            # NOTE: Hacky way of evaluating performance, but just works
                            obs = envs.reset()
                        logs["Eval/eval_avg_return"] = eval_avg_return
                        logs["Eval/eval_avg_length"] = eval_avg_length

                    if (
                        args.render_interval > 0
                        and global_step % args.render_interval == 0
                    ):
                        renders = render_with_rollout()
                        render_video = wandb.Video(
                            np.array(renders).transpose(
                                0, 3, 1, 2
                            ),  # Convert to (T, C, H, W) format
                            fps=30,
                            format="gif",
                        )
                        logs["render_video"] = render_video
                if args.use_wandb:
                    wandb.log(
                        {
                            "Misc/speed": speed,
                            "Misc/frame": global_step * args.num_envs,
                            "Misc/critic_lr": q_scheduler.get_last_lr()[0],
                            "Misc/actor_lr": actor_scheduler.get_last_lr()[0],
                            **logs,
                        },
                        step=global_step,
                    )

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving model at global step {global_step}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    f"models/{run_name}_{global_step}.pt",
                )

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)

        explore_hidden_in = explore_hidden_out
        if args.memory_type == "gru":
            explore_hidden_in[:, dones == 1] = 0.0
        elif args.memory_type == "lstm":
            explore_hidden_in[0][:, dones == 1] = 0.0
            explore_hidden_in[1][:, dones == 1] = 0.0
        else:
            raise NotImplementedError

    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        f"models/{run_name}_final.pt",
    )


if __name__ == "__main__":
    main()
