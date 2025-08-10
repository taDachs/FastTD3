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

from tensordict import TensorDict

from fast_td3.fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    PerTaskRewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)
from fast_td3.fast_td3_deploy import load_policy
from fast_td3.hyperparams import get_args

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass


def main():
    args = get_args()
    print(args)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

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
        from fast_td3.environments.humanoid_bench_env import HumanoidBenchEnv

        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
        eval_envs = envs
        render_env = HumanoidBenchEnv(
            args.env_name, 1, render_mode="rgb_array", device=device
        )
    elif args.env_name.startswith("Isaac-") or args.env_name.startswith("Unitree-"):
        from fast_td3.environments.isaaclab_env import IsaacLabEnv

        env_type = "isaaclab"
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds,
            headless=False,
        )
        eval_envs = envs
        render_env = envs
    elif args.env_name.startswith("MTBench-"):
        from fast_td3.environments.mtbench_env import MTBenchEnv

        env_name = "-".join(args.env_name.split("-")[1:])
        env_type = "mtbench"
        envs = MTBenchEnv(env_name, args.device_rank, args.num_envs, args.seed)
        eval_envs = envs
        render_env = envs
    else:
        from fast_td3.environments.mujoco_playground_env import make_env

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

    policy = load_policy(args.checkpoint_path).to(device)

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()

    global_step = 0

    dones = None
    start_time = None

    while global_step < args.total_timesteps:
        with torch.no_grad():
            actions = policy(obs)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]

if __name__ == "__main__":
    main()
