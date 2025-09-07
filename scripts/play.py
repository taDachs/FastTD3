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
from fast_td3.vision_backbone import DepthOnlyFCBackbone58x87

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

        envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
    elif args.env_name.startswith("Isaac-") or args.env_name.startswith("Unitree-"):
        from fast_td3.environments.isaaclab_env import IsaacLabEnv

        print("actions bounds: ", args.action_bounds)
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds if args.squash else None,
            headless=False,
        )
    elif args.env_name.startswith("MTBench-"):
        from fast_td3.environments.mtbench_env import MTBenchEnv

        env_name = "-".join(args.env_name.split("-")[1:])
        envs = MTBenchEnv(env_name, args.device_rank, args.num_envs, args.seed)
    else:
        from fast_td3.environments.mujoco_playground_env import make_env

        # TODO: Check if re-using same envs for eval could reduce memory usage
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

    policy = load_policy(args.checkpoint_path, use_memory=True).to(device)

    if envs.asymmetric_obs:
        if args.use_vision:
            obs, critic_obs, vision_obs = envs.reset_with_critic_and_vision_obs()
            critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
            vision_obs = torch.as_tensor(vision_obs, device=device, dtype=torch.float)
        else:
            vision_obs = None
            obs, critic_obs = envs.reset_with_critic_obs()
            critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()


    global_step = 0

    memory_hidden_in = torch.zeros((1, args.num_envs, args.memory_hidden_dim)).to(device)
    while global_step < args.total_timesteps:
        with torch.no_grad():
            actions, memory_hidden_out = policy(obs, memory_hidden_in, image=vision_obs)

        obs, _, dones, infos = envs.step(actions.float())
        memory_hidden_in = memory_hidden_out
        if args.use_vision:
            vision_obs = infos["observations"]["vision"]
        memory_hidden_in[:, dones == 1] = 0.0

if __name__ == "__main__":
    main()
