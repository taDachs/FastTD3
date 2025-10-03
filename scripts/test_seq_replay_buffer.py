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
import matplotlib.pyplot as plt
import cv2

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

from fast_td3.fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    PerTaskRewardNormalizer,
    SequenceReplayBuffer,
    save_params,
    mark_step,
)
from fast_td3.hyperparams import get_args, BaseArgs
from fast_td3.vision_backbone import DepthOnlyFCBackbone58x87, HeightMapHead

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass


def main():
    args: BaseArgs = get_args()
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

    n_act = 12
    n_obs = 45
    n_critic_obs = 434

    if args.teacher_obs_path:
        teacher_dict: TensorDict = TensorDict.load_memmap(args.teacher_obs_path)
        teacher_buffer_size = len(teacher_dict["observations"][0])
        teacher_n_envs = len(teacher_dict["observations"])
        print(f"Loading teacher observations with {teacher_n_envs} envs and length {teacher_buffer_size}")
        teacher_rb = SequenceReplayBuffer(
            n_env=teacher_n_envs,
            buffer_size=teacher_buffer_size,
            n_obs=n_obs,
            n_act=n_act,
            n_critic_obs=n_critic_obs,
            asymmetric_obs=True,
            playground_mode=False,
            n_steps=args.num_steps,
            gamma=args.gamma,
            use_vision=args.use_vision,
            vision_size=(128, 128),
            vision_update_rate=args.vision_update_rate,
            use_next_vision_obs=args.use_next_vision_obs,
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

    batch = teacher_rb.sample(args.batch_size, 1000)

    print(batch["vision_observations"].shape)
    vision_obs = batch["vision_observations"].detach().cpu().numpy()
    imgs = vision_obs[2]
    # previous = imgs[0]
    # for i, img in enumerate(imgs):
    #     print(i)
    #     plt.imshow(img, vmin=0.0, vmax=1.0)
    #     plt.show()
    #     print((previous == img).all())
    #     previous = img

    for i, (depth, color) in enumerate(zip(teacher_dict["vision_observations"], teacher_dict["color_vision_observations"])):
        aj = np.zeros((depth.shape[0], depth.shape[1] * 2, depth.shape[2], 3), dtype=np.uint8)
        aj[:, :depth.shape[1], :, :] = color.detach().cpu().numpy()[..., [2, 1, 0]]
        aj[:, depth.shape[1]:, :, :] = depth.detach().cpu().numpy().clip(0, 1) * 255
        # aj = (depth.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        fps = 10
        out = cv2.VideoWriter(f'videos/env_{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (aj.shape[2], aj.shape[1]))
        for i in range(aj.shape[0]):
            data = aj[i]
            # print(data.shape)
            out.write(data)
        out.release()


main()
