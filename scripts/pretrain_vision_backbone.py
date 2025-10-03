#!/usr/bin/env python3

import numpy as np
import torch
from tensordict import TensorDict
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from fast_td3.vision_backbone import DepthOnlyFCBackbone58x87, HeightMapHead

LATENT_DIM = 128
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
NUM_STEPS = 1_000_000
GRID_RES = 0.1
GRID_SIZE = (1.0, 1.6)

device = torch.device("cuda:0")

height_map_discrete_size = (int(GRID_SIZE[0] / GRID_RES) + 1, int(GRID_SIZE[1] / GRID_RES) + 1)
height_map_length = int(np.prod(height_map_discrete_size))

print(height_map_discrete_size)
print(height_map_length)

# teacher_dict: TensorDict = TensorDict.load_memmap("../../stored_transitions_ppo_teacher_real_parkour/")
teacher_dict: TensorDict = TensorDict.load_memmap("../../stored_transitions/")
vision_obs = teacher_dict["vision_observations"].view(-1, 1, 48, 48)
vision_obs = torch.clamp(vision_obs, 0.0, 1.0)
print(teacher_dict["critic_observations"].shape)
print(teacher_dict["critic_observations"][..., -height_map_length*2:-height_map_length][0])
height_maps = (
    teacher_dict["critic_observations"][..., -height_map_length*2:-height_map_length]
    .view(-1, height_map_length)
    .view(-1, int(height_map_discrete_size[0]), int(height_map_discrete_size[1]))
)
roof_maps = (
    teacher_dict["critic_observations"][..., -height_map_length:]
    .view(-1, height_map_length)
    .view(-1, int(height_map_discrete_size[0]), int(height_map_discrete_size[1]))
)

max_height = height_maps.max()
min_height = height_maps.min()
max_roof = roof_maps.max()
min_roof = roof_maps.min()

encoder = DepthOnlyFCBackbone58x87(LATENT_DIM).to(device)
height_map_decoder = HeightMapHead(LATENT_DIM, height_map_discrete_size).to(device)
roof_map_decoder = HeightMapHead(LATENT_DIM, height_map_discrete_size).to(device)

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(height_map_decoder.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.1,
)


print("Vision obs shape: ", vision_obs.shape)

pbar = tqdm(range(NUM_STEPS))

running_loss = None

for step in pbar:
    batch_idx = torch.randint(0, len(vision_obs), (BATCH_SIZE, ))

    optimizer.zero_grad()


    x = vision_obs[batch_idx].to(device)

    y_height = height_maps[batch_idx].to(device)
    y_height_hat = height_map_decoder(encoder(x))

    y_roof = roof_maps[batch_idx].to(device)
    y_roof_hat = roof_map_decoder(encoder(x))

    loss = (y_height_hat - y_height) ** 2 + (y_roof_hat - y_roof) ** 2
    loss = loss.sum(dim=(1, 2)).mean()

    loss.backward()
    optimizer.step()

    if running_loss is None:
        running_loss = loss.detach()
    else:
        running_loss *= 0.8
        running_loss += loss.detach() * 0.2

    pbar.set_description(f"Loss: {running_loss:4f}")

    if step % 1000 == 0:
        torch.save(encoder.state_dict(), f"vision_models/encoder_{step}.pt")
        torch.save(height_map_decoder.state_dict(), f"vision_models/decoder_{step}.pt")

        sample_idx = torch.randint(0, len(vision_obs), (100, ))
        x = vision_obs[sample_idx].to(device)
        latent = encoder(x)
        y_height = height_maps[sample_idx].to(device)
        y_height_hat = height_map_decoder(latent)
        y_roof = roof_maps[sample_idx].to(device)
        y_roof_hat = roof_map_decoder(latent)

        for i in range(len(sample_idx)):
            canvas = np.zeros((height_map_discrete_size[0], height_map_discrete_size[1] * 2))

            canvas[:height_map_discrete_size[0], :height_map_discrete_size[1]] = y_height[i].detach().cpu().numpy()
            canvas[:height_map_discrete_size[0], height_map_discrete_size[1]:] = y_height_hat[i].detach().cpu().numpy()
            plt.imshow(canvas, vmin=min_height, vmax=max_height)
            plt.savefig(f"vision_models/imgs/height_{i}.png")
            plt.close()

            canvas[:height_map_discrete_size[0], :height_map_discrete_size[1]] = y_roof[i].detach().cpu().numpy()
            canvas[:height_map_discrete_size[0], height_map_discrete_size[1]:] = y_roof_hat[i].detach().cpu().numpy()
            plt.imshow(canvas, vmin=min_roof, vmax=max_roof)
            plt.savefig(f"vision_models/imgs/roof_{i}.png")
            plt.close()

        sample_idx = torch.randint(0, len(vision_obs), (10_000, ))
        x = vision_obs[sample_idx].to(device)
        latent = encoder(x)
        pca = PCA(2)
        pts = pca.fit_transform(latent.detach().cpu().numpy())
        plt.scatter(pts[:, 0], pts[:, 1])
        plt.savefig(f"vision_models/imgs/pca_{step}.png")
        plt.close()
