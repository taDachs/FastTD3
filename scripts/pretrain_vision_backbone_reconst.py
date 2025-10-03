#!/usr/bin/env python3

import numpy as np
import torch
from tensordict import TensorDict
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from fast_td3.vision_backbone import DepthOnlyFCBackbone58x87, DepthOnlyFCDecoder

LATENT_DIM = 128
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_STEPS = 1_000_000

device = torch.device("cuda:0")

teacher_dict: TensorDict = TensorDict.load_memmap("../../stored_transitions/")
vision_obs = teacher_dict["vision_observations"].view(-1, 1, 48, 48)
vision_obs = torch.clamp(vision_obs, 0.0, 1.0)


encoder = DepthOnlyFCBackbone58x87(LATENT_DIM).to(device)
decoder = DepthOnlyFCDecoder(LATENT_DIM).to(device)

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
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
    y = decoder(encoder(x))

    loss = (x - y) ** 2
    loss = loss.sum(dim=(1, 2, 3))
    loss = loss.mean()

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
        torch.save(decoder.state_dict(), f"vision_models/decoder_{step}.pt")

        sample_idx = torch.randint(0, len(vision_obs), (100, ))
        x = vision_obs[sample_idx].to(device)
        latent = encoder(x)
        y = decoder(latent)

        for i in range(len(sample_idx)):
            canvas = np.zeros((48, 48 * 2))
            canvas[:48, :48] = x[i].detach().cpu().numpy()
            canvas[:48, 48:] = y[i].detach().cpu().numpy()
            plt.imshow(canvas, vmin=0.0, vmax=1.0)
            plt.savefig(f"vision_models/imgs/{i}.png")
            plt.close()

        sample_idx = torch.randint(0, len(vision_obs), (1000, ))
        x = vision_obs[sample_idx].to(device)
        latent = encoder(x)
        pca = PCA(2)
        pts = pca.fit_transform(latent.detach().cpu().numpy())
        plt.scatter(pts[:, 0], pts[:, 1])
        plt.savefig(f"vision_models/imgs/pca_{step}.png")
