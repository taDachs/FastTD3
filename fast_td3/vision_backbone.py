import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def random_noise(images, noise_std=0.02):
    N, C, H, W = images.size()
    noise = noise_std*torch.randn_like(images)
    noise *= torch.bernoulli(0.5*torch.ones(N, 1, 1, 1).to(images)) # Only applied on half of the images
    return images + noise


def random_shift(imgs, pad=4):
    # identical behavior to your version (replicate-pad, integer crop)
    N, C, H, W = imgs.shape
    padded = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')  # (N,C,H+2p,W+2p)

    # random offsets in [0, 2*pad]
    off_x = torch.randint(0, 2*pad + 1, (N,), device=imgs.device)
    off_y = torch.randint(0, 2*pad + 1, (N,), device=imgs.device)

    # build per-image x/y index grids
    x = torch.arange(W, device=imgs.device)[None, :] + off_x[:, None]  # (N, W)
    y = torch.arange(H, device=imgs.device)[None, :] + off_y[:, None]  # (N, H)

    # expand to (N,C,H,W)
    idx_x = x[:, None, None, :].expand(N, C, H, W)
    idx_y = y[:, None, :, None].expand(N, C, H, W)
    b = torch.arange(N, device=imgs.device)[:, None, None, None].expand(N, C, H, W)
    c = torch.arange(C, device=imgs.device)[None, :, None, None].expand(N, C, H, W)

    return padded[b, c, idx_y, idx_x]

# def random_shift(images, padding=4):
#     N, C, H, W = images.size()
#     padded_images = torch.nn.functional.pad(images, (padding, padding, padding, padding), mode='replicate')
#     crop_x = torch.randint(0, 2 * padding, (N,))
#     crop_y = torch.randint(0, 2 * padding, (N,))
#     shifted_images = torch.zeros_like(images)
#     for i in range(N):
#         shifted_images[i] = padded_images[i, :, crop_y[i]:crop_y[i] + H, crop_x[i]:crop_x[i] + W]
#     return shifted_images
#
# def random_cutout(images, min_size=2, max_size=24):
#     N, C, H, W = images.size()
#     for i in range(N):
#         size_h = random.randint(min_size, max_size)
#         size_w = random.randint(min_size, max_size)
#         top = random.randint(0, H - size_h)
#         left = random.randint(0, W - size_w)
#         coin_flip = random.random()
#         if coin_flip < 0.2:
#             fill_value = 0.0
#             images[i, :, top:top + size_h, left:left + size_w] = fill_value
#         elif coin_flip < 0.4:
#             fill_value = 1.0
#             images[i, :, top:top + size_h, left:left + size_w] = fill_value
#         elif coin_flip < 0.6:
#             fill_value = torch.rand((C, size_h, size_w), device=images.device)
#             images[i, :, top:top + size_h, left:left + size_w] = fill_value
#     return images

def random_cutout(images, min_size=2, max_size=24):
    # In-place behavior like your version (mutates the input tensor).
    # Vectorized: builds per-image rectangle masks and applies three fill modes.
    N, C, H, W = images.shape
    device = images.device
    dtype = images.dtype

    # rectangle sizes and positions per image
    size_h = torch.randint(min_size, max_size + 1, (N,), device=device)
    size_w = torch.randint(min_size, max_size + 1, (N,), device=device)
    top = torch.randint(0, H, (N,), device=device).clamp_max(H - 1)
    left = torch.randint(0, W, (N,), device=device).clamp_max(W - 1)
    bottom = (top + size_h).clamp_max(H)
    right  = (left + size_w).clamp_max(W)

    # choose mode per image: 0->fill 0, 1->fill 1, 2->random, 3/4->do nothing (to match your 60% apply rate)
    r = torch.rand(N, device=device)
    mode0 = r < 0.2
    mode1 = (r >= 0.2) & (r < 0.4)
    modeR = (r >= 0.4) & (r < 0.6)
    apply_mask = mode0 | mode1 | modeR  # only 60% applied

    # build a (N,H,W) boolean mask for rectangles
    ys = torch.arange(H, device=device)[None, :, None]            # (1,H,1)
    xs = torch.arange(W, device=device)[None, None, :]            # (1,1,W)
    top_ = top[:, None, None]                                     # (N,1,1)
    left_ = left[:, None, None]
    bottom_ = bottom[:, None, None]
    right_ = right[:, None, None]

    rect_mask = (ys >= top_) & (ys < bottom_) & (xs >= left_) & (xs < right_)  # (N,H,W)
    rect_mask &= apply_mask[:, None, None]

    if rect_mask.any():
        # prepare fillers
        # zeros/ones fillers only where their mode applies
        zero_mask = rect_mask & mode0[:, None, None]
        one_mask  = rect_mask & mode1[:, None, None]
        rand_mask = rect_mask & modeR[:, None, None]

        # broadcast to channels
        zero_mask_c = zero_mask[:, None, :, :].expand(N, C, H, W)
        one_mask_c  = one_mask[:,  None, :, :].expand(N, C, H, W)
        rand_mask_c = rand_mask[:,  None, :, :].expand(N, C, H, W)

        # random filler for the random mode
        rand_fill = torch.rand(N, C, H, W, device=device, dtype=dtype)

        # apply fills
        images = images.clone()  # avoid in-place on view-shared tensors
        images[zero_mask_c] = 0.0
        images[one_mask_c]  = 1.0
        images[rand_mask_c] = rand_fill[rand_mask_c]

    return images

class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, output_dim: int, use_layer_norm: bool = False):
        super().__init__()

        # activation = nn.ELU()
        activation = nn.Tanh()
        self.image_compression = nn.Sequential(
            nn.Conv2d(1, 16, 5), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 4), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1568, 128), # 48x48
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)
        )
        self.ln = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        self.output_activation = activation

    def forward(self, vobs, augment=False, hist=False):
        bs, c, w, h = vobs.size()
        assert c == 1

        vobs = vobs.view(-1, 1, w, h)
        if augment:
            vobs = random_cutout(random_noise(random_shift(vobs)))

        vision_latent = self.output_activation(self.ln(self.image_compression(vobs)))
        vision_latent = vision_latent.view(bs, 128)

        if hist:
            vision_latent = vision_latent.repeat_interleave(5, axis = 1)

        return vision_latent

class DepthOnlyFCDecoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        # activation = nn.ELU()
        activation = nn.Tanh()
        self.image_decompression = nn.Sequential(
                        # Inverse of the encoder's last linear layer
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1568),
            nn.LeakyReLU(),
            nn.Unflatten(1, (32, 7, 7)),

            nn.ConvTranspose2d(32, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5),
            nn.Sigmoid()
        )

        self.output_activation = activation

    def forward(self, x: torch.Tensor):
        x = self.image_decompression(x)

        return x


class HeightMapHead(nn.Module):
    def __init__(self, input_dim: int, output_size: tuple[int, int]):
        super().__init__()

        self.output_size = output_size

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, output_size[0] * output_size[1])
        )


    def forward(self, x: torch.Tensor):
        x = self.mlp(x)

        x = x.reshape(-1, *self.output_size)

        return x


if __name__ == "__main__":
    bb = DepthOnlyFCBackbone58x87(128)
    de = DepthOnlyFCDecoder(128)

    data = torch.zeros((16, 1, 48, 48))

    out = bb(data)
    reconst = de(out)

    print(data.shape)
    print(out.shape)

    print(reconst.shape)
