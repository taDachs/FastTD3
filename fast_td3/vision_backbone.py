import torch
import torch.nn as nn
import random


def random_translate(imgs, pad = 4):
    n, c, h, w = imgs.size()
    imgs = torch.nn.functional.pad(imgs, (pad, pad, pad, pad)) #, mode = "replicate")
    w1 = torch.randint(0, 2*pad + 1, (n,))
    h1 = torch.randint(0, 2*pad + 1, (n,))
    cropped = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i][:] = img[:, h11:h11 + h, w11:w11 + w]
    return cropped

def random_noise(images, noise_std=0.02):
    N, C, H, W = images.size()
    noise = noise_std*torch.randn_like(images)
    noise *= torch.bernoulli(0.5*torch.ones(N, 1, 1, 1).to(images)) # Only applied on half of the images
    return images + noise

def random_shift(images, padding=4):
    N, C, H, W = images.size()
    padded_images = torch.nn.functional.pad(images, (padding, padding, padding, padding), mode='replicate')
    crop_x = torch.randint(0, 2 * padding, (N,))
    crop_y = torch.randint(0, 2 * padding, (N,))
    shifted_images = torch.zeros_like(images)
    for i in range(N):
        shifted_images[i] = padded_images[i, :, crop_y[i]:crop_y[i] + H, crop_x[i]:crop_x[i] + W]
    return shifted_images

def random_cutout(images, min_size=2, max_size=24):
    N, C, H, W = images.size()
    for i in range(N):
        size_h = random.randint(min_size, max_size)
        size_w = random.randint(min_size, max_size)
        top = random.randint(0, H - size_h)
        left = random.randint(0, W - size_w)
        coin_flip = random.random()
        if coin_flip < 0.2:
            fill_value = 0.0
            images[i, :, top:top + size_h, left:left + size_w] = fill_value
        elif coin_flip < 0.4:
            fill_value = 1.0
            images[i, :, top:top + size_h, left:left + size_w] = fill_value
        elif coin_flip < 0.6:
            fill_value = torch.rand((C, size_h, size_w), device=images.device)
            images[i, :, top:top + size_h, left:left + size_w] = fill_value
    return images

class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, output_dim: int):
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

        self.output_activation = activation

    def forward(self, vobs, augment=False, hist=False):
        bs, c, w, h = vobs.size()
        assert c == 1

        vobs = vobs.view(-1, 1, w, h)
        if augment:
            vobs = random_cutout(random_noise(random_shift(vobs)))

        vision_latent = self.output_activation(self.image_compression(vobs))
        vision_latent = vision_latent.view(bs, 128)

        if hist:
            vision_latent = vision_latent.repeat_interleave(5, axis = 1)

        return vision_latent


if __name__ == "__main__":
    bb = DepthOnlyFCBackbone58x87(128)

    data = torch.zeros((16, 1, 48, 48))

    out = bb(data, augment=True, hist=False)
    print(data.shape)
    print(out.shape)
