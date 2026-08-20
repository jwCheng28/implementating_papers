"""
The Generator: 
Takes a 1D vector of random noise (Batch, 100, 1, 1) 
and uses Up Transposed Convolutions
to upsample the spatial dimensions 
from 1×1 to 7×7 to 14×14 to 28×28 shape

The Discriminator: 
A convolutional downsampler. 
Takes an image (Batch, 1, 28, 28) 
and squeezes it down into binary classification 
scalar between 0 (fake) and 1 (real)
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.gen_model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.Tanh(True)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.gen_model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.disc_model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.disc_model(img).view(-1, 1)
