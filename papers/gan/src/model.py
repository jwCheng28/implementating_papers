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
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision


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
            nn.Tanh()
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


class GANLightning(pl.LightningModule):
    def __init__(self, latent_dim: int = 100, lr: float = 0.0002):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        self.criterion = nn.BCELoss()

        self.validation_z = torch.randn(64, self.hparams.latent_dim, 1, 1)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)

        opt_g, opt_d = self.optimizers()

        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # train discriminator
        self.toggle_optimizer(opt_d)

        pred_reals = self.discriminator(real_imgs).view(batch_size, -1)
        d_real_loss = self.criterion(pred_reals, real_labels)

        z = torch.randn(batch_size, self.hparams.latent_dim,
                        1, 1, device=self.device)
        fake_imgs = self.generator(z)
        pred_fakes = self.discriminator(
            fake_imgs.detach()).view(batch_size, -1)
        d_fake_loss = self.criterion(pred_fakes, fake_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # train generator
        self.toggle_optimizer(opt_g)
        fake_imgs = self.generator(z)
        pred_fakes = self.discriminator(fake_imgs).view(batch_size, -1)
        g_loss = self.criterion(pred_fakes, real_labels)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.log_dict({"d_loss": d_loss, "g_loss": g_loss}, prog_bar=True)

    def on_train_epoch_end(self):
        z = self.validation_z.to(self.device)
        sample_imgs = self.generator(z)

        # Invert the [-1, 1] normalization back to [0, 1]
        sample_imgs = (sample_imgs + 1) / 2

        # Build a visual grid showing 64 synthesized outfit items
        grid = torchvision.utils.make_grid(sample_imgs, nrow=8)
        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image(
            "Generated_Images", grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []
