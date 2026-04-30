"""
model.py
========
Convolutional Variational Autoencoder for radar micro-Doppler anomaly
detection.

Architecture:
  Encoder: 4 stride-2 Conv2d(k=4)+BN+LeakyReLU blocks
  128x256 → 64x128 → 32x64 → 16x32 → 8x16 (with channels base*1..base*8)
  Bottleneck: flatten → 2 linear heads (mu, logvar) of latent_dim

  Decoder: mirror of encoder via ConvTranspose2d
  8x16 → 16x32 → 32x64 → 64x128 → 128x256

Notes:
  * Final decoder output has NO nonlinearity — input is standardized
    log-magnitude in dB, not in [0,1].
  * BatchNorm is used despite per-sample input standardization because
    activation distributions still drift per-batch.
  * Default param count ≈ 14M; model file ~55 MB.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _deconv_block(in_c, out_c, last=False):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
    if not last:
        layers += [nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, base=32, latent_dim=64):
        super().__init__()
        self.b1 = _conv_block(in_channels, base)
        self.b2 = _conv_block(base, base * 2)
        self.b3 = _conv_block(base * 2, base * 4)
        self.b4 = _conv_block(base * 4, base * 8)
        self.flatten = nn.Flatten()
        self.fc_mu     = nn.Linear(base * 8 * 8 * 16, latent_dim)
        self.fc_logvar = nn.Linear(base * 8 * 8 * 16, latent_dim)

    def forward(self, x):
        h = self.b4(self.b3(self.b2(self.b1(x))))
        h = self.flatten(h)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, out_channels=1, base=32, latent_dim=64):
        super().__init__()
        self.base = base
        self.fc = nn.Linear(latent_dim, base * 8 * 8 * 16)
        self.b1 = _deconv_block(base * 8, base * 4)
        self.b2 = _deconv_block(base * 4, base * 2)
        self.b3 = _deconv_block(base * 2, base)
        self.b4 = _deconv_block(base, out_channels, last=True)

    def forward(self, z):
        h = self.fc(z).view(-1, self.base * 8, 8, 16)
        return self.b4(self.b3(self.b2(self.b1(h))))


class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, base=32, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(in_channels, base, latent_dim)
        self.decoder = Decoder(in_channels, base, latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        # Clamp logvar to a sane range. Without this, the encoder can produce
        # logvar > 30, exp(15) overflows, KL → inf, gradients → NaN. This is
        # standard VAE hardening and would have been needed earlier if not
        # for per-sample standardization keeping inputs tiny.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor,
                      recon_weight: float = 1.0,
                      kl_weight: float = 0.5,
                      n_samples: int = 8) -> torch.Tensor:
        self.eval()
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        recon_errors = []
        for _ in range(n_samples):
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            x_hat = self.decoder(z)
            recon_errors.append(((x - x_hat) ** 2).flatten(1).mean(dim=1))
        recon = torch.stack(recon_errors, dim=0).mean(dim=0)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return recon_weight * recon + kl_weight * kl


def vae_loss(x, x_hat, mu, logvar, beta: float = 1.0):
    """β-VAE loss with mean-over-pixels reconstruction so that the KL
    weight β is meaningful at order-of-magnitude 1.

    Old code used sum-over-pixels for recon and sum-over-latents for KL,
    so for a 128×256 input and 64-d latent the recon term was ~500×
    larger than the KL term at β=1. Effective β was ~0.002, and the
    posterior never got pushed toward the prior — manifesting as
    minutes of training where the model overfit normals and didn't
    discriminate falls. Using mean for both terms keeps β interpretable.
    """
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + beta * kl, recon, kl
