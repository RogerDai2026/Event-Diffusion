import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# === VAE ARCHITECTURE ===
class VAE(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int, input_height: int, input_width: int):
        """
        VAE with 3-level conv encoder and decoder.
        Args:
            in_ch: number of input channels (e.g., 5 for nbin5 event maps)
            latent_dim: size of the bottleneck vector
            input_height: height of input images/events (must be divisible by 8)
            input_width: width of input images/events (must be divisible by 8)
        """
        super().__init__()
        # Compute spatial dims after 3 downsamples
        h_lat = input_height // 8
        w_lat = input_width  // 8
        flat_dim = 256 * h_lat * w_lat

        # Encoder: downsample by 2 three times
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch,  64, 4, 2, 1), nn.ReLU(),  # H/2 × W/2
            nn.Conv2d(64,   128, 4, 2, 1), nn.ReLU(),  # H/4 × W/4
            nn.Conv2d(128,  256, 4, 2, 1), nn.ReLU(),  # H/8 × W/8
            nn.Flatten()
        )
        # Bottleneck to μ and logσ²
        self.to_mu     = nn.Linear(flat_dim, latent_dim)
        self.to_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder: project back up
        self.dec_fc = nn.Linear(latent_dim, flat_dim)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (256, h_lat, w_lat)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),  # ×2
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(),  # ×2
            nn.ConvTranspose2d( 64, in_ch,4, 2, 1), nn.Sigmoid()# ×2 back to input size
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # Encode
        h = self.enc(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        # Decode
        h2 = self.dec_fc(z)
        recon = self.dec(h2)
        return recon, mu, logvar
