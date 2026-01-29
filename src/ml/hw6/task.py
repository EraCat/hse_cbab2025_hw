import torch
from torch import nn


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5):
        super().__init__()

        layers = []
        ch = start_channels
        layers += [
            nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(downsamplings):
            layers += [
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2

        self.conv = nn.Sequential(*layers)

        self.final_spatial = img_size // (2 ** downsamplings)
        flat_dim = ch * self.final_spatial * self.final_spatial

        # вместо одного Linear на 2*latent
        self.flatten = nn.Flatten()
        self.mu_head = nn.Linear(flat_dim, latent_size)
        self.logvar_head = nn.Linear(flat_dim, latent_size)

        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x):
        h = self.conv(x)
        h = self.flatten(h)

        mu = self.mu_head(h)
        mu = 2.0 * torch.tanh(mu / 2.0)

        logvar = self.logvar_head(h)
        logvar = logvar.clamp(-6, 2)
        sigma = torch.exp(0.5 * logvar)

        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z, (mu, sigma)


# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5):
        super().__init__()

        self.img_size = img_size
        self.latent_size = latent_size
        self.end_channels = end_channels
        self.upsamplings = upsamplings

        self.s0 = img_size // (2 ** upsamplings)
        self.c0 = end_channels * (2 ** upsamplings)
        flat_dim = self.c0 * self.s0 * self.s0

        self.fc = nn.Linear(latent_size, flat_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.c0, self.s0, self.s0))

        # convtranspose-блоки: каждый шаг удваивает H,W и уменьшает каналы в 2 раза
        blocks = []
        ch = self.c0
        for _ in range(upsamplings):
            out_ch = ch // 2
            blocks += [
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch

        self.up = nn.Sequential(*blocks)

        self.out = nn.Sequential(
            nn.Conv2d(ch, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )


    def forward(self, z):
        h = self.fc(z)  # (B, C0*S0*S0)
        h = self.unflatten(h)
        h = self.up(h)  # (B, end_channels, img_size, img_size)
        x_pred = self.out(h)  # (B, 3, img_size, img_size)
        return x_pred


# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=3, latent_size=128, down_channels=6, up_channels=12):
        super().__init__()

        self.hparams = dict(
            img_size=img_size,
            downsamplings=downsamplings,
            latent_size=latent_size,
            down_channels=down_channels,
            up_channels=up_channels,
        )

        # Encoder: start_channels = down_channels, downsamplings как в VAE
        self.encoder = Encoder(
            img_size=img_size,
            latent_size=latent_size,
            start_channels=down_channels,
            downsamplings=downsamplings,
        )

        # Decoder: end_channels = up_channels, upsamplings = downsamplings
        self.decoder = Decoder(
            img_size=img_size,
            latent_size=latent_size,
            end_channels=up_channels,
            upsamplings=downsamplings,
        )


    def forward(self, x):
        z, (mu, sigma) = self.encoder(x)          # z: (B, latent), mu/sigma: (B, latent)

        x_pred = self.decoder(z)                   # (B, 3, H, W)

        # KL: 0.5 * (sigma^2 + mu^2 - log(sigma^2) - 1)
        sigma2 = sigma.pow(2)
        kld_per_dim = 0.5 * (sigma2 + mu.pow(2) - torch.log(sigma2 + 1e-8) - 1.0)
        kld = kld_per_dim

        return x_pred, kld

    def encode(self, x):
        z, _ = self.encoder(x)
        return z

    def decode(self, z):
        x_pred = self.decoder(z)
        return x_pred

    def save(self):
        path = __file__[:-7] + "model.pth"
        sd = {k: v.half() for k, v in self.state_dict().items()}
        torch.save({"state_dict": sd, "hparams": self.hparams}, path)

    def load(self):
        path = __file__[:-7] + "model.pth"
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["state_dict"])
        if "hparams" in ckpt:
            self.hparams = ckpt["hparams"]
        return self
