import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent):
        super(VAE, self).__init__()
        self.k = latent
        self.mu = nn.Linear(2*self.k, self.k)
        self.var = nn.Linear(2*self.k, self.k)

        encoder = []
        for i, o in [(3, 24), (24, 36)]:
            encoder.append(nn.Conv2d(i, o, kernel_size=5, stride=2, padding=0))
            encoder.append(nn.ReLU())
        for i, o in [(36, 48), (48, 64), (64, 64)]:
            if i == 36:
                encoder.append(nn.Conv2d(i, o, kernel_size=5, stride=2, padding=0))
            else:
                encoder.append(nn.Conv2d(i, o, kernel_size=3, stride=1, padding=0))
            encoder.append(nn.ReLU())
        encoder.append(nn.Flatten())
        for i, o in [(1152, 1000), (1000, 100), (100, 2*self.k)]:
            encoder.append(nn.Linear(i, o))
            if i == 1152 or i == 1000:
                encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for i, o in [(self.k, 2*self.k), (2*self.k, 100), (100, 1000), (1000, 1152)]:
            decoder.append(nn.Linear(i, o))
            if i == 100 or i == 1000:
                decoder.append(nn.ReLU())
        decoder.append(nn.Unflatten(1, (64, 1, 18)))
        for i, o in [(64, 64), (64, 48), (48, 36)]:
            if i == 48:
                decoder.append(nn.ConvTranspose2d(i, o, kernel_size=5, stride=2, padding=0))
            else:
                decoder.append(nn.ConvTranspose2d(i, o, kernel_size=3, stride=1, padding=0))
            decoder.append(nn.ReLU())
        for i, o in [(36, 24), (24, 3)]:
            decoder.append(nn.ConvTranspose2d(i, o, kernel_size=5, stride=2, padding=0))
            decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def run_sample(self, mu, logvar):
        std = torch.exp(logvar/2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def forward(self, x):
        x = self.encode(x)
        # return x
        mu, logvar = self.mu(x), self.var(x)
        p, q, z = self.run_sample(mu, logvar)
        xhat = F.pad(self.decode(z), (2, 1, 3, 2), "constant", 0)
        return xhat, z, p, q
    
def loss_fn(y, x_hat, p, q):
    recon_loss = F.mse_loss(x_hat, y, reduction="mean")

    kl = torch.distributions.kl_divergence(q, p)
    kl = kl.mean()

    loss = kl * 0.4 + recon_loss
    return loss, kl, recon_loss