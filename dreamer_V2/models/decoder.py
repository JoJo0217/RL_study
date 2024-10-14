import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class Decoder2D(nn.Module):
    def __init__(self, args, obs_channel=3):
        super(Decoder2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, 1024),
            nn.ELU(),
            nn.Unflatten(1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, 4, stride=2),  # 2x2x256 -> 6x6x128
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),  # 6x6x128 -> 14x14x64
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),  # 14x14x64 -> 31x31x32
            nn.ELU(),
            nn.ConvTranspose2d(32, obs_channel, 4, stride=2),  # 31x31x32 -> 64x64x3
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
        )

    def forward(self, state, deterministic):
        x = torch.cat([state, deterministic], dim=-1)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        pred = self.layers(x)
        # (batch*seq, obs_channel, 64, 64)
        pred_shape = pred.shape
        pred = pred.reshape(*shape[:-1], *pred_shape[1:])
        m = Normal(pred, 1)
        dist = Independent(m, 3)
        return dist


class Decoder1D(nn.Module):
    def __init__(self, args, obs_size):
        super(Decoder1D, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, obs_size)
        )

    def forward(self, state, deterministic):
        x = torch.cat([state, deterministic], dim=-1)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        pred = self.decoder(x)
        pred = pred.reshape(*shape[:-1], -1)
        dist = Normal(pred, 1)
        dist = Independent(dist, 1)
        return dist
