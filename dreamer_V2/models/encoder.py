import torch.nn as nn


class Encoder2D(nn.Module):
    def __init__(self, args, obs_channel=3):
        super(Encoder2D, self).__init__()
        self.observation_size = args.observation_size
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channel, 32, 4, stride=2),  # 64x64x3 -> 31x31x32
            nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 31x31x32 -> 14x14x64
            nn.ELU(),
            nn.Conv2d(64, 128, 4, stride=2),  # 14x14x64 -> 6x6x128
            nn.ELU(),
            nn.Conv2d(128, 256, 4, stride=2),  # 6x6x128 -> 2x2x256
            nn.Flatten(),
            nn.Linear(1024, self.observation_size),
        )

    def forward(self, obs):
        return self.encoder(obs)


class Encoder1D(nn.Module):
    def __init__(self, args, obs_size):
        super(Encoder1D, self).__init__()
        self.observation_size = args.observation_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, self.observation_size),
        )

    def forward(self, obs):
        return self.encoder(obs)
