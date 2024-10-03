import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# state_dim: state의 차원
class Encoder(nn.Sequential):
    def __init__(self, state_dim, obs_channel=3):
        super(Encoder, self).__init__(
            nn.Conv2d(obs_channel, 32, 4, stride=2),  # 96x96x3 -> 47x47x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 47x47x32 -> 22x22x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),  # 22x22x64 -> 10x10x128
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),  # 10x10x128 -> 4x4x256
            nn.Flatten(),
            nn.Linear(4096, state_dim),
            nn.ReLU()
        )


# latent -> obs
class Decoder(nn.Module):
    def __init__(self, state_dim, deterministic_dim, obs_channel=3):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + deterministic_dim, 4096),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2),  # 4x4x256 -> 10x10x128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),  # 10x10x128 -> 22x22x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),  # 22x22x64 -> 47x47x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_channel, 4, stride=2),  # 47x47x32 -> 96x96x3
            nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )

    def forward(self, state, deterministic):
        return self.layers(torch.cat([state, deterministic], dim=-1))


# recurrent state space model
# prior -> (s_t-1, a_t-1) -> s_t
# posterior -> (s_t-1, a_t-1, o_t) -> s_t
class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, deterministic_dim):
        super(RSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.deterministic_dim = deterministic_dim

        # state, action, prev_deter -> deterministic
        self.rnn = nn.RNNCell(state_dim + action_dim, deterministic_dim)
        self.fc_proior = nn.Linear(deterministic_dim, state_dim * 2)  # mean, log_std
        self.fc_posterior = nn.Linear(state_dim + deterministic_dim, state_dim * 2)  # mean, log_std

    # prior
    # s_t-1, a_t-1 -> s_t
    def forward(self, prev_state, prev_action, prev_deter):
        deter = self.rnn(torch.cat([prev_state, prev_action], dim=-1), prev_deter)
        prior_mean, prior_log_var = self.fc_proior(deter).chunk(2, dim=-1)
        prior_std = torch.exp(0.5 * prior_log_var)
        return prior_mean, prior_std, deter

    # posterior
    # s_t-1, a_t-1, o_t -> s_t
    def posterior(self, prev_state, prev_action, prev_deter, obs_embedding):
        deter = self.rnn(torch.cat([prev_state, prev_action], dim=-1), prev_deter)
        posterior_mean, posterior_log_var = self.fc_posterior(
            torch.cat([deter, obs_embedding], dim=-1)).chunk(2, dim=-1)
        posterior_std = torch.exp(0.5 * posterior_log_var)
        return posterior_mean, posterior_std, deter

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.deterministic_dim)


# Reward model -> s -> r
# dense layer로 구현
class RewardModel(nn.Module):
    def __init__(self, state_dim, deterministic_dim):
        super(RewardModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + deterministic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, deterministic):
        return self.layers(torch.cat([state, deterministic], dim=-1))


# Observation model -> s -> o
# decoder로 구현
class ObservationModel(nn.Module):
    def __init__(self, state_dim, deterministic_dim, obs_channel=3):
        super(ObservationModel, self).__init__()
        self.decoder = Decoder(state_dim, deterministic_dim, obs_channel)

    def forward(self, state, deterministic):
        return self.decoder(state, deterministic)

# Transition model -> s, a -> q(s')
# Representation model -> s,a,o -> p(s')
# RSSM + CNN으로 구현
# RSSN으로 구현


class TransitionRepresentationModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.rssm = RSSM(latent_dim, action_dim, 256)

    def forward(self, action, prev_latent, prev_hidden):
        return self.rssm(prev_latent, action, prev_hidden)

    def posterior(self, action, prev_latent, prev_hidden, obs):
        obs_embedding = self.encoder(obs)
        return self.rssm.posterior(prev_latent, action, prev_hidden, obs_embedding)

    def init_hidden(self, batch_size):
        return self.rssm.init_hidden(batch_size)


# Agent -> s -> a
# 딱히 언급이 없으니 dense layer로 구성
class Agent(nn.Module):
    def __init__(self, state_dim, deterministic_dim, action_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(state_dim + deterministic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_var = nn.Linear(256, action_dim)

    def forward(self, state, deterministic):
        x = self.seq(torch.cat([state, deterministic], dim=-1))
        mu = self.mu(x)
        log_var = self.log_var(x)
        std = torch.exp(0.5 * log_var)
        return mu, std


# Value model -> s -> v
# 역시 언급이 없어서 dense layer로 구현
class ValueModel(nn.Module):
    def __init__(self, state_dim, deterministic_dim):
        super(ValueModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(state_dim + deterministic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, deterministic):
        return self.seq(torch.cat([state, deterministic], dim=-1))


class ReplayBufferSeq:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, data):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample_seq(self, batch_size, seq_len):
        indices = np.random.randint(0, len(self.memory) - seq_len, size=batch_size)
        batch = []
        for idx in indices:
            seq = [self.memory[idx + i] for i in range(seq_len)]
            batch.append(seq)
        return batch

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
