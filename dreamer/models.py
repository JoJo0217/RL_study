import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    def __init__(self, state_dim, action_dim, deterministic_dim, hidden_dim=256):
        super(RSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.deterministic_dim = deterministic_dim

        # state, action, prev_deter -> deterministic
        self.rnn_input = nn.Linear(state_dim + action_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.rnn = nn.RNNCell(hidden_dim, deterministic_dim)
        self.fc_proior = nn.Linear(deterministic_dim, state_dim * 2)  # mean, log_std
        self.fc_posterior = nn.Linear(state_dim + deterministic_dim, state_dim * 2)  # mean, log_std

    # prior
    # s_t-1, a_t-1 -> s_t
    def forward(self, prev_state, prev_action, prev_deter):
        hidden = self.relu(self.rnn_input(torch.cat([prev_state, prev_action], dim=-1)))
        deter = self.rnn(hidden, prev_deter)
        prior_mean, prior_std = self.fc_proior(deter).chunk(2, dim=-1)
        prior_std = F.softplus(prior_std) + 0.1
        return prior_mean, prior_std, deter

    # posterior
    # s_t-1, a_t-1, o_t -> s_t
    def posterior(self, prev_state, prev_action, prev_deter, obs_embedding):
        hidden = self.relu(self.rnn_input(torch.cat([prev_state, prev_action], dim=-1)))
        deter = self.rnn(hidden, prev_deter)
        posterior_mean, posterior_std = self.fc_posterior(
            torch.cat([deter, obs_embedding], dim=-1)).chunk(2, dim=-1)
        posterior_std = F.softplus(posterior_std) + 0.1
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
            nn.Linear(256, 1)
        )

    def forward(self, state, deterministic):
        return self.layers(torch.cat([state, deterministic], dim=-1))


# Observation model -> s -> o
# decoder로 구현
class ObservationModel(nn.Module):
    def __init__(self, decoder):
        super(ObservationModel, self).__init__()
        self.decoder = decoder

    def forward(self, state, deterministic):
        return self.decoder(state, deterministic)

# Transition model -> s, a -> q(s')
# Representation model -> s,a,o -> p(s')
# RSSM + CNN으로 구현
# RSSN으로 구현


class TransitionRepresentationModel(nn.Module):
    def __init__(self, encoder, latent_dim, action_dim):
        super().__init__()
        self.encoder = encoder
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
            nn.Linear(256, 256)
        )
        self.mu = nn.Linear(256, action_dim)
        self.std = nn.Linear(256, action_dim)

    def forward(self, state, deterministic):
        x = self.seq(torch.cat([state, deterministic], dim=-1))
        mu = self.mu(x)
        std = self.std(x)
        std = F.softplus(std) + 1e-4

        return mu, std


# Value model -> s -> v
# 역시 언급이 없어서 dense layer로 구현
class ValueModel(nn.Module):
    def __init__(self, state_dim, deterministic_dim):
        super(ValueModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(state_dim + deterministic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, deterministic):
        return self.seq(torch.cat([state, deterministic], dim=-1))


class ReplayBufferSeq:
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool_)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:])
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1])
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        return sampled_observations, sampled_actions, sampled_rewards, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.index
