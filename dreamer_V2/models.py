import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import OneHotCategorical, Normal, Independent, Bernoulli, kl_divergence


# Recurrent model: (h_t-1, s_t-1, a_t) -> h_t
class RSSM(nn.Module):
    def __init__(self, args, action_size):
        super(RSSM, self).__init__()
        self.action_size = action_size
        self.stoch_size = args.state_size
        self.determinisic_size = args.deterministic_size
        self.rnn_input = nn.Sequential(
            nn.Linear(args.state_size + self.action_size, args.hidden_size),
            nn.ELU()
        )
        self.rnn = nn.GRUCell(input_size=args.hidden_size, hidden_size=self.determinisic_size)

    def forward(self, state, action, hidden):
        x = torch.cat([state, action], dim=-1)
        x = self.rnn_input(x)
        hidden = self.rnn(x, hidden)
        return hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.determinisic_size)

    def init_state(self, batch_size):
        return torch.zeros(batch_size, self.stoch_size)


class Encoder3D(nn.Module):
    def __init__(self, args, obs_channel=3):
        super(Encoder3D, self).__init__()
        self.observation_size = args.observation_size
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channel, 32, 4, stride=2),  # 96x96x3 -> 47x47x32
            nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 47x47x32 -> 22x22x64
            nn.ELU(),
            nn.Conv2d(64, 128, 4, stride=2),  # 22x22x64 -> 10x10x128
            nn.ELU(),
            nn.Conv2d(128, 256, 4, stride=2),  # 10x10x128 -> 4x4x256
            nn.Flatten(),
            nn.Linear(4096, self.observation_size),
        )


class Encoder2D(nn.Module):
    def __init__(self, args, obs_size):
        super(Encoder2D, self).__init__()
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


def get_categorical_state(logits, categorical_size, class_size):
    shape = logits.shape
    logit = torch.reshape(logits, shape=[*shape[:-1], categorical_size, class_size])
    dist = OneHotCategorical(logits=logit)
    stoch = dist.sample()
    stoch += dist.probs - dist.probs.detach()
    return dist, torch.flatten(stoch, start_dim=-2, end_dim=-1)


def get_dist_stopgrad(logits, categorical_size, class_size):
    logits = logits.detach()
    shape = logits.shape
    logit = torch.reshape(logits, shape=[*shape[:-1], categorical_size, class_size])
    dist = OneHotCategorical(logits=logit)
    stoch = dist.sample()
    stoch += dist.probs - dist.probs.detach()
    return dist, torch.flatten(stoch, start_dim=-2, end_dim=-1)


class RepresentationModel(nn.Module):
    def __init__(self, args):
        super(RepresentationModel, self).__init__()
        self.state_size = args.state_size
        self.category_size = args.categorical_size
        self.class_size = args.class_size
        self.MLP = nn.Sequential(
            nn.Linear(args.deterministic_size + args.observation_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, self.state_size),
        )

    def forward(self, hidden, obs):
        x = torch.cat([hidden, obs], dim=-1)
        logits = self.MLP(x)
        return get_categorical_state(logits, self.category_size, self.class_size)

    def stop_grad(self, hidden, obs):
        x = torch.cat([hidden, obs], dim=-1)
        logits = self.MLP(x)
        return get_dist_stopgrad(logits, self.category_size, self.class_size)


class TransitionModel(nn.Module):
    def __init__(self, args):
        super(TransitionModel, self).__init__()
        self.state_size = args.state_size
        self.category_size = args.categorical_size
        self.class_size = args.class_size
        self.MLP = nn.Sequential(
            nn.Linear(args.deterministic_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.state_size),
        )

    def forward(self, hidden):
        logits = self.MLP(hidden)
        return get_categorical_state(logits, self.category_size, self.class_size)

    def stop_grad(self, hidden):
        logits = self.MLP(hidden)
        return get_dist_stopgrad(logits, self.category_size, self.class_size)


class Decoder3D(nn.Module):
    def __init__(self, args, obs_channel=3):
        super(Decoder3D, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, 4096),
            nn.ELU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2),  # 4x4x256 -> 10x10x128
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),  # 10x10x128 -> 22x22x64
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),  # 22x22x64 -> 47x47x32
            nn.ELU(),
            nn.ConvTranspose2d(32, obs_channel, 4, stride=2),  # 47x47x32 -> 96x96x3
            nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False),
        )

    def forward(self, state, deterministic):
        pred = self.layers(torch.cat([state, deterministic], dim=-1))
        # pred (batch_size, obs_channel, 96, 96)
        m = Normal(pred, 1)
        dist = Independent(m, 3)
        return dist


class Decoder2D(nn.Module):
    def __init__(self, args, obs_size):
        super(Decoder2D, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, obs_size)
        )

    def forward(self, state, deterministic):
        pred = self.decoder(torch.cat([state, deterministic], dim=-1))
        dist = Normal(pred, 1)
        return dist


class RewardModel(nn.Module):
    def __init__(self, args):
        super(RewardModel, self).__init__()
        self.reward = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, 1),
        )

    def forward(self, deterministic, state):
        x = torch.cat([state, deterministic], dim=-1)
        dist = Normal(self.reward(x), 1)
        return dist


class DiscountModel(nn.Module):
    def __init__(self, args):
        super(DiscountModel, self).__init__()
        self.discount = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, 1),
        )

    def forward(self, deterministic, state):
        x = torch.cat([state, deterministic], dim=-1)
        dist = Bernoulli(logits=self.discount(x))
        return dist


class ActionContinuous(nn.Module):
    def __init__(self, args, action_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, 256),
            nn.ELU(),
            nn.Linear(256, 256)
        )
        self.mu = nn.Linear(256, action_dim)
        self.std = nn.Linear(256, action_dim)

    def forward(self, state, deterministic, training=True):
        x = self.seq(torch.cat([state, deterministic], dim=-1))
        mu = self.mu(x)
        std = self.std(x)
        std = F.softplus(std) + 1e-4

        if training:
            action_dist = Normal(mu, std)
            return action_dist
        else:
            action = torch.tanh(mu)
            return action


class ActionDiscrete(nn.Module):
    def __init__(self, args, action_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, 256),
            nn.ELU(),
            nn.Linear(256, 256)
        )
        self.logits = nn.Linear(256, action_dim)

    def forward(self, state, deterministic, training=True):
        x = self.seq(torch.cat([state, deterministic], dim=-1))
        logits = self.logits(x)

        if training:
            dist = OneHotCategorical(logits=logits)
            action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            action = torch.argmax(logits, dim=-1)
        return action


class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, 256),
            nn.ELU(),
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
        self.done = np.zeros((capacity, 1), dtype=np.int8)

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
