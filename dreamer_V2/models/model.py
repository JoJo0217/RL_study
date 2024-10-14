import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal, Independent, Bernoulli, kl_divergence, OneHotCategoricalStraightThrough


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


def get_categorical_state(logits, categorical_size, class_size):
    shape = logits.shape
    logit = torch.reshape(logits, shape=[*shape[:-1], categorical_size, class_size])
    dist = OneHotCategorical(logits=logit)
    stoch = dist.sample() + dist.probs - dist.probs.detach()

    dist = Independent(OneHotCategoricalStraightThrough(logits=logit), 1)
    return dist, torch.flatten(stoch, start_dim=-2, end_dim=-1)


def get_dist_stopgrad(logits, categorical_size, class_size):
    logits = logits.detach()
    shape = logits.shape
    logit = torch.reshape(logits, shape=[*shape[:-1], categorical_size, class_size])
    dist = OneHotCategorical(logits=logit)
    stoch = dist.sample() + dist.probs - dist.probs.detach()

    dist = Independent(OneHotCategoricalStraightThrough(logits=logit), 1)
    return dist, torch.flatten(stoch, start_dim=-2, end_dim=-1)


class RepresentationModel(nn.Module):
    def __init__(self, args):
        super(RepresentationModel, self).__init__()
        self.args = args
        self.state_size = args.state_size
        self.category_size = args.categorical_size
        self.class_size = args.class_size
        self.MLP = nn.Sequential(
            nn.Linear(args.deterministic_size + args.observation_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, 2 * self.state_size if args.rssm_continue else self.state_size),
        )

    def forward(self, hidden, obs):
        x = torch.cat([hidden, obs], dim=-1)
        logits = self.MLP(x)
        if self.args.rssm_continue:
            mean, std = torch.chunk(logits, 2, dim=-1)
            std = F.softplus(std) + 0.1
            dist = Normal(mean, std)
            return dist, dist.rsample()
        else:
            return get_categorical_state(logits, self.category_size, self.class_size)

    def stop_grad(self, hidden, obs):
        x = torch.cat([hidden, obs], dim=-1)
        logits = self.MLP(x)
        if self.args.rssm_continue:
            logits = logits.detach()
            mean, std = torch.chunk(logits, 2, dim=-1)
            std = F.softplus(std) + 0.1
            dist = Normal(mean, std)
            return dist, dist.rsample()
        else:
            return get_dist_stopgrad(logits, self.category_size, self.class_size)


class TransitionModel(nn.Module):
    def __init__(self, args):
        super(TransitionModel, self).__init__()
        self.args = args
        self.state_size = args.state_size
        self.category_size = args.categorical_size
        self.class_size = args.class_size
        self.MLP = nn.Sequential(
            nn.Linear(args.deterministic_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, 2 * args.state_size if args.rssm_continue else args.state_size),
        )

    def forward(self, hidden):
        logits = self.MLP(hidden)
        if self.args.rssm_continue:
            mean, std = torch.chunk(logits, 2, dim=-1)
            std = F.softplus(std) + 0.1
            dist = Normal(mean, std)
            return dist, dist.rsample()
        else:
            return get_categorical_state(logits, self.category_size, self.class_size)

    def stop_grad(self, hidden):
        logits = self.MLP(hidden)
        if self.args.rssm_continue:
            logits = logits.detach()
            mean, std = torch.chunk(logits, 2, dim=-1)
            std = F.softplus(std) + 0.1
            dist = Normal(mean, std)
            return dist, dist.rsample()
        else:
            return get_dist_stopgrad(logits, self.category_size, self.class_size)


class RewardModel(nn.Module):
    def __init__(self, args):
        super(RewardModel, self).__init__()
        self.reward = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, 1),
        )

    def forward(self, state, deterministic):
        x = torch.cat([state, deterministic], dim=-1)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        pred = self.reward(x)
        pred = pred.reshape(*shape[:-1], -1)
        dist = Normal(pred, 1)
        dist = Independent(dist, 1)
        return dist


class DiscountModel(nn.Module):
    def __init__(self, args):
        super(DiscountModel, self).__init__()
        self.discount = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, 1),
        )

    def forward(self, state, deterministic):
        x = torch.cat([state, deterministic], dim=-1)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        pred = self.discount(x)
        pred = pred.reshape(*shape[:-1], -1)
        dist = Bernoulli(logits=pred)
        return dist
