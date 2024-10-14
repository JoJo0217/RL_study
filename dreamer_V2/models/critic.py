import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, 1)
        )

    def forward(self, state, deterministic):
        x = torch.cat([state, deterministic], dim=-1)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        pred = self.seq(x)
        pred = pred.reshape(*shape[:-1], -1)
        dist = Normal(pred, 1)
        dist = Independent(dist, 1)
        return dist
