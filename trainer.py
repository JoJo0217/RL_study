import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state = np.array(state)
        next_state = np.array(next_state)
        return torch.tensor(state, device=self.device), torch.tensor(action, device=self.device).view(-1, 1), \
            torch.tensor(reward, device=self.device).view(-1, 1), torch.tensor(next_state, device=self.device), \
            torch.tensor(done, device=self.device).view(-1, 1)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, data_len, device):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.data = deque(maxlen=data_len)
        self.gamma = 0.99
        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.buffer = ReplayBuffer(data_len, device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            state = state.to(self.device)
            with torch.no_grad():
                return torch.argmax(self.forward(state)).item()

    def input_data(self, transition):
        # data는 (state, action, reward, next_state, done)으로 이루어진다.
        self.buffer.push(transition)

    def train(self, batch_size, target_net):
        if len(self.buffer) < batch_size:
            return
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.buffer.sample(
            batch_size)

        q = self.forward(batch_state)
        q = q.gather(1, batch_action)

        max_q = target_net.forward(batch_next_state).detach().max(1)[0].view(-1, 1)
        target = batch_reward + (1 - batch_done) * self.gamma * max_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
