import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np


class QuantileMLP(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles=51, hidden_dim=256):
        super(QuantileMLP, self).__init__()
        self.num_quantiles = num_quantiles
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim * num_quantiles)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values.view(-1, self.action_dim, self.num_quantiles)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class QRDQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.frame_idx = 0
        self.num_quantiles = cfg.num_quantiles
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = QuantileMLP(state_dim, action_dim, num_quantiles=cfg.num_quantiles, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = QuantileMLP(state_dim, action_dim, num_quantiles=cfg.num_quantiles, hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                quantile_q_values = self.policy_net(state)
                mean_q_values = quantile_q_values.mean(dim=2)
                action = mean_q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float32).unsqueeze(1)
        
        quantiles = torch.linspace(0.0, 1.0, self.num_quantiles + 1)[1:].to(self.device) - 0.5 / self.num_quantiles

        # Get current quantile estimates
        quantile_q_values = self.policy_net(state_batch)
        quantile_q_values = quantile_q_values.gather(1, action_batch.repeat(1, 1, self.num_quantiles))

        # Get target quantile estimates
        with torch.no_grad():
            next_quantile_q_values = self.target_net(next_state_batch)
            next_mean_q_values = next_quantile_q_values.mean(dim=2)
            next_actions = next_mean_q_values.max(1)[1].unsqueeze(1).unsqueeze(1).repeat(1, 1, self.num_quantiles)
            next_quantile_q_values = next_quantile_q_values.gather(1, next_actions).squeeze(1)
            target_quantile_q_values = reward_batch + self.gamma * next_quantile_q_values * (1 - done_batch)
        
        # Compute loss
        diff = target_quantile_q_values.unsqueeze(1) - quantile_q_values
        loss = quantiles * diff.clamp(min=0) + (1 - quantiles) * (-diff).clamp(min=0)
        loss = loss.mean(dim=2).sum(dim=1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'qr_dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'qr_dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


