import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, n_states, n_hidden):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class PPO:
    def __init__(self, n_states, n_hidden, n_actions, actor_lr, critic_lr,
                 _lambda, epochs, epsilon, gamma, device):
        self.device = device

        self.actor = Actor(n_states, n_hidden, n_actions).to(device)
        self.critic = Actor(n_states, n_hidden).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self._lambda = _lambda # GAE advantages estimation factor
        self.epochs = epochs
        self.epsilon = epsilon

    def take_action(self, state):
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        probs = self.actor(state)
        actions = torch.distributions.Categorical(probs)
        action = actions.sample().item()
        return action



