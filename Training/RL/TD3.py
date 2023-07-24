import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = self.l1(state)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = F.softmax(x, dim=1)
        return x


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, user_num):
        super(Q_Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        state = torch.cat((state, action))

        x1 = self.l1(state)
        x1 = F.relu(x1)
        x1 = self.l2(x1)
        x1 = F.relu(x1)
        x1 = self.l3(x1)

        x2 = self.l1(state)
        x2 = F.relu(x2)
        x2 = self.l2(x2)
        x2 = F.relu(x2)
        x2 = self.l3(x2)
        return x1, x2

    def Q1(self, state, action):
        state = torch.cat((state, action))

        x1 = self.l1(state)
        x1 = F.relu(x1)
        x1 = self.l2(x1)
        x1 = F.relu(x1)
        x1 = self.l3(x1)
        return x1


class TD3(object):
    def __init__(
            self,
            env_with_Dead,
            state_dim,
            action_dim,
            max_action,
            user_num,
            train_path,
            gamma=0.99,
            net_width=128,
            a_lr=1e-4,
            c_lr=1e-4,
            Q_batchsize=256,
            delay_freq=1,
            policy_noise_std=0.2
    ):
        self.writer = SummaryWriter(train_path)
        max_action = torch.tensor(max_action, dtype=torch.float32).to(device)
        self.actor = Actor(state_dim, action_dim, net_width, max_action, user_num).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_critic = Q_Critic(state_dim, action_dim, net_width, user_num).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        self.env_with_Dead = env_with_Dead
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.policy_noise_std = policy_noise_std
        self.policy_noise = policy_noise_std * max_action
        self.noise_clip = 0.5 * max_action
        self.tau = 0.005
        self.Q_batchsize = Q_batchsize
        self.delay_counter = -1
        self.delay_freq = delay_freq
        self.q_iteration = 0
        self.a_iteration = 0

    def select_action(self, state):  # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            a = self.actor(state)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer):
        self.delay_counter = self.delay_counter + 1
        with torch.no_grad():
            s, a, r, s_prime, dead_mask = replay_buffer.sample(self.Q_batchsize)
            noise = torch.max(-self.noise_clip, torch.min(torch.randn_like(a) * self.policy_noise, self.noise_clip))
            smoothed_target_a = torch.max(-self.max_action, torch.min(
                self.actor_target(s_prime) + noise, self.max_action))

        # Compute the target Q value
        target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
        target_Q = torch.min(target_Q1, target_Q2)
        '''DEAD OR NOT'''
        if self.env_with_Dead:
            target_Q = r + (1 - dead_mask) * self.gamma * target_Q  # env with dead
        else:
            target_Q = r + self.gamma * target_Q  # env without dead

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        # Compute critic loss
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.writer.add_scalar('q_loss', q_loss, self.q_iteration)
        # Optimize the q_critic
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()
        self.q_iteration += 1
        if self.delay_counter == self.delay_freq:
            # Update Actor
            a_loss = -self.q_critic.Q1(s, self.actor(s)).mean()
            self.writer.add_scalar('a_loss', a_loss, self.a_iteration)
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()
            self.a_iteration += 1
            # Update the frozen target models
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.delay_counter = -1

    def save(self, episode, model_path):
        torch.save(self.actor.state_dict(), model_path + "td3_actor{}.pth".format(episode))
        torch.save(self.q_critic.state_dict(), model_path + "td3_q_critic{}.pth".format(episode))

    def load(self, episode):

        self.actor.load_state_dict(torch.load("td3_actor{}.pth".format(episode)))
        self.q_critic.load_state_dict(torch.load("td3_q_critic{}.pth".format(episode)))


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, dead):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.dead[ind]).to(self.device)
        )
