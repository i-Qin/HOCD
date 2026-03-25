import os
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ReplayBuffer:
    def __init__(self, config):
        self._load_config(config)
        self._setup()

    def _load_config(self, config):
        self._buffer_capacity = config.buffer_capacity
        self._state_dim = config.state_dim
        self._action_dim = config.action_dim
    
    def _setup(self):
        store_type = np.dtype([('state', np.float32, (self._state_dim,)),
                               ('action', np.float32, (self._action_dim,)),
                               ('action_logprob', np.float32),
                               ('next_state', np.float32, (self._state_dim,)),
                               ('reward', np.float32),
                               ('done', np.float32),
                               ])
        self.buffer = np.empty(self._buffer_capacity, store_type)
        self.count = 0

    def store(self, value):
        self.buffer[self.count] = value
        self.count += 1
        if self.count == self._buffer_capacity:
            self.count = 0
            return True
        else:
            return False

    def numpy_to_tensor(self, device):
        state = torch.tensor(self.buffer['state'], dtype=torch.float32).to(device)
        action = torch.tensor(self.buffer['action'], dtype=torch.float32).to(device)
        action_logprob = torch.tensor(self.buffer['action_logprob'], dtype=torch.float32).unsqueeze(1).to(device)
        next_state = torch.tensor(self.buffer['next_state'], dtype=torch.float32).to(device)
        reward = torch.tensor(self.buffer['reward'], dtype=torch.float32).unsqueeze(1).to(device)
        done = torch.tensor(self.buffer['done'], dtype=torch.float32).unsqueeze(1).to(device)
        return state, action, action_logprob, next_state, reward, done


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_layer = nn.Linear(hidden_dim, action_dim)
        self.beta_layer = nn.Linear(hidden_dim, action_dim)
        self.activate_func = nn.ReLU()

    def forward(self, x):
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        alpha = F.softplus(self.alpha_layer(x)) + 1.0
        beta = F.softplus(self.beta_layer(x)) + 1.0
        return alpha, beta

    def get_dist(self, x):
        alpha, beta = self.forward(x)
        dist = Beta(alpha, beta)
        return dist


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activate_func = nn.ReLU()

    def forward(self, x):
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        v = self.fc3(x)
        return v


class PPO:
    def __init__(self, config):
        self._load_config(config)
        self._setup()

    def _load_config(self, config):
        self._device = config.device
        self._train = config.train
        self._lr_a = config.lr_a
        self._lr_c = config.lr_c
        self._hidden_dim = config.hidden_dim
        self._state_dim = config.state_dim
        self._action_dim = config.action_dim
        self._batch_size = config.batch_size
        self._buffer_capacity = config.buffer_capacity
        self._gamma = config.gamma
        self._gae_lambda = config.gae_lambda
        self._clip_param = config.clip_param
        self._entropy_coef = config.entropy_coef
        self._ppo_epoch = config.ppo_epoch
        if not self._train:
            self._actor_path = config.model.actor
            self._critic_path = config.model.critic

    def _setup(self):
        self.actor = ActorNet(self._state_dim, self._action_dim, self._hidden_dim).to(self._device)
        self.critic = CriticNet(self._state_dim, self._hidden_dim).to(self._device)

        if not self._train:
            self.actor.load_state_dict(torch.load(self._actor_path))
            self.critic.load_state_dict(torch.load(self._critic_path))
            print(f"Loading model...Actor:{self._actor_path} Critic:{self._critic_path}")

        self.optimizer_actor = Adam(self.actor.parameters(), lr=float(self._lr_a))
        self.optimizer_critic = Adam(self.critic.parameters(), lr=float(self._lr_c))

    def choose_action(self, state, train):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            if not train:
                dist = self.actor.get_dist(state)
                action = dist.mean
                return action.squeeze(0).cpu().numpy()
            # Beta
            dist = self.actor.get_dist(state)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(1, keepdim=True)
        return action.squeeze(0).cpu().numpy(), action_logprob.squeeze(0).cpu().numpy()

    def update(self, buffer):
        state, action, action_logprob, next_state, reward, done = buffer.numpy_to_tensor(self._device)

        with torch.no_grad():
            v_state = self.critic(state)
            v_next_state = self.critic(next_state)
            v_target = reward + self._gamma * v_next_state * (1.0 - done)
            td_error = v_target - v_state

            advantage = []
            gae = 0
            for delta in reversed(td_error.detach().cpu().numpy().flatten()):
                gae = delta + self._gamma * self._gae_lambda * gae
                advantage.insert(0, gae)
            advantage = torch.tensor(advantage, dtype=torch.float32).unsqueeze(1).to(self._device)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        for _ in range(self._ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self._buffer_capacity)), self._batch_size, False):
                new_dist = self.actor.get_dist(state[index])
                new_dist_entropy = new_dist.entropy().sum(1, keepdim=True)
                new_logprob = new_dist.log_prob(action[index])

                ratio = torch.exp(new_logprob - action_logprob[index])
                sur1 = ratio * advantage[index]
                sur2 = torch.clamp(ratio, 1-self._clip_param, 1+self._clip_param) * advantage[index]
                loss_policy = torch.min(sur1, sur2)
                loss_policy_entropy = self._entropy_coef * new_dist_entropy
                loss_actor = -(loss_policy + loss_policy_entropy).mean()

                vs = self.critic(state[index])
                loss_critic = F.mse_loss(v_target[index], vs)

                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                self.optimizer_critic.step()

    def save_model(self, save_path, episode):
        now = datetime.now()
        time_string = now.strftime("%m-%d")
        path = os.path.join(save_path, time_string)

        if not os.path.exists(path): 
            os.makedirs(path)

        torch.save(self.actor.state_dict(), os.path.join(path, 'actor_model-' + str(episode) + '.pkl'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic_model-' + str(episode) + '.pkl'))

