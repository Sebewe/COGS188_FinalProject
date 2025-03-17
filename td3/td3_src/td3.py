# td3_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from td3_src.network import Actor, Critic
from copy import deepcopy

class TD3:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 act_limit,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 device='cpu'):

        self.act_limit = act_limit
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = device

        # Main networks
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target = deepcopy(self.actor)

        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic1_target = deepcopy(self.critic1)

        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.critic2_target = deepcopy(self.critic2)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Count of updates to delay policy update
        self.total_it = 0

    def select_action(self, obs, noise_scale=0.0):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]
        if noise_scale > 0:
            action += noise_scale * np.random.randn(len(action))
        return np.clip(action, -self.act_limit, self.act_limit)

    def update(self, replay_buffer, batch_size=256):
        self.total_it += 1
        batch = replay_buffer.sample_batch(batch_size)

        obs = batch['obs'].to(self.device)
        obs2 = batch['obs2'].to(self.device)
        act = batch['act'].to(self.device)
        rew = batch['rew'].unsqueeze(1).to(self.device)
        done = batch['done'].unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Target action smoothing
            noise = (
                torch.randn_like(act) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(obs2) + noise).clamp(-self.act_limit, self.act_limit)

            # Target Q-value
            target_q1 = self.critic1_target(obs2, next_action)
            target_q2 = self.critic2_target(obs2, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = rew + self.gamma * (1 - done) * target_q

        # Critic losses
        current_q1 = self.critic1(obs, act)
        current_q2 = self.critic2(obs, act)
        critic1_loss = nn.MSELoss()(current_q1, target)
        critic2_loss = nn.MSELoss()(current_q2, target)

        # Optimize critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Polyak averaging
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)

    def _soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
