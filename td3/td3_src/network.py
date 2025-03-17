import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_dim),
            nn.Tanh()
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))
