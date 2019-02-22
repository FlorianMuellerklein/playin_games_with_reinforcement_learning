import torch
import numpy as np

class ActorCriticStyle:
    '''
    Parent class for actor critic style algorithms
    
    Contains init the function for discounting rewards that both PPO and A2C will use
    '''

    def __init__(self, policy, optimizer, num_steps=2048, num_envs=None, 
                 state_size=None, entropy_coef=0.001, lmbda=0.95, gamma=0.99, 
                 device='cpu', recurrent=False, rnn_size=None):
        # training hyper params
        self.policy = policy
        self.optimizer = optimizer
        self.lmbda = lmbda
        self.gamma = gamma
        self.device = device
        self.value_coef = 0.5
        self.entropy_coef = entropy_coef

        # rollouts shape info
        self.state_size = state_size
        self.num_steps = num_steps
        self.num_envs = num_envs

        # rollout storage
        self.states = torch.zeros((num_steps, num_envs, *state_size), device=device)
        self.actions = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.masks = torch.zeros((num_steps, num_envs), device=device)

        # whether policy net is recurrent
        self.recurrent = recurrent
        self.rnn_size = rnn_size
        if self.recurrent:
            self.memory = torch.zeros((num_steps, num_envs, rnn_size), device=device)

        # keep track of training info
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_logs = []

    # discount function
    def discount_rewards(self, _val):
        returns = torch.zeros(self.rewards.size())

        _msk = torch.zeros((self.num_envs), device=self.device)
        gae = 0.
        for i in reversed(range(self.num_steps)):
            _val = self.values[i+1] if i < self.num_steps - 1 else _val
            _msk = self.masks[i+1] if i < self.num_steps -1 else _msk

            delta = self.rewards[i] + _val * self.gamma * _msk - self.values[i]
            gae = delta + self.gamma * self.lmbda * gae
            returns[i] = gae + self.values[i]

        return returns.transpose(0,1).reshape(-1).to(self.device)

    def insert(self, step, lp, s, a, v, r, d):
        self.actions[step] = a.int()
        self.states[step] = s.float()
        self.values[step] = v.float()
        self.log_probs[step] = lp.float()
        self.rewards[step] = torch.from_numpy(r).float()
        self.masks[step] = torch.tensor(1. - d).float()

    def reset(self):
        self.states = torch.zeros((self.num_steps, self.num_envs, 
                                   *self.state_size), device=self.device)
        self.actions = torch.zeros((self.num_steps, 
                                    self.num_envs), device=self.device)
        self.values = torch.zeros((self.num_steps, 
                                    self.num_envs), device=self.device)
        self.log_probs = torch.zeros((self.num_steps, 
                                      self.num_envs), device=self.device)
        self.rewards = torch.zeros((self.num_steps, 
                                    self.num_envs), device=self.device)
        self.masks = torch.zeros((self.num_steps, 
                                  self.num_envs), device=self.device)
        if self.recurrent:
            self.memory = torch.zeros((self.num_steps, 
                                       self.num_envs, 
                                       self.rnn_size), device=self.device)

    def update(self):
        pass

        