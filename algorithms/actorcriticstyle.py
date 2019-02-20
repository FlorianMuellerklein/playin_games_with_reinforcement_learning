import torch
import numpy as np

class ActorCriticStyle:
    '''
    Parent class for actor critic style algorithms
    
    Contains init the function for discounting rewards that both PPO and A2C will use
    '''

    def __init__(self, policy, optimizer, num_steps, num_envs, entropy_coef,
                 state_size, gamma, device, epochs=4, batch_size=16, clip=0.2):
        self.rollouts = Rollouts(num_steps, num_envs, state_size)
        self.policy = policy
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.gamma = gamma
        self.device = device
        self.ppo_epochs = epochs
        self.ppo_batch_size = batch_size
        self.ppo_clip = clip
        self.value_coef = 0.5
        self.entropy_coef = entropy_coef
        self.state_size = state_size
        self.actor_losses = []
        self.critic_losses = []
        self.entropy = []

    def discount_rewards(self, next_val, rewards, masks):
        returns = torch.zeros(rewards.size())
        r = next_val.to(self.device)

        return_idx = 0
        for step in reversed(range(len(rewards))):
            r = rewards[step].to(self.device) + r * self.gamma * masks[step].to(self.device)
            returns[step] = r

        return returns.view(-1, 1).to(self.device)

class Rollouts:
    '''
    Hold all of the info for training
    '''
    def __init__(self, num_steps, num_envs, state_size):
        self.state_size = state_size
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.states = torch.zeros((num_steps, num_envs, *state_size),)
        self.actions = torch.zeros((num_steps, num_envs))
        self.log_probs = torch.zeros((num_steps, num_envs))
        self.values = torch.zeros((num_steps, num_envs))
        self.rewards = torch.zeros((num_steps, num_envs))
        self.masks = torch.zeros((num_steps, num_envs))
        self.entropy = torch.zeros((num_steps, num_envs))
        self.entropy = 0.


    def insert(self, step, lp, e, v, r, d, a=None, s=None):
        if a is not None:
            self.actions[step] = a.int()
        if s is not None:
            self.states[step] = s.float()
        self.values[step] = v
        self.log_probs[step] = lp.float()
        self.rewards[step] = torch.from_numpy(r).float()
        self.masks[step] = torch.tensor(1. - d).float()
        self.entropy += e.mean()

    def reset(self):
        self.states = torch.zeros((self.num_steps, self.num_envs, *self.state_size))
        self.actions = torch.zeros((self.num_steps, self.num_envs))
        self.log_probs = torch.zeros((self.num_steps, self.num_envs))
        self.values = torch.zeros((self.num_steps, self.num_envs))
        self.rewards = torch.zeros((self.num_steps, self.num_envs))
        self.masks = torch.zeros((self.num_steps, self.num_envs))
        self.entropy = torch.zeros((self.num_steps, self.num_envs))
        self.entropy = 0.
        