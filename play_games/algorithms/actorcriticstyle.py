import torch
import numpy as np

class ActorCriticStyle:
    '''
    Parent class for actor critic style algorithms
    
    Contains init the function for discounting rewards that both PPO and A2C will use
    '''

    def __init__(self, policy, optimizer, num_steps, num_envs, 
                 state_size, gamma, device, lr, epochs=4, clip=0.2):
        self.policy = policy
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.rollouts = Rollouts(num_steps, num_envs, state_size)
        self.gamma = gamma
        self.device = device
        self.ppo_epochs = epochs
        self.ppo_clip = clip
        self.value_coef = 0.5
        self.entropy_coef = 0.001
        self.actor_losses = []
        self.critic_losses = []
        self.entropy = []

        # put rollouts on the correct device
        self.rollouts.log_probs.to(device)
        self.rollouts.values.to(device)
        self.rollouts.rewards.to(device)
        self.rollouts.masks.to(device)
        self.rollouts.entropy.to(device)

    def discount_rewards(self, next_val, rewards, masks):
        returns = torch.zeros(rewards.size())
        # get discounted rewards, replacing the values in rewards
        r = next_val.to(self.device)
        #returns[-1] = r

        for step in reversed(range(rewards.size(0))):
            r_ = rewards[step].to(self.device) + r * self.gamma * masks[step].to(self.device)
            returns[step] = r_
            r = r_

        return returns

class Rollouts:
    '''
    Hold all of the info for training
    '''
    def __init__(self, num_steps, num_envs, state_size):
        #self.states = torch.zeros((num_steps, num_envs, *state_size)).float()
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.actions = torch.zeros((num_steps, num_envs, 1)).int()
        self.log_probs = torch.zeros((num_steps, num_envs, 1)).float()
        self.values = torch.zeros((num_steps, num_envs, 1)).float()
        self.rewards = torch.zeros((num_steps, num_envs, 1)).float()
        self.returns = torch.zeros((num_steps, num_envs, 1)).float()
        self.masks = torch.zeros((num_steps, num_envs, 1)).float()
        self.entropy = torch.zeros((num_steps, num_envs, 1)).float()

    def insert(self, step, lp, e, v, r, d, a=None):
        if a is not None:
            self.actions[step] = a
        self.values[step] = v
        self.log_probs[step] = lp.unsqueeze(-1)
        self.rewards[step] = torch.from_numpy(r).unsqueeze(-1)
        self.masks[step] = torch.tensor(1. - d).unsqueeze(-1)
        self.entropy[step] = e.unsqueeze(-1)

    def reset(self):
        self.actions = torch.zeros((self.num_steps, self.num_envs, 1)).int()
        self.log_probs = torch.zeros((self.num_steps, self.num_envs, 1)).float()
        self.values = torch.zeros((self.num_steps, self.num_envs, 1)).float()
        self.rewards = torch.zeros((self.num_steps, self.num_envs, 1)).float()
        self.returns = torch.zeros((self.num_steps, self.num_envs, 1)).float()
        self.masks = torch.zeros((self.num_steps, self.num_envs, 1)).float()
        self.entropy = torch.zeros((self.num_steps, self.num_envs, 1)).float()
        