import torch
import numpy as np

class ActorCriticStyle:
    '''
    Parent class for actor critic style algorithms
    
    Contains init the function for discounting rewards that both PPO and A2C will use
    '''

    def __init__(self, policy, optimizer, num_steps, num_envs, entropy_coef,
                 state_size, gamma, device, epochs=4, clip=0.2):
        self.rollouts = Rollouts(num_steps, num_envs)
        self.policy = policy
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.gamma = gamma
        self.device = device
        self.ppo_epochs = epochs
        self.ppo_clip = clip
        self.value_coef = 0.5
        self.entropy_coef = entropy_coef
        self.state_size = state_size
        self.actor_losses = []
        self.critic_losses = []
        self.entropy = []

    def discount_rewards(self, next_val, rewards, masks):
        returns = []
        r = next_val.to(self.device)

        for step in reversed(range(len(rewards))):
            r = rewards[step].to(self.device) + r * self.gamma * masks[step].to(self.device)
            returns.insert(0, r)

        return torch.cat(returns)

class Rollouts:
    '''
    Hold all of the info for training
    '''
    def __init__(self, num_steps, num_envs):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.entropy = 0.


    def insert(self, lp, e, v, r, d, a=None, s=None):
        if a is not None:
            self.actions.append(a.unsqueeze(0).int())
        if s is not None:
            self.states.append(s.unsqueeze(0).float())
        self.values.append(v)
        self.log_probs.append(lp.float())
        self.rewards.append(torch.from_numpy(r).unsqueeze(1).float())
        self.masks.append(torch.tensor(1. - d).unsqueeze(1).float())
        self.entropy += e.mean()

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.entropy = 0.
        