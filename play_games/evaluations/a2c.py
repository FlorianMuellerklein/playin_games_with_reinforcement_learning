import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class A2C(object):
    '''Advantage actor critic'''
    def __init__(self, policy, optimizer, gamma, device, lr):
        self.policy = policy
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device
        self.value_coef = 0.5
        self.entropy_coef = 0.001
        self.actor_losses = []
        self.critic_losses = []
        self.entropy = []

    def update(self, states, actions, rewards, dones):
        # get discounted rewards
        discounted_rewards = self.discount_rewards(states, rewards, dones)

        # concate lists
        states = torch.cat(states)
        actions = torch.cat(actions)

        # get probs, value preds and log probs
        action_probs, value_preds = self.policy(states.to(self.device))
        action_log_probs = action_probs.log_prob(actions)

        # get entropy
        entropy = action_probs.entropy()

        # get advantage
        advantage = (discounted_rewards - value_preds.detach())

        # do the policy loss
        actor_loss = (-action_log_probs * advantage).mean()
        # do the value loss
        value_loss = F.mse_loss(value_preds, discounted_rewards)
        # combine into total loss
        total_loss = (actor_loss + self.value_coef * value_loss - 
                      self.entropy_coef * entropy.mean())
          
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(value_loss.item())
        self.entropy.append(entropy.mean().item())

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

    def discount_rewards(self, states, rewards, dones):
        # get discounted rewards
        rewards.reverse()
        discounted_rewards = []
        
        ## at terminal state
        #if dones[-1] == True:
        #    next_return = 0
        #else:
            # the last return is our estimate (according to deepmind preso RL6: @ 1:16)
        #    with torch.no_grad():
        #        next_return = self.policy.sample_value(states[-1].to(self.device))
        
        # put predicted value in reward
        #discounted_rewards.append(next_return)
        #dones.reverse()
        R = 0
        for r in range(0, len(rewards)):
            #if not dones[r]:
            R = rewards[r] + R * self.gamma
            #else:
            #    current_return = 0
            discounted_rewards.append(R)
        
        # put back in original order
        discounted_rewards.reverse()
        # convert to tensor
        discounted_rewards = torch.tensor(discounted_rewards).to(self.device).unsqueeze(1)
        #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        return discounted_rewards.float()