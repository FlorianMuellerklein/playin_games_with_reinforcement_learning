import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.actorcriticstyle import *

class A2C(ActorCriticStyle):
    '''Advantage actor critic'''
            
    def update(self, next_val):
        # get discounted rewards
        discounted_rewards = self.discount_rewards(next_val, 
                                                   self.rollouts.rewards, 
                                                   self.rollouts.masks)

        # get advantage
        advantage = (discounted_rewards - self.rollouts.values.detach())
        
        # do the policy loss
        actor_loss = (-self.rollouts.log_probs * advantage).mean()
        # do the value loss
        value_loss = F.mse_loss(self.rollouts.values, discounted_rewards)
        # combine into total loss
        total_loss = (actor_loss + self.value_coef * value_loss - 
                      self.entropy_coef * self.rollouts.entropy.mean().detach())
          
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(value_loss.item())
        self.entropy.append(self.rollouts.entropy.mean().item())

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
        self.optimizer.step()
