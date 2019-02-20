import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.actorcriticstyle import *

class A2C(ActorCriticStyle):
    '''Advantage actor critic'''
            
    def update(self, next_val):
        log_probs = self.rollouts.log_probs.view(-1).to(self.device)
        values = self.rollouts.values.view(-1).to(self.device)
    
        # get discounted rewards
        returns = self.discount_rewards(next_val, 
                                        self.rollouts.rewards, 
                                        self.rollouts.masks)

        # get advantage
        advantage = (returns - values)
        
        # do the policy loss
        actor_loss = - (log_probs * advantage.detach()).mean()
        # do the value loss
        value_loss = F.mse_loss(values, returns.detach())

        # combine into total loss
        total_loss = (actor_loss + self.value_coef * value_loss - 
                      self.entropy_coef * self.rollouts.entropy.item())
          
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(value_loss.item())
        self.entropy.append(self.rollouts.entropy.item())

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        self.rollouts.reset()
