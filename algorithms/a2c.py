import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.actorcriticstyle import *

class A2C(ActorCriticStyle):
    '''Advantage actor critic'''
            
    def update(self, next_val, next_mask):
        states = self.states.transpose(0,1).reshape(-1,*self.state_size)
        actions = self.actions.transpose(0,1).reshape(-1)

        if not self.recurrent:
            # get new probs, value preds and log probs
            action_probs, value_preds = self.policy(states)
            log_probs = action_probs.log_prob(actions)
        else:
            memory = self.memory.transpose(0,1).reshape(-1,self.rnn_size)
            masks = self.masks.transpose(0,1).reshape(-1,1)
            # get new probs, value preds and log probs
            action_probs, value_preds, _ = self.policy(states, memory * masks)
            log_probs = action_probs.log_prob(actions)

        entropy = action_probs.entropy().mean()

        # get discounted rewards
        returns = self.discount_rewards(next_val, next_mask)

        # get advantage
        advantage = (returns - value_preds)
        
        # do the policy loss
        actor_loss = -(log_probs * advantage).mean()
        # do the value loss
        value_loss = F.mse_loss(value_preds, returns)

        # combine into total loss
        total_loss = (actor_loss + self.value_coef * value_loss - 
                      self.entropy_coef * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.policy.parameters()) ** 0.5
        nn.utils.clip_grad_norm_(self.policy.parameters(), 40.)
        self.optimizer.step()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(value_loss.item())
        self.entropy_logs.append(entropy.item())
        self.grad_norms.append(grad_norm.item())

        self.reset()
