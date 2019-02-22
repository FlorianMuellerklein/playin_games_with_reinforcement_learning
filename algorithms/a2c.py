import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.actorcriticstyle import *

class A2C(ActorCriticStyle):
    '''Advantage actor critic'''
            
    def update(self, next_val):
        states = self.states.transpose(0,1).reshape(-1, *self.state_size)
        actions = self.actions.transpose(0,1).reshape(-1)

        if not self.recurrent:
            # get new probs, value preds and log probs
            action_probs, value_preds = self.policy(states)
            log_probs = action_probs.log_prob(actions)
        else:
            memory = self.memory.transpose(0,1).reshape(-1, self.rnn_size)
            # get new probs, value preds and log probs
            action_probs, value_preds, _ = self.policy(states, memory.to(self.device))
            log_probs = action_probs.log_prob(actions)

        entropy = action_probs.entropy().mean()

        # get discounted rewards
        returns = self.discount_rewards(next_val)

        # get advantage
        advantage = (returns - value_preds)
        
        # do the policy loss
        actor_loss = - (log_probs * advantage.detach()).mean()
        # do the value loss
        value_loss = F.mse_loss(value_preds, returns.detach())

        # combine into total loss
        total_loss = (actor_loss + self.value_coef * value_loss - 
                      self.entropy_coef * entropy)
          
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(value_loss.item())
        self.entropy_logs.append(entropy.item())

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        self.reset()
