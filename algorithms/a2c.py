import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from algorithms.actorcriticstyle import *

import matplotlib.pyplot as plt

def viz_state(state):
    fix, ax = plt.subplots(4)
    for idx in range(state.shape[0]):
        ax[idx].imshow(state[idx,:,:].squeeze())
    plt.show()

class A2C(ActorCriticStyle):
    '''Advantage actor critic'''
            
    def update(self, next_val, next_mask):
        self.policy.train()
        states = self.states[:-1].reshape(-1,*self.state_size)
        actions = self.actions.reshape(-1,1).long()

        if not self.recurrent:
            # get new probs, value preds and log probs
            action_probs, value_preds = self.policy(states)
        else:
            memory = self.memory.reshape(-1,self.rnn_size)
            masks = self.masks.reshape(-1,1)
            # get new probs, value preds and log probs
            action_probs, value_preds, _ = self.policy(states, memory * masks)

        # turn into categorical
        action_probs = Categorical(action_probs)
        # get log probs
        log_probs = action_probs.log_prob(actions.squeeze())

        entropy = action_probs.entropy().mean()

        # get discounted rewards
        self.discount_rewards(next_val, next_mask)

        # get advantage
        value_preds = value_preds.view(self.num_steps, self.num_envs)
        log_probs = log_probs.view(self.num_steps, self.num_envs)
        advantage = (self.returns[:-1] - value_preds)
        
        # do the policy loss
        actor_loss = -(log_probs * advantage.detach()).mean()
        # do the value loss
        value_loss = advantage.pow(2).mean()

        self.optimizer.zero_grad()
        # combine into total loss
        total_loss = (actor_loss + self.value_coef * value_loss - 
                      self.entropy_coef * entropy)

        total_loss.backward()
        grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.policy.parameters()) ** 0.5
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(value_loss.item())
        self.entropy_logs.append(entropy.item())
        self.grad_norms.append(grad_norm.item())

        self.reset()
