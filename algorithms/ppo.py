import time
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.actorcriticstyle import ActorCriticStyle

class PPO(ActorCriticStyle):
    '''Proximal Policy Optimization'''

    def __init__(self, policy, optimizer, num_steps=2048, num_envs=8, 
                 state_size=None, entropy_coef=0.01, lmbda=0.95, gamma=0.99, 
                 device='cpu', recurrent=False, rnn_size=None, 
                 epochs=4, batch_size=32, clip=0.2):
        '''Initialize specific variables for ppo'''
        super(PPO, self).__init__(policy, optimizer, num_steps, num_envs, 
                                  state_size, entropy_coef, lmbda, gamma, device, 
                                  recurrent, rnn_size)
        self.ppo_epochs = epochs
        self.ppo_batch_size = batch_size
        self.ppo_clip = clip

    def update(self, next_val, next_mask):
        # concate lists
        old_log_probs = self.log_probs.transpose(0,1).reshape(-1)
        actions = self.actions.transpose(0,1).reshape(-1)
        states = self.states.transpose(0,1).reshape(-1, *self.state_size)
        if self.recurrent:
            masks = self.masks.transpose(0,1).reshape(-1,1)
            memory = self.memory.transpose(0,1).reshape(-1, self.rnn_size)

        # get discounted rewards
        returns = self.discount_rewards(next_val, next_mask)

        for e in range(self.ppo_epochs):
            for strt in range(0, states.size(0), self.ppo_batch_size):
                end = strt + self.ppo_batch_size

                if not self.recurrent:
                    # get new probs, value preds and log probs
                    action_probs, value_preds = self.policy(states[strt:end])
                    action_log_probs = action_probs.log_prob(actions[strt:end])
                else:
                    # get new probs, value preds and log probs
                    action_probs, value_preds, h = self.policy(states[strt:end], 
                                                               memory[strt:end] * masks[strt:end])
                    action_log_probs = action_probs.log_prob(actions[strt:end])
                
                # get advantage
                advantage = (returns[strt:end] - value_preds)

                # do the policy loss
                ratio = torch.exp(action_log_probs - old_log_probs[strt:end])
                # get regular policy grad and clipped ratio
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1. - self.ppo_clip, 
                                           1. + self.ppo_clip) * advantage
                actor_loss = - torch.min(surr1, surr2).mean()
                # do the value loss
                value_loss = F.mse_loss(value_preds, returns[strt:end])

                # combine into total loss
                total_loss = (actor_loss + self.value_coef * value_loss - 
                              self.entropy_coef * action_probs.entropy().mean())

                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(value_loss.item())
                self.entropy_logs.append(action_probs.entropy().mean().item())

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 40.)
                self.optimizer.step()

                if self.recurrent:
                    memory[strt+1:end+1] = h[:memory[strt+1:end+1].size(0),:].detach()

        self.reset()
