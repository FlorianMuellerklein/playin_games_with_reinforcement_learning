import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.actorcriticstyle import ActorCriticStyle

class PPO(ActorCriticStyle):
    '''Proximal Policy Optimization'''

    def update(self, next_val):
        # concate lists
        old_log_probs = torch.cat(self.rollouts.log_probs)
        states = torch.cat(self.rollouts.states).view(-1, *self.state_size).detach()
        actions = torch.cat(self.rollouts.actions).view(-1).detach()

        # get discounted rewards
        discounted_rewards = self.discount_rewards(next_val, 
                                                   self.rollouts.rewards, 
                                                   self.rollouts.masks)

        for e in range(self.ppo_epochs):
            
            # get new probs, value preds and log probs
            action_probs, value_preds = self.policy(states.to(self.device))
            action_log_probs = action_probs.log_prob(actions.to(self.device))
            
            # get advantage
            advantage = (discounted_rewards - value_preds)

            # do the policy loss
            ratio = (action_log_probs - old_log_probs.detach()).exp()
            # get regular policy grad and clipped ratio
            pg = ratio * advantage.detach()
            clipped = torch.clamp(ratio, 1. - self.ppo_clip, 
                                  1. + self.ppo_clip) * advantage.detach()
            actor_loss = - torch.min(pg, clipped).mean()
            # do the value loss
            value_loss = F.mse_loss(value_preds, discounted_rewards.detach())
            # combine into total loss
            total_loss = (actor_loss + self.value_coef * value_loss - 
                          self.entropy_coef * self.rollouts.entropy)

            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(value_loss.item())
            self.entropy.append(self.rollouts.entropy)

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            self.rollouts.reset()
