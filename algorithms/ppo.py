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
        # get discounted rewards
        discounted_rewards = self.discount_rewards(next_val, 
                                                   self.rollouts.rewards, 
                                                   self.rollouts.masks)

        # concate lists
        old_log_probs = self.rollouts.log_probs.detach()

        for e in range(self.epochs):

            # get probs, value preds and log probs
            action_probs, value_preds = self.policy(states.to(self.device))
            action_log_probs = action_probs.log_prob(self.rollouts.actions)
            
            # get advantage
            advantage = (discounted_rewards - self.rollouts.values.detach())

            # do the policy loss
            ratio = (action_log_probs - old_log_probs).exp()
            pg = ratio * advantage
            clipped = torch.clamp(ratio, 1. - self.clip, 1. + self.clip) * advantage
            actor_loss = -torch.min(pg, clipped).mean()
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
