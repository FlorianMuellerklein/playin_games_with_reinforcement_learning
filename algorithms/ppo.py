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
        old_log_probs = self.rollouts.log_probs.view(-1, 1).to(self.device)
        states = self.rollouts.states.view(-1, *self.state_size)
        actions = self.rollouts.actions.view(-1, 1).to(self.device)

        # get discounted rewards
        returns = self.discount_rewards(next_val, 
                                        self.rollouts.rewards, 
                                        self.rollouts.masks)

        for e in range(self.ppo_epochs):
            for i in range((states.size(0) + self.ppo_batch_size - 1) // self.ppo_batch_size):
                sl = slice(i * self.ppo_batch_size, (i + 1) * self.ppo_batch_size)
                # get new probs, value preds and log probs
                action_probs, value_preds = self.policy(states[sl].to(self.device))
                action_log_probs = action_probs.log_prob(actions[sl].to(self.device))
                
                # get advantage
                advantage = (returns[sl] - value_preds)

                # do the policy loss
                ratio = (action_log_probs - old_log_probs[sl].detach()).exp()
                # get regular policy grad and clipped ratio
                pg = ratio * advantage.detach()
                clipped = torch.clamp(ratio, 1. - self.ppo_clip, 
                                    1. + self.ppo_clip) * advantage.detach()
                actor_loss = - torch.min(pg, clipped).mean()
                # do the value loss
                value_loss = F.mse_loss(value_preds, returns[sl].detach())
                # combine into total loss
                total_loss = (actor_loss + self.value_coef * value_loss - 
                            self.entropy_coef * action_probs.entropy().mean())

                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(value_loss.item())
                self.entropy.append(action_probs.entropy().sum(-1).mean().item())

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.rollouts.reset()
