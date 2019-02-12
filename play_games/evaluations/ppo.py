import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO(object):
    '''Proximal Policy Optimization'''
    def __init__(self, policy, optimizer, gamma, device, lr, epochs=4, clip=0.2):
        self.policy = policy
        self.old_policy = copy.deepcopy(policy)
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device
        self.epochs = epochs
        self.clip = clip
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.actor_losses = []
        self.critic_losses = []
        self.entropy = []

        # initialize old policy to be same as main for first epoch
        self.old_policy.load_state_dict(self.policy.state_dict())

    def update(self, states, actions, values, rewards, dones):
        # get discounted rewards
        discounted_rewards = self.discount_rewards(states, values, rewards, dones)

        # concate lists
        states = torch.cat(states)
        hiddens = torch.cat(hiddens)
        actions = torch.cat(actions)

        for e in range(self.epochs):

            # get probs, value preds and log probs
            action_probs, value_preds = self.policy(states.to(self.device))
            action_log_probs = action_probs.log_prob(actions)
            
            # get the old probs
            with torch.no_grad():
                old_action_probs, _ = self.old_policy(states.to(self.device))
                old_log_probs = old_action_probs.log_prob(actions)

            # get entropy
            entropy = action_probs.entropy()

            # get advantage
            advantage = (discounted_rewards - value_preds.detach())

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

        self.old_policy.load_state_dict(self.policy.state_dict())

    def discount_rewards(self, states, values, rewards, dones):
        # get discounted rewards
        rewards.reverse()
        discounted_rewards = []
        
        ## at terminal state
        if dones[-1] == True:
            next_return = 0
        else:
            # instead of starting at 0 we bootstrap with value prediction
            next_return = values[-1]
        
        # put predicted value in reward
        discounted_rewards.append(next_return)
        dones.reverse()
        for r in range(1, len(rewards)):
            if not dones[r]:
                current_return = rewards[r] + next_return * self.gamma
            else:
                current_return = 0
            discounted_rewards.append(current_return)
            next_return = current_return

        # put back in original order
        discounted_rewards.reverse()
        # convert to tensor
        discounted_rewards = torch.tensor(discounted_rewards).to(self.device).unsqueeze(1)
        #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        return discounted_rewards.float()