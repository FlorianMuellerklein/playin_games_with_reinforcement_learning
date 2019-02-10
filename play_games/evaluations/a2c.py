import torch
import torch.nn as nn
import torch.nn.functional as F

class A2C(object):
    '''Advantage actor critic'''
    def __init__(self, policy, optimizer, gamma, device, lr):
        self.policy = policy
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device

    def update(self, states, hiddens, actions, rewards, dones):
        # get discounted rewards
        discounted_rewards = self.discount_rewards(states, hiddens, rewards, dones)

        # concate lists
        states = torch.cat(states)
        hiddens = torch.cat(hiddens)
        actions = torch.cat(actions)

        # get probs, value preds and log probs
        action_probs, value_preds, _ = self.policy(states.to(self.device), 
                                                  hiddens.to(self.device))
        action_log_probs = action_probs.log().gather(1, actions)

        # get entropy
        entropy = (action_probs * action_probs.log()).sum(1)

        # get advantage
        advantage = (discounted_rewards - value_preds)

        # do the policy loss
        actor_loss = -(action_log_probs * advantage.detach()).mean()
        # do the value loss
        value_loss = F.mse_loss(discounted_rewards, value_preds)
        # combine into total loss
        total_loss = 0.5 * value_loss + actor_loss - 0.0001 * entropy.mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

    def discount_rewards(self, states, hiddens, rewards, dones):
        # get discounted rewards
        rewards.reverse()
        discounted_rewards = []
        
        # at terminal state
        if dones[-1] == True:
            next_return = 0
        else:
            # the last return is our estimate (according to deepmind preso RL6: @ 1:16)
            with torch.no_grad():
                next_return, _ = self.policy.sample_value(states[-1].to(self.device), 
                                                          hiddens[-1].to(self.device))
        
        # put predicted value in reward
        discounted_rewards.append(next_return)
        dones.reverse()

        for r in range(1, len(rewards)):
            if not dones[r]:
                current_return = rewards[r] + next_return * self.gamma
            else:
                current_return = 0
            discounted_rewards.append(current_return)
        
        # put back in original order
        discounted_rewards.reverse()
        # convert to tensor
        discounted_rewards = torch.tensor(discounted_rewards).to(self.device).unsqueeze(1)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        return discounted_rewards.float()