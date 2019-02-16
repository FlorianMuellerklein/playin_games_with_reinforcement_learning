import torch

def discount_rewards(states, values, rewards, dones):
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