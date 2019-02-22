# playin_games_with_reinforcement_learning

I purchased the new edition of the [Sutton and Barto Reinforcement Learning book](http://incompleteideas.net/book/the-book-2nd.html) so I couldn't help myself. 

This repo implements Synchronous Advantage Actor Critic (A2C) and [PPO](https://arxiv.org/pdf/1707.06347.pdf) for openai gym environments. The two algorithms contain classes to record the data collected from the environments and for updating the network weights. It's up to the user to write their own training loops. The package attemps to stay out of the way as much as possible, only be used to record data and update the models. 

## Algorithms

The RL algorithms are designed to be modular, and agnostic to the training environments. Each algorithm expects to have an actor-critic network and optimizer for that network. Training data is captured via the `Rollouts` class. The user chooses how many steps to capture data for before updating the network weights. 

For example using A2C on an openai gym environment

```python
import gym
import numpy as np

import torch
import torch.optim as optim
from models.mlp import MLPolicy
from algorithms import A2C

env = gym.make('CartPole-v1')
n_states = env.observation_space.shape
n_actions = env.action_space.n

# mlp with 32 hidden units
policy = MLPolicy(n_states[0], n_actions, 32)
optimizer = optim.Adam(policy.parameters(), lr=0.0003)

update_algo = A2C(policy=policy, 
                  optimizer=optimizer,
                  num_envs=1,
                  state_size=n_states,
                  device=torch.device('cpu'))

idx = 0
ep_reward = 0
state = env.reset()
while idx < 10: # play for 10 episodes
    for t in range(5): # update every 8 time steps
        state = torch.from_numpy(state).unsqueeze(0).float()

        dist, value = update_algo.policy(state)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        new_state, reward, done, _ = env.step(action.item())
        ep_reward += reward
        reward = np.asarray([reward])

        env.render()

        update_algo.insert(t, s=state, a=action, 
                              v=value, lp=logprob, 
                              r=reward, d=done)

        state = new_state

        if done:
            print('{}: {}'.format(idx, ep_reward))
            ep_reward = 0
            env.reset()
            idx += 1
    
    next_val = torch.zeros(1,1)
    if not done:
        new_state = torch.from_numpy(new_state).unsqueeze(0).float()
        with torch.no_grad():
            _, next_val = update_algo.policy(new_state)

    update_algo.update(next_val)
```

## Installation

Get [pytorch](https://pytorch.org/) and [openai gym](https://gym.openai.com/)

## References

[David Silver Lectures](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

[DeepMind Lectures](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)

[Openai A2C blog](https://blog.openai.com/baselines-acktr-a2c/#a2canda3c)

[higgsfield rl-adventure-2](https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb)

[Intuitive RL: Intro to Advantage-Actor-Critic (A2C)](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752)