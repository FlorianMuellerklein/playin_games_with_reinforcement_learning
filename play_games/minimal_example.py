
import gym
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from models.mlp import MLPolicy
import sys
sys.path.append('../')
from algorithms import A2C

N_STEPS = 8
N_EPISODE = 5000

env = gym.make('CartPole-v1')
n_states = env.observation_space.shape
n_actions = env.action_space.n

# mlp with 32 hidden units
policy = MLPolicy(n_states[0], n_actions, n_hidden=32)
optimizer = optim.Adam(policy.parameters(), lr=0.0003)

update_algo = A2C(policy=policy, 
                  optimizer=optimizer,
                  num_envs=1,
                  num_steps=N_STEPS,
                  state_size=n_states,
                  device=torch.device('cpu'))

idx = 0
done = False
ep_reward = 0
ep_rewards = []
state = env.reset()
state = torch.from_numpy(state).unsqueeze(0).float()
while idx < N_EPISODE: # play for N_EPISODE episodes
    for t in range(N_STEPS): # update every N_STEPS time steps
        with torch.no_grad(): # record experience without keep track of grads
            # insert state before getting actions
            update_algo.insert_state(step=t, s=state, d=done)

            dist, value = update_algo.policy(state)
            action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()

            state, reward, done, _ = env.step(action.item())

            # episode only ends when pole falls, punish this heavily
            reward = -100 if (done and not ep_reward == 499) else reward

            ep_reward += reward
            reward = np.asarray([reward])

            #env.render()
            
            # insert the environment response from <state, action>
            update_algo.insert_response(step=t, 
                                        a=action, 
                                        v=value, 
                                        lp=logprob, 
                                        r=reward)

            if done:
                print('{}: {}'.format(idx, ep_reward + 100))
                ep_rewards.append(ep_reward + 100)
                ep_reward = 0
                state = env.reset()
                idx += 1

            state = torch.from_numpy(state).unsqueeze(0).float()
        
        next_val = torch.zeros(1,1)

        if not done:
            _, next_val = update_algo.policy(state)

    update_algo.update(next_val, next_mask=torch.tensor(1.-done).float())

env.close()

plt.plot(ep_rewards)
plt.show()