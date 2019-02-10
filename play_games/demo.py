import argparse

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from models.gru import GRUPolicy

device = torch.device('cpu')

env_name = "LunarLander-v2"
env = gym.make(env_name)
n_states = env.observation_space.shape
n_actions = env.action_space.n
print('states:', n_states, 'actions:', n_actions)


policy = GRUPolicy(n_states[0], n_actions).to(device)
policy.load_state_dict(torch.load('../model_weights/{}_mlp.pth'.format(env_name),
                                  map_location=lambda storage,
                                  loc: storage))


def state_proc(state):
    #state = state[::2, ::2]
    state = state.transpose((2,0,1))
    state = state.astype(np.float32) / 255.
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    return state

policy_losses = []
def main():
    running_reward = 10
    try:
        for _ in range(5):
            h = torch.zeros(1, 64)
            state = env.reset()
            done = False
            while not done:
            
                #h = torch.zeros(1, 64).to(device)
                #for t in range(10000):  # Don't infinite loop while learning
                #h.detach()
                #state = state_proc(state)
                
                with torch.no_grad():
                    probs, _, h = policy(torch.from_numpy(state).float().unsqueeze(0),
                                        h)
                print(probs)
                _, action_idx = probs.max(-1)
                action = action_idx.item()
                print(action)
                state, _, done, _ = env.step(action)
                
                env.render()

                print(done)
                
                #if done:
                #    break

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
