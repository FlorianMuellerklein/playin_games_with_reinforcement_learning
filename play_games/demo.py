import sys
import time
import argparse

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from models.convnet import ConvPolicy 
from helpers import StateProc, MultiGym

device = torch.device('cpu')

algo = 'a2c'
env_name = "BreakoutNoFrameskip-v4"
env = gym.make(env_name)
n_states = env.observation_space.shape
n_actions = env.action_space.n
print('states:', n_states, 'actions:', n_actions)

state_proc = StateProc(num_envs=1, frame_shape=(105,80))

policy = ConvPolicy(n_actions).to(device)
policy.eval()
policy.load_state_dict(torch.load('../model_weights/{}_{}_conv.pth'.format(env_name, algo),
                                  map_location=lambda storage,
                                  loc: storage))
policy.eval()


def main():
    try:
        for _ in range(5):
            h = torch.zeros(1, 64)
            frame = env.reset()
            mask = torch.ones(1)
            done = False
            while not done:
                # stack the frames
                state = state_proc.proc_state(frame, mask=mask)
            
                with torch.no_grad():
                    probs, _ = policy(state)
                action = probs.argmax(dim=-1)
                frame, _, done, _ = env.step(action.item())
                mask = torch.tensor(1. - done).float()
                
                env.render()
                time.sleep(0.001)

                
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
