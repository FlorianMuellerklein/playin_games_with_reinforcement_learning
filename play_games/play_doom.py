import cv2
import time
import random
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from common.multiproc_env import SubprocVecEnv
from models.convnet import ConvPolicy

from vizdoom import *

global args
import argparse
parser = argparse.ArgumentParser(description='PyTorch gym with pixel inputs')
parser.add_argument('--num_episode', type=int, default=1000000,
                    help='number of total game episodes')
parser.add_argument('--num_steps', type=int, default=32,
                    help='number of steps before reflecting on your life')
parser.add_argument('--ppo_epochs', type=int, default=4,
                    help='number of epochs for ppo updates')
parser.add_argument('--rnn_size', type=int, default=256,
                    help='number of units in the rnn')
parser.add_argument('--lr', type=float, default=3e-6,
                    help='learning rate for adam')
parser.add_argument('--gamma', type=float, default=0.95,
                    help='discount factor (default: 0.99)')
parser.add_argument('--clip', type=float, default=0.1,
                    help='clip epsilon (default: 0.2)')
parser.add_argument('--num_envs', type=int, default=1,
                    help='number of parallel games')
parser.add_argument('--seed', type=int, default=543,
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='interval between training status logs (default: 10)')
parser.add_argument('--env_name', type=str, default='doom',
                    help='Which game to play')
parser.add_argument('--doom_cfg', type=str, 
                    default='whoopsie!',
                    help='Which game to play')
parser.add_argument('--algo', type=str, default='a2c',
                    help='which rl algo to use for weight updates')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--gpu', action='store_true',
                    help='whether to use gpu')
args = parser.parse_args()


game = DoomGame()
game.load_config(args.doom_cfg)
game.init()

n = game.get_available_buttons_size()
doom_actions = [list(a) for a in itertools.product([0, 1], repeat=n)]
allowable_buttons = game.get_available_buttons()
for i in range(len(allowable_buttons)):
    print(allowable_buttons[i], doom_actions[i])

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
doom_actionsactions = [shoot, left, right]

device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')

n_actions = len(doom_actions)
print('actions:', n_actions)

if args.algo == 'ppo':
    from evaluations.ppo import PPO
    update_algo = PPO(policy = ConvPolicy(n_actions).to(device), 
                      optimizer=optim.Adam, 
                      gamma=args.gamma, 
                      device=device,
                      lr=args.lr,
                      epochs=args.ppo_epochs)
else:
    from evaluations.a2c import A2C
    update_algo = A2C(policy = ConvPolicy(n_actions).to(device), 
                      optimizer=optim.Adam, 
                      gamma=args.gamma, 
                      device=device,
                      lr=args.lr)


end_rewards = []
def main():
    try:
        print('starting episodes') 
        ep_idx = 0
        restart = True
        while ep_idx < args.num_episode:
            
            states, actions, rewards, dones = [], [], [], []

            if restart:
                reward_sum = 0.
                game.new_episode()

            # play a game
            for t in range(args.num_steps):  # Don't infinite loop while learning
                state = game.get_state()
                s = state.screen_buffer
                s = torch.from_numpy(s).float().unsqueeze(0)

                with torch.no_grad():
                    p_ = update_algo.policy.sample_action_probs(s.to(device))
                    if torch.isnan(p_).any(): restart = True; break
                    a = p_.multinomial(num_samples=1).data

                r = game.make_action(doom_actions[a.item()])
                d = game.is_episode_finished()

                reward_sum += r.mean() if args.num_envs > 1 else r
                restart = d

                states.append(s)
                actions.append(a)
                rewards.append(r)
                dones.append(d)

                if args.render:
                    if args.num_envs == 1:
                        env.render()
                    else:
                        cv2.imshow('game', s[0])
                        cv2.waitKey(1)

                if (d if args.num_envs == 1 else d.any()):
                    restart = True
                    ep_idx += 1
                    end_rewards.append(reward_sum)

                    if ep_idx % args.log_interval == 0:
                        print('Episode {}\t Last Sum Reward: {:.5f}'.format(
                            ep_idx, reward_sum))
                    break

            if len(dones) > 1:
                update_algo.update(states, actions, rewards, dones)

    except KeyboardInterrupt:
        pass

    game.close()

    torch.save(update_algo.policy.state_dict(), '../model_weights/{}_conv-rnn.pth'.format(args.env_name))

    import pandas as pd

    out_dict = {'avg_end_rewards': end_rewards}
    out_log = pd.DataFrame(out_dict)
    out_log.to_csv('../logs/{}_policy_rewards.csv'.format(args.env_name), 
                   index=False)

    out_dict = {'actor losses': update_algo.policy_losses,
                'critic losses': update_algo.critic_losses}
    out_log = pd.DataFrame(out_dict)
    out_log.to_csv('../logs/{}_training_behavior.csv'.format(args.env_name), 
                   index=False)

if __name__ == '__main__':
    main()