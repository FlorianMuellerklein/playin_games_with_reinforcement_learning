import sys
import gym
import time
import random 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from models.convnet import ConvPolicy 
from helpers import StateProc, MultiGym

#from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
#from stable_baselines.common import set_global_seeds

import matplotlib.pyplot as plt

global args
import argparse
parser = argparse.ArgumentParser(description='PyTorch gym with pixel inputs')
parser.add_argument('--num_updates', type=int, default=250000,
                    help='number of total game episodes')
parser.add_argument('--num_steps', type=int, default=5,
                    help='number of steps before reflecting on your life')
parser.add_argument('--ppo_epochs', type=int, default=4,
                    help='number of epochs for ppo updates')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='learning rate for adam')
parser.add_argument('--lr_decay', action='store_true',
                    help='whether to decay learning rate linearly')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--entropy', type=float, default=0.01,
                    help='coefficient for entropy')
parser.add_argument('--clip', type=float, default=0.2,
                    help='clip epsilon (default: 0.2)')
parser.add_argument('--num_envs', type=int, default=8,
                    help='number of parallel games')
parser.add_argument('--seed', type=int, default=543,
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='interval between training status logs (default: 10)')
parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4',
                    help='Which game to play')
parser.add_argument('--algo', type=str, default='a2c',
                    help='which rl algo to use for weight updates')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--gpu', action='store_true',
                    help='whether to use gpu')
args = parser.parse_args()

lr_min = args.lr * 0.001

device = torch.device('cuda:1' if torch.cuda.is_available() and args.gpu else 'cpu')

train_state_proc = StateProc(num_envs=args.num_envs, frame_shape=(105,80))
test_state_proc = StateProc(num_envs=1, frame_shape=(105,80))

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def viz_state(state):
    fix, ax = plt.subplots(4)
    for idx in range(state.shape[1]):
        ax[idx].imshow(state[0, idx,:,:].squeeze())
    plt.show()

def main():

    # make the environments
    if args.num_envs == 1:
        env = [gym.make(args.env_name)]
    else:
        env = [gym.make(args.env_name) for i in range(args.num_envs)]

    env = MultiGym(env, render=args.render)

    n_states = env.observation_space.shape
    n_actions = env.action_space.n
    print('state shape:', n_states, 'actions:', n_actions)

    policy = ConvPolicy(n_actions).to(device)
    optimizer = optim.RMSprop(policy.parameters(), lr=args.lr)

    if args.algo == 'ppo':
        sys.path.append('../')
        from algorithms.ppo import PPO
        update_algo = PPO(policy=policy,
                        optimizer=optimizer, 
                        num_steps=args.num_steps,
                        num_envs=args.num_envs,
                        state_size=(4, 105, 80),
                        entropy_coef=args.entropy,
                        gamma=args.gamma, 
                        device=device,
                        epochs=args.ppo_epochs)
    else:
        sys.path.append('../')
        from algorithms.a2c import A2C
        update_algo = A2C(policy=policy, 
                        optimizer=optimizer, 
                        num_steps=args.num_steps,
                        num_envs=args.num_envs,
                        state_size=(4, 105, 80),
                        entropy_coef=args.entropy,
                        gamma=args.gamma, 
                        device=device)

            
    end_rewards = []

    try:
        print('starting episodes') 
        idx = 0
        d = False
        reward_sum = np.zeros((args.num_envs))
        restart = True
        frame = env.reset()
        mask = torch.ones(args.num_envs)
        all_start = time.time()

        for update_idx in range(args.num_updates):
            update_algo.policy.train()

            # stack the frames
            s = train_state_proc.proc_state(frame, mask=mask)

            # insert state before getting actions
            update_algo.states[0].copy_(s)

            start = time.time()
            for step in range(args.num_steps):

                with torch.no_grad():
                    # get probability dist and values
                    p, v = update_algo.policy(update_algo.states[step])
                    a = Categorical(p).sample()

                # take action get response
                frame, r, d = env.step(a.cpu().numpy() if args.num_envs > 1 else [a.item()])
                s = train_state_proc.proc_state(frame, mask)

                update_algo.insert_experience(step=step,
                                              s=s,
                                              a=a,
                                              v=v,
                                              r=r,
                                              d=d)


                mask = torch.tensor(1. - d).float()
                reward_sum = (reward_sum + r)

                # if any episode finished append episode reward to list
                if d.any():
                    end_rewards.extend(reward_sum[d])
                
                # reset any rewards that finished
                reward_sum = reward_sum * mask.numpy()

                idx += 1

            with torch.no_grad():
                _, next_val = update_algo.policy(update_algo.states[-1])

            update_algo.update(next_val.view(1, args.num_envs).to(device), 
                               next_mask=mask.to(device))

            if args.lr_decay:
                for params in update_algo.optimizer.param_groups:
                    params['lr'] = (lr_min + 0.5 * (args.lr - lr_min) *
                                    (1 + np.cos(np.pi * idx / args.num_updates)))

            # update every so often by displaying results in term
            if (update_idx % args.log_interval == 0) and (len(end_rewards) > 0):
                total_steps = (idx + 1) * args.num_envs * args.num_steps
                end = time.time()
                print(end_rewards[-10:])
                print('Updates {}\t  Time: {:.4f} \t FPS: {}'.format(
                            update_idx, end-start, int(total_steps / (end - all_start))))
                print('Mean Episode Rewards: {:.2f} \t Min/Max Current Rewards: {}/{}'.format(
                       np.mean(end_rewards[-10:]), reward_sum.min(), reward_sum.max()
                ))

    except KeyboardInterrupt:
        pass

    torch.save(update_algo.policy.state_dict(), 
               '../model_weights/{}_{}_conv.pth'.format(args.env_name,
                                                        args.algo))

    import pandas as pd

    out_dict = {'avg_end_rewards': end_rewards}
    out_log = pd.DataFrame(out_dict)
    out_log.to_csv('../logs/{}_{}_rewards.csv'.format(args.env_name,
                                                      args.algo), index=False)

    out_dict = {'actor losses': update_algo.actor_losses,
                'critic losses': update_algo.critic_losses,
                'entropy': update_algo.entropy_logs}
    out_log = pd.DataFrame(out_dict)
    out_log.to_csv('../logs/{}_{}_training_behavior.csv'.format(args.env_name,
                                                                args.algo), index=False)

    plt.plot(end_rewards)
    plt.show()

    #env.close()

if __name__ == '__main__':
    main()