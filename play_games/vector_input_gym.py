import sys
import gym
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from common.multiproc_env import SubprocVecEnv
from models.gru import GRUPolicy

global args
import argparse
parser = argparse.ArgumentParser(description='PyTorch gym with pixel inputs')
parser.add_argument('--num_episode', type=int, default=100000,
                    help='number of total game episodes')
parser.add_argument('--num_steps', type=int, default=5,
                    help='number of steps before reflecting on your life')
parser.add_argument('--ppo_epochs', type=int, default=4,
                    help='number of epochs for ppo updates')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for ppo')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate for adam')
parser.add_argument('--lr_decay', action='store_true',
                    help='whether to decay learning rate linearly')
parser.add_argument('--hid_size', type=int, default=256,
                    help='number of units in the rnn')
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
parser.add_argument('--log-interval', type=int, default=10000,
                    help='interval between training status logs (default: 10)')
parser.add_argument('--env_name', type=str, default='LunarLander-v2',
                    help='Which game to play')
parser.add_argument('--algo', type=str, default='a2c',
                    help='which rl algo to use for weight updates')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--gpu', action='store_true',
                    help='whether to use gpu')
args = parser.parse_args()

lr_min = args.lr * 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')


if args.num_envs == 1:
     env = gym.make(args.env_name)
     env.seed(40807)
else:
    def make_env():
        def _thunk():
            env = gym.make(args.env_name)
            return env

        return _thunk

    env = [make_env() for i in range(args.num_envs)]
    env = SubprocVecEnv(env)

tenv = gym.make(args.env_name)

def test():
    s = tenv.reset()
    if args.render: tenv.render()
    d = False
    r_total = 0.
    h = torch.zeros((1, args.hid_size)).detach().to(device)
    while not d:
        s = torch.from_numpy(s).float().unsqueeze(0)
        with torch.no_grad():
            p_, _, h = update_algo.policy(s.to(device), h)
        a = p_.sample()
        s_, r, d, _ = tenv.step(a.cpu().numpy()[0])
        r_total += r
        s = s_
        if args.render: tenv.render()
    return r_total

n_states = env.observation_space.shape
n_actions = env.action_space.n
print('states:', n_states, 'actions:', n_actions)

policy = GRUPolicy(n_states[0], n_actions, args.hid_size,
                   args.num_steps, args.num_envs).to(device)
optimizer = optim.RMSprop(policy.parameters(), lr=args.lr, eps=1e-5)

if args.algo == 'ppo':
    sys.path.append('../')
    from algorithms.ppo import PPO
    update_algo = PPO(policy=policy,
                      optimizer=optimizer, 
                      num_steps=args.num_steps,
                      num_envs=args.num_envs,
                      state_size=n_states,
                      entropy_coef=args.entropy,
                      gamma=args.gamma, 
                      device=device,
                      recurrent=True,
                      rnn_size=args.hid_size,
                      epochs=args.ppo_epochs,
                      batch_size=args.batch_size)
else:
    sys.path.append('../')
    from algorithms.a2c import A2C
    update_algo = A2C(policy=policy, 
                      optimizer=optimizer, 
                      num_steps=args.num_steps,
                      num_envs=args.num_envs,
                      state_size=n_states,
                      entropy_coef=args.entropy,
                      gamma=args.gamma, 
                      device=device,
                      recurrent=True,
                      rnn_size=args.hid_size)

end_rewards = []
gt = 0

def main():
    try:
        print('starting episodes')
        d = False
        idx = 0
        episodes = 0
        restart = True
        s = env.reset()
        h = torch.zeros((args.num_envs, args.hid_size)).to(device)
        s = torch.from_numpy(s).float() if args.num_envs > 1 else torch.from_numpy(s).float().unsqueeze(0)
        while episodes < args.num_episode:
            reward_sum = 0.
            # play a game
            for t in range(args.num_steps):
                with torch.no_grad():
                    # insert state before getting actions
                    update_algo.insert_state(step=t, s=s, h=h, d=d)

                    p, v, h = update_algo.policy(s.to(device), h)
                    a = p.sample()
                    lp = p.log_prob(a)
                    e = p.entropy()

                    s, r, d, _ = env.step(a.cpu().numpy() if args.num_envs > 1 else a.item())

                    reward_sum += r.mean() if args.num_envs > 1 else np.asarray([r])
                    update_algo.insert_response(step=t, 
                                                a=a, 
                                                v=v, 
                                                lp=lp, 
                                                r=np.asarray([r]))

                    if (d if args.num_envs == 1 else d.any()):
                        episodes += 1
                        s = env.reset()
                        h = torch.zeros((args.num_envs, args.hid_size)).to(device)
                    else:
                        idx += 1

                    if idx % args.log_interval == 0:
                        test_rewards = np.mean([test() for _ in range(10)])
                        end_rewards.append(test_rewards)
                        print('Frames {}\t Test Reward: {:.5f} \t Episodes {}'.format(
                            idx, test_rewards, episodes))

                    s = torch.from_numpy(s).float() if args.num_envs > 1 else torch.from_numpy(s).float().unsqueeze(0)
                    _, next_val, _ = update_algo.policy(s.to(device), h)

            update_algo.update(next_val.unsqueeze(0), 
                               torch.tensor(1.-d, device=device).float())

            if args.lr_decay:
                for params in update_algo.optimizer.param_groups:
                    params['lr'] = (lr_min + 0.5 * (args.lr - lr_min) *
                                   (1 + np.cos(np.pi * idx / args.num_episode)))

    except KeyboardInterrupt:
        pass

    torch.save(update_algo.policy.state_dict(), 
               '../model_weights/{}_{}_mlp.pth'.format(args.env_name,
                                                       args.algo))

    import pandas as pd

    out_dict = {'avg_end_rewards': end_rewards}
    out_log = pd.DataFrame(out_dict)
    out_log.to_csv('../logs/{}_{}_policy_rewards.csv'.format(args.env_name,
                                                             args.algo), index=False)

    out_dict = {'actor losses': update_algo.actor_losses,
                'critic losses': update_algo.critic_losses,
                'entropy': update_algo.entropy_logs}
    out_log = pd.DataFrame(out_dict)
    out_log.to_csv('../logs/{}_{}_training_behavior.csv'.format(args.env_name,
                                                                args.algo), index=False)

if __name__ == '__main__':
    main()
