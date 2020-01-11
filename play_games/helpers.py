import torch
import numpy as np

import multiprocessing as mp

class StateProc:
    '''stacks four frames together'''
    def __init__(self, num_envs, frame_shape):
        self.num_envs = num_envs
        self.frame_shape = frame_shape
        # num_envs, channels, width, height
        self.frame_q = torch.zeros((num_envs, 4, 
                                    frame_shape[0],
                                    frame_shape[1]), dtype=torch.float)

    def proc_state(self, state, mask):
        # convert scene to gray
        state = self._rgb2gray(state.squeeze())

        # apply mask to frame stack, clearning stack if env is done
        self.frame_q = self.frame_q * mask.view(self.num_envs, 1, 1, 1)

        # if we only have one environment
        if self.num_envs == 1:
            state = state[::2, ::2] / 148.
        else:
            state = state[:, ::2, ::2] / 148.

        self._frame_stacker(state)

        return self.frame_q   

    def _frame_stacker(self, new_frame):
        # slide last three over
        self.frame_q[:, :-1] = self.frame_q[:, 1:].clone()
        # add new frame
        self.frame_q[:, -1] = torch.from_numpy(new_frame).clone()

    def _rgb2gray(self, frame):
        return np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])

def collect_exp(connection, gym_env, render=False):
    while True:
        cmd, action = connection.recv()
        if cmd == 'step':
            state, reward, done, _ = gym_env.step(action)
            reward = reward

            if render: gym_env.render()
            
            if done:
                gym_env.reset()
            connection.send((state, reward, done))
        else:
            state = gym_env.reset()
            connection.send(state)

class MultiGym:
    def __init__(self, gyms, render=False):
        self.gyms = gyms
        self.render = render
        self.connections = []

        for env in self.gyms:
            local, remote = mp.Pipe()
            self.connections.append(local)
            p = mp.Process(target=collect_exp, args=(remote, env, render))
            p.daemon = True
            p.start()
            remote.close()

        self.observation_space = self.gyms[0].observation_space
        self.action_space = self.gyms[0].action_space

    def step(self, actions):
        for conn, action in zip(self.connections, actions):
            conn.send(('step', action))

        results = [x.recv() for x in self.connections]
        state, reward, done = zip(*results)

        return (np.stack(state),
                np.stack(reward),
                np.stack(done))

    def reset(self):
        for conn in self.connections:
            conn.send(('reset', None))
        return np.stack([x.recv() for x in self.connections])