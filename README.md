# playin_games_with_reinforcement_learning

I purchased the new edition of the [Sutton and Barto Reinforcement Learning book](http://incompleteideas.net/book/the-book-2nd.html) so I couldn't help myself. 

This repo implements Actor-Critic (A2C) and [PPO](https://arxiv.org/pdf/1707.06347.pdf) for openai gym environments. The environments either pass the screen image or a vector as input so there are two variants of training code to handle the two input types. 

To train on an environment use the `env_name` flag and enter the name of the openai gym environment that you'd like to play. By default `LunarLander-v2` is chosen for vector inputs, and `Pong-v0` for pixel inputs. Training is done from the `play_games` folder. 

## Training on pixel input 

Train with A2C and default params.

```
python pixel_input_gym.py --env_name Pong-v0 
```

Train with PPO and different number of ppo epochs
```
python pixel_input_gym.py --env_name Pong-v0 --eval_algo ppo --ppo_epochs 5
```

## Training on vector input 

Train with A2C and default params.

```
python vector_input_gym.py --env_name Pong-ram-v0 
```

Train with PPO and different number of ppo epochs
```
python vector_input_gym.py --env_name Pong-ram-v0 --eval_algo ppo --ppo_epochs 5
```

## Installation

Get [pytorch](https://pytorch.org/) and [openai gym](https://gym.openai.com/)

## References

(David Silver Lectures)[https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ]

(DeepMind Lectures)[https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs]

(Openai A2C blog)[https://blog.openai.com/baselines-acktr-a2c/#a2canda3c]

(higgsfield rl-adventure-2)[https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb]

(Intuitive RL: Intro to Advantage-Actor-Critic (A2C))[https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752]