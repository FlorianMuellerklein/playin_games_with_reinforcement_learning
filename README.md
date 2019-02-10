# playin_games_with_reinforcement_learning

I purchased the new edition of the [Sutton and Barto Reinforcement Learning book](http://incompleteideas.net/book/the-book-2nd.html) so I couldn't help myself. 

This repo implements Actor-Critic (A2C) and PPO for openai gym environments. The environments either pass the screen image or a vector as input so there are two variants of training code to handle the two input types. 

To train on an environment use the `env_name` flag and enter the name of the openai gym environment that you'd like to play. By default `lunar lander` is chosen. 

## Training on pixel input 

Train with A2C and default params.

'''
python pixel_input_gym.py --env_name Pong-v0 
'''

Train with PPO and different number of ppo epochs
'''
python pixel_input_gym.py --env_name Pong-v0 --eval_algo ppo --ppo_epochs 5
'''

## Training on vector input 

Train with A2C and default params.

'''
python vector_input_gym.py --env_name Pong-ram-v0 
'''

Train with PPO and different number of ppo epochs
'''
python vector_input_gym.py --env_name Pong-ram-v0 --eval_algo ppo --ppo_epochs 5
'''