# play_games

The environments either pass the screen image or a vector as input so there are two variants of training code to handle the two input types. 

To train on an environment use the `env_name` flag and enter the name of the openai gym environment that you'd like to play. By default `LunarLander-v2` is chosen for vector inputs, and `Pong-v0` for pixel inputs. Training examples with Openai Gym can be found in the `play_games` folder. 

## Training on pixel input 

Train with A2C and default params.

```
python pixel_input_gym.py
```

Train with PPO and different number of ppo epochs
```
python pixel_input_gym.py --eval_algo ppo --ppo_epochs 5
```

## Training on vector input 

Train with A2C and default params.

```
python vector_input_gym.py 
```

Train with PPO and different number of ppo epochs
```
python vector_input_gym.py --eval_algo ppo --ppo_epochs 5
```

## Installation

Get [pytorch](https://pytorch.org/) and [openai gym](https://gym.openai.com/)