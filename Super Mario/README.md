Mario Reinforcement Learning Project

Project Description
This project involves creating a Reinforcement Learning (RL) agent that learns to play the classic Nintendo game, Super Mario Bros. The agent is trained using the Proximal Policy Optimization (PPO) algorithm to navigate and complete levels in the game.

Table of Contents

1. Introduction
2. Technologies Used
3. Requirements
4. Installation Instructions
5. Usage Instructions
6. Features
7. Dataset
8. Model
9. Results
10. Documentation
11. Visuals
12. Conclusion

Introduction
The objective of this project is to create an AI agent that can autonomously play Super Mario Bros. using reinforcement learning techniques. The project leverages OpenAI's Gym environment for the game simulation and Stable Baselines3 for the RL algorithms.

Technologies Used
Python
OpenAI Gym
Gym Super Mario Bros
Stable Baselines3
NES-py
Matplotlib

Requirements
Python 3.6 or higher
Gym Super Mario Bros
Stable Baselines3
NES-py
Matplotlib

Installation Instructions
1. Clone the repository:
git clone https://github.com/SamratChanda26/Deep-Reinforcement-Learning.git
cd Super Mario

2. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required packages:
pip install -r requirements.txt

Usage Instructions

1. Setup Mario Environment:

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

2. Preprocess the Environment:

from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

state = env.reset()

3. Train the RL Model:

from stable_baselines3 import PPO

LOG_DIR = './logs/'
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
model.learn(total_timesteps=1000000)

Features
Simplified controls for the Mario environment.
Preprocessing steps including grayscale conversion and frame stacking.
Training using the PPO algorithm with extensive logging.

Dataset
The project does not use a traditional dataset. Instead, it uses the Super Mario Bros. game environment where the agent interacts and learns from the game frames and rewards.

Model
The model used in this project is based on the PPO algorithm implemented in the Stable Baselines3 library. The policy network is a Convolutional Neural Network (CNN) suitable for processing image-based inputs.

Results
The results of the training process include improved performance of the Mario agent over time, as evidenced by increasing scores and more consistent level completions.

Documentation
For more detailed documentation on using the Gym Super Mario Bros environment and the Stable Baselines3 library, refer to:

Gym Super Mario Bros Documentation
Stable Baselines3 Documentation

Visuals
During training, frame stacking can be visualized as follows:

plt.figure(figsize=(20, 16))
for idx in range(state.shape[3]):
    plt.subplot(1, 4, idx+1)
    plt.imshow(state[0][:, :, idx])
plt.show()

Conclusion
This project demonstrates the application of reinforcement learning to a classic game, showing the potential of AI in mastering complex tasks. The use of PPO, combined with effective preprocessing techniques, enables the agent to learn and improve its performance in Super Mario Bros.