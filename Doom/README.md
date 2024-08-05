VizDoom AI: Training Reinforcement Learning Agents in Doom

Project Description
This project demonstrates how to set up the VizDoom environment, convert it into a Gym environment, and train a reinforcement learning model using the PPO algorithm.

Table of Contents

1. Introduction
2. Technologies Used
3. Requirements
4. Installation Instructions
5. Usage Instructions
6. Features
7. Dataset
8. Model
9. Result
10. Documentation
11. Visuals
12. Conclusion

Introduction

This project covers the following steps:
1. Setting up the VizDoom environment.
2. Converting the VizDoom environment to a Gym environment.
3. Viewing the game state.
4. Training a reinforcement learning model using the PPO algorithm.

Technologies Used
Python
VizDoom
OpenAI Gym
NumPy
OpenCV
Matplotlib
Stable-Baselines3

Requirements
Python 3.x
VizDoom
OpenAI Gym
NumPy
OpenCV
Matplotlib
Stable-Baselines3

Installation Instructions

1. Clone the repository:
git clone https://github.com/SamratChanda26/Deep-Reinforcement-Learning.git
cd Doom

2. Install the required packages:
pip install vizdoom gym numpy opencv-python matplotlib stable-baselines3

Usage Instructions

1. Run the script to set up the VizDoom environment:
python setup_vizdoom.py

2. Run the script to convert the VizDoom environment to a Gym environment:
python convert_to_gym.py

3. View the game state by running:
python view_game_state.py

4. Train the model using:
python train_model.py

Features
Set up VizDoom environment.
Convert VizDoom to a Gym environment.
View game state using Matplotlib.
Train a reinforcement learning model using PPO.

Dataset
This project does not use an external dataset. The game state and rewards are generated within the VizDoom environment.

Model
The model used in this project is a reinforcement learning model trained using the PPO algorithm provided by Stable-Baselines3.

Result
The model's performance is logged and can be visualized using TensorBoard.

Documentation
VizDoom Documentation
OpenAI Gym Documentation
Stable-Baselines3 Documentation

Visuals

The game state can be visualized using Matplotlib. Example:

from matplotlib import pyplot as plt
import cv2

# Assuming 'state' is the game state
plt.imshow(cv2.cvtColor(state, cv2.COLOR_BGR2RGB))
plt.show()

Conclusion
This project demonstrates the process of setting up and using the VizDoom environment for reinforcement learning tasks, including converting it to a Gym environment and training a model using PPO.