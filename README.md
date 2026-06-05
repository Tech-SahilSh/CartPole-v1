# 🚀 CartPole-v1
A GitHub repository containing code and resources for the CartPole-v1 environment, a classic reinforcement learning problem. The repo includes implementations of various algorithms to solve the CartPole task. 

## Short Description
This repository provides a comprehensive solution to the CartPole-v1 environment using reinforcement learning algorithms.

## Tech Stack
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

## Features
* Implementation of Deep Q-Network (DQN) algorithm
* Training and testing scripts for the CartPole-v1 environment
* Pre-trained model and training rewards data
* Example usage of the trained model in a GUI environment

## Folder Structure
```markdown
CartPole-v1
├── gui_dqn.py
├── models
│   ├── dqn_cartpole.pth
│   ├── rewards.npy
│   └── training_rewards.png
├── output.png
├── train_dqn.py
└── README.md
```

## How to Run Locally
1. Clone the repository using `git clone https://github.com/your-username/CartPole-v1.git`
2. Navigate to the repository directory using `cd CartPole-v1`
3. Install required dependencies using `pip install -r requirements.txt` (create a requirements.txt file with necessary libraries like torch, numpy)
4. Train the model using `python train_dqn.py`
5. Run the GUI example using `python gui_dqn.py`

## Contributing Guide
To contribute to this repository, please fork the project, make your changes, and submit a pull request. Ensure that your code is well-documented and follows standard professional guidelines. 🚀