# Deep Reinforcement Learning Tutorial

Welcome to the Deep Reinforcement Learning (DRL) tutorial repository. This project aims to provide a comprehensive guide and practical examples to help you understand and implement various Deep Reinforcement Learning algorithms. Whether you're a beginner or an experienced practitioner, you'll find useful resources and code snippets here to enhance your understanding of DRL.

## Table of Contents

- [Introduction](#introduction)
- [Structure of the Project](#structure-of-the-project)

## Introduction

Deep Reinforcement Learning is a rapidly growing field in machine learning, combining the principles of reinforcement learning with deep learning techniques. This tutorial covers fundamental concepts and advanced techniques, offering code examples, explanations, and practical tips to help you build and understand DRL models.

## Structure of the Project

This project is divided into subdirectories, each corresponding to a different learning algorithm. Each of these subdirectories contains a folder named after the algorithm, which holds the Python implementation of the algorithm and a README that explains the theory behind how the algorithm works. Additionally, there are subdirectories specific to different environments, where you will find the training results, preprocessing functions specific to each training, potentially a wrapper to modify the environment's behavior, and a script to launch the training.

```
Algo_name/
│
├── README.md # Explains the maths behind the algorithm and my implementation of it
│
├── Algo/ # Contains the source code of the algorithm
│
├── Env1/
│   ├── Training_Data_1/ # Contains the data from a training session
│   |   ├── checkpoints/ # Contains saved models from the training session
│   |   |   └── cp_100.pth
│   |   |
│   │   ├── config.yaml # Hyperparameters of the training session and environment configuration
│   │   ├── network.py # Contains the network architecture of the model
│   │   ├── rewards.txt
│   │   └── rewards.png
│   │
│   ├── Env1_Processing.py
│   ├── train.py
│   └── README.md # Explains the environment and gives an analysis of the training results
│
└── Env2/
    ├── Training_Data_1/
    ├── Env2_Processing.py
    ├── train.py
    └── README.md
```

## Algorithms

- [1 - DQN](https://github.com/iamtitouche/DeepRL/tree/main/1-DQN) ![Status](https://img.shields.io/badge/Status-Implemented-brightgreen)
- [2 - DDQN](https://github.com/iamtitouche/DeepRL/tree/main/2-DDQN) ![Status](https://img.shields.io/badge/Status-Implemented-brightgreen)
- [3 - Dueling DQN](https://github.com/iamtitouche/DeepRL/tree/main/3-DuelingDQN) ![Status](https://img.shields.io/badge/Status-Implemented-brightgreen)
- [4 - Dueling DDQN](https://github.com/iamtitouche/DeepRL/tree/main/4-DuelingDQN) ![Status](https://img.shields.io/badge/Status-Implemented-brightgreen)
- [5 - DDPG](https://github.com/iamtitouche/DeepRL/tree/main/5-DDPG) ![Status](https://img.shields.io/badge/Status-ONGOING-orange)
- [6 - TD3](https://github.com/iamtitouche/DeepRL/tree/main/6-TD3) ![Status](https://img.shields.io/badge/Status-ONGOING-orange)
- 7 - PPO ![Status](https://img.shields.io/badge/Status-TODO-red)
- 8 - SAC ![Status](https://img.shields.io/badge/Status-TODO-red)
- 9 - A2C ![Status](https://img.shields.io/badge/Status-TODO-red)
- 10 - A3C ![Status](https://img.shields.io/badge/Status-TODO-red)
- 11 - ACER ![Status](https://img.shields.io/badge/Status-TODO-red)
- 12 - GAIL ![Status](https://img.shields.io/badge/Status-TODO-red)
- 13 - ACER ![Status](https://img.shields.io/badge/Status-TODO-red)

## Training Status

The following table keeps track of the environments I successfully trained a model on, using a given learning algorithm. 

| Algorithm/Environment | FrozenLake                                                           | CartPole | LunarLander |
|-----------------------|----------------------------------------------------------------------|------|------|
| **DQN**               | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen)         | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) |
| **DDQN**              | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen)         | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) |
| **Dueling DQN**       | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen)         | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) |
| **Dueling DDQN**      | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen)         | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) |
| **DDPG**              | ![Not Adapted](https://img.shields.io/badge/Trained-NotAdapted-gray) | ![Not Adapted](https://img.shields.io/badge/Trained-NotAdapted-gray) | ![Not Adapted](https://img.shields.io/badge/Trained-NotAdapted-gray) |
| **TD3**               | ![Not Adapted](https://img.shields.io/badge/Trained-NotAdapted-gray) | ![Not Adapted](https://img.shields.io/badge/Trained-NotAdapted-gray) | ![Not Adapted](https://img.shields.io/badge/Trained-NotAdapted-gray) |
| **PPO**               | ![No](https://img.shields.io/badge/Trained-No-red)                   | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **A2C**               | ![No](https://img.shields.io/badge/Trained-No-red)                   | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **A3C**               | ![No](https://img.shields.io/badge/Trained-No-red)                   | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **ACER**              | ![No](https://img.shields.io/badge/Trained-No-red)                   | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |


## Things I want to learn about 

- Noisy DQN
- Rainbow algorithm
- BPTT
- Prioritised experience replay
- SHAC
- AHAC
- Distributional DQN
- TRPO
- HER
- AWAC
- AWETT
- ACKTR