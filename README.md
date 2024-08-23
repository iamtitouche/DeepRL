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
├── Algo/
│
├── Env1/
│   ├── Training_Data_1/
│   ├── Env1_Processing.py
│   └── train.py
│
└── Env2/
    ├── Training_Data_1/
    ├── Env2_Processing.py
    └── train.py
```

## Algorithms

- [1 - DQN](https://github.com/iamtitouche/DeepRL/tree/main/1-DQN)
- [2 - DDQN](https://github.com/iamtitouche/DeepRL/tree/main/2-DDQN)
- [3 - Dueling DQN](https://github.com/iamtitouche/DeepRL/tree/main/3-DuelingDQN)
- [4 - Dueling DDQN](https://github.com/iamtitouche/DeepRL/tree/main/4-DuelingDQN)
- 5 - DDPG
- 6 - TD3
- 7 - PPO
- 8 - SAC
- 9 - A2C
- 10 - A3C
- 11 - ACER
- 12 - GAIL
- 13 - ACER

## Training Status

Ce tableau résume l'état de l'entraînement pour chaque combinaison d'algorithme et d'environnement.

| Algorithm/Environment | FrozenLake | CartPole | LunarLander |
|-----------------------|------|------|------|
| **DQN**               | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) |
| **DDQN**              | ![Yes](https://img.shields.io/badge/Trained-Yes-brightgreen) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **Dueling DQN**       | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **Dueling DDQN**      | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **DDPG**              | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **TD3**               | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **PPO**               | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **SAC**               | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **A2C**               | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **A3C**               | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **ACER**              | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
| **GAIL**              | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) | ![No](https://img.shields.io/badge/Trained-No-red) |
