"""Code de la classe DQN"""
import time

import sys
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
from wrapper import FrozenLake
from FrozenLake_Processing import state_preprocess, get_initial_state

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DQN')))

from dqn import AgentDQN


if __name__ == "__main__":
    env = FrozenLake(render_mode="ansi", is_slippery=True, map_name='8x8', reward_mode='original')

    network = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(env.observation_space.n, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4)
    )

    hyperparameters = {
        'optimizer_type' : 'adam',
        'mode_training': True,
        # Environnement Informations
        "environment": env,
        "action_space_size": 4,
        "get_initial_state": get_initial_state,
        "state_preprocess": state_preprocess,

        "state_shape": (1, env.observation_space.n),

        # Training Parameters
        "network": network,
        "learning_rate": 1e-3,
        "clip_grad_norm": 3,
        "discount_factor": 0.99,
        "max_episodes": 10000,
        "memory_capacity": 10_000,
        "batch_size": 64,

        # Training Mode Parameters
        "update_frequency": 1,
        "update_mode": "soft_update",
        "exploration_mode": "epsilon-greedy",

        # For Soft and t-soft update modes
        "tau": 0.1,

        # For t-soft update mode
        "nu": 1,

        # For Softmax Exploration
        "tau_softmax": 5,

        # For Epsilon-Greedy Exploration
        "epsilon": 1,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.999,
        "working_directory": "Training_Data_2"
    }

    dqn = AgentDQN(hyperparameters)
    dqn.train(1000)

