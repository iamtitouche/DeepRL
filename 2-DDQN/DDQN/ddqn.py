"""Double Deep Q-Network Algorithm Implementation"""

import sys
import os

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '1-DQN', 'DQN')))
from dqn import AgentDQN


class AgentDDQN(AgentDQN):
    """Class AgentDDQN for DOuble Deep Q-Network Algorithm"""

    def __init__(self, hyperparams_dict):
        """Class AgentDDQN Constructor

        Args:
            hyperparams_dict (dict): dictionnay of the hyperparameters
        """
        super().__init__(hyperparams_dict)

    def compute_target(self, rewards, dones, next_states):
        next_actions = torch.argmax(self.network_target(next_states), dim=1, keepdim=True)
        next_q_values = self.network_target(next_states).gather(1, next_actions)

        return rewards + self.discount_factor * next_q_values * (1 - dones)
