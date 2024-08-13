"""Deep Q-Learning Algorithm Implementation"""
import time

import sys
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import gymnasium as gm
from tensorboardX import SummaryWriter


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DQN', 'DQN')))
from dqn import AgentDQN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

from optimizer import create_optimizer

DEBUG = False


def debug_log(entry: str):
    """
    Print an entry only if the constant DEBUG is set to True

    Args:
        entry (str): printable entry
    """
    if DEBUG:
        print(entry)


class AgentDDQN(AgentDQN):
    """Class DDQN for Deep Q-Learning Algorithm"""

    def __init__(self, hyperparams_dict):
        """Class AgentDDQN Constructor

        Args:
            hyperparams_dict (dict): dictionnay of the hyperparameters
        """
        super().__init__(hyperparams_dict)

    def replay_experience(self):
        """Replay and learn from the experience stored in the replay buffer."""
        debug_log("Starting Experience Replay...")
        states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.batch_size, self.device)

        debug_log(f"States : {states}")
        assert tuple(states.shape) == (self.batch_size,) + self.state_shape
        assert not states.requires_grad
        debug_log(f"Actions : {actions}")
        assert tuple(actions.shape) == (self.batch_size, 1)
        assert not actions.requires_grad
        debug_log(f"Q-Values from states : {self.network_policy(states)}")
        q_values = self.network_policy(states).gather(1, actions)
        assert tuple(q_values.shape) == (self.batch_size, 1)
        assert q_values.requires_grad
        debug_log(f"Q-Values for taken actions : {q_values}")

        with torch.no_grad():
            debug_log(f"Next states : {next_states}")
            debug_log(f"Target Q-Values from next states : {self.network_target(next_states)}")
            next_actions =  torch.argmax(self.network_target(next_states), dim=1, keepdim=True)


            #debug_log(f"Best Target Q-Values from next states : {next_q_values}")
            expected_q_value = rewards + self.discount_factor * self.network_policy(next_states).gather(1, next_actions) * (1 - dones)

        assert expected_q_value.shape == (self.batch_size, 1)
        assert not expected_q_value.requires_grad

        loss = F.mse_loss(q_values, expected_q_value)
        self.running_loss += loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.network_policy.parameters(), 3)
        self.optimizer.step()

        return loss