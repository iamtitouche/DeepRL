"""Dueling Double Deep Q-Learning Algorithm Implementation"""

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DuelingDQN', 'DuelingDQN')))
from network import DuelingNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DDQN', 'DDQN')))
from ddqn import AgentDDQN



class AgentDuelingDDQN(AgentDDQN):
    """Class AgentDuelingDQN for Deep Q-Learning Algorithm"""

    def __init__(self, hyperparams_dict):
        """Class AgentDuelingDQN Constructor

        Args:
            hyperparams_dict (dict): dictionnay of the hyperparameters
        """
        self.advantage_mode = hyperparams_dict['advantage_mode']
        dueling_network = DuelingNetwork(hyperparams_dict['cnn'], hyperparams_dict['advantage_network'], hyperparams_dict['value_network'], self.advantage_mode)
        hyperparams_dict['network'] = dueling_network
        super().__init__(hyperparams_dict)