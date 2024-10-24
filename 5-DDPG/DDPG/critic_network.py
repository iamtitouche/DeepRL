import os
import sys
import torch
import numpy as np
from torch import optim
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

from optimizer import create_optimizer


class CriticNetwork:

    def __init__(self, hyperparams_dict: dict, device: torch.device):
        self.network_state_value = hyperparams_dict['critic_network_state'].to(device)
        self.network_action_value = hyperparams_dict['critic_network_action'].to(device)
        self.network_final = hyperparams_dict['critic_network_final'].to(device)

        #self.check_networks(hyperparams_dict)

        self.network = nn.Sequential(
            self.network_state_value,
            self.network_action_value,
            self.network_final
        ).to(device)

        self.opt_type = hyperparams_dict['optimizer']

        self.device = device
        if hyperparams_dict['mode_training']:
            self.optimizer = create_optimizer(self.network, hyperparams_dict['learning_rate_critic'], hyperparams_dict['optimizer_type'])


    def __call__(self, states: torch.Tensor, actions) -> torch.Tensor:
        state_value = self.network_state_value(states)
        action_value = self.network_action_value(actions)

        state_action_value = torch.cat((state_value, action_value), dim=1)

        return self.network_final(state_action_value)

    def parameters(self):
        return self.network.parameters()

    def check_networks(self, hyperparams_dict: dict):
        if not isinstance(self.network_action_value[0], nn.Linear):
            raise TypeError(f'Expected first layer of critic network action value network to be '
                            f'a nn.Linear but got {type(self.network_action_value[0])}')
        if hyperparams_dict['n_actions'] != self.network_action_value[0].weight.shape[1]:
            raise ValueError(f'Expected first layer of critic network action value network to take '
                             f'{hyperparams_dict["n_actions"]} inputs but got {type(self.network_action_value[0])} instead')