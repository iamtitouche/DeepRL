import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

from optimizer import create_optimizer


class ActorNetwork:

    def __init__(self, hyperparams_dict: dict, device: torch.device):
        self.network = hyperparams_dict['actor_network'].to(device)
        self.device = device
        self.opt_type = hyperparams_dict['optimizer']
        if hyperparams_dict['mode_training']:
            self.optimizer = create_optimizer(self.network, hyperparams_dict['learning_rate_actor'], hyperparams_dict['optimizer_type'])

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.network(input)

    def parameters(self):
        return self.network.parameters()