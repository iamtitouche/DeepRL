"""Deep Q-Learning Algorithm Implementation"""
import time

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import gymnasium as gm
from tensorboardX import SummaryWriter


class DuelingNetwork(nn.Module):

    def __init__(self, cnn, advantage_network, value_network, mode=None):
        self.cnn = cnn
        self.advantage_network = advantage_network
        self.value_network = value_network

        self.mode

    def forward(self, states):
        cnn_output = self.cnn(states).to(self.device)
        value_output = self.value_network(cnn_output).to(self.device)
        advantage_output = self.advantage_network(cnn_output).to(self.device)

        if self.mode is None:
            return value_output + advantage_output
        
        if self.mode == "max":
            advantage_output.sub_(advantage_output.max(dim=1, keepdim=True))
        elif self.mode == "mean":
            advantage_output.sub_(advantage_output.mean(dim=1, keepdim=True))

        return value_output + advantage_output