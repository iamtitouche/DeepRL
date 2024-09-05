import torch.nn as nn

def init_xavier_uniform(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_xavier_normal(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_he_normal(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def apply_initialization(network, mode):
    if mode == 'xavier_uniform':
        network.apply(init_xavier_uniform)
    elif mode == 'xavier_normal':
        network.apply(init_xavier_normal)
    elif mode == 'he_normal':
        network.apply(init_he_normal)
