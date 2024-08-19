import torch
import torch.nn.functional as F
import numpy as np
import random
import time


def state_preprocess(state, state_shape, previous_state = None, device=torch.device('cpu')):
    """Preproccessing specialized for the FrozenLake environment.

    Args:
        state (int): state of the game

    Returns:
        torch.tensor: tensor of a single normalized value
    """
    onehot_vector = torch.zeros(state_shape, dtype=torch.float32)
    onehot_vector[0][state] = 1
    return onehot_vector.to(device)



def get_initial_state(env, state_shape, device):
    return state_preprocess(env.reset()[0], state_shape, None, device)

