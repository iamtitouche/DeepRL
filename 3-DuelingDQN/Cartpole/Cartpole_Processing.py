import torch
import torch.nn.functional as F
import numpy as np
import random
import time



def state_preprocess(obs, state_shape, previous_state = None, device=torch.device('cpu')):
    """Preproccessing specialized for the FrozenLake environment.

    Args:
        state (int): state of the game

    Returns:
        torch.tensor: tensor of a single normalized value
    """
    state = []
    if previous_state is not None:
        for i in range(1, state_shape[0]):
            state.append(previous_state[i].squeeze(0))

    state.append(torch.tensor(obs, dtype=torch.float32, device=device))

    return torch.stack(state)

def get_initial_state(env, state_shape, device):
    initial_state = env.reset()[0]
    return torch.tensor(np.array([initial_state] * state_shape[0]), dtype=torch.float32, device=device)