import torch
import torch.nn.functional as F
import numpy as np
import random
import time



def state_preprocess(obs, state_shape, previous_state = None, device=torch.device('cpu')):
    """Pre-proccessing specialized for the LunarLander environment. It takes the current observation of the environment
    and combines it with the last preprocessed state of the environment to create the new one.

    Args:
        obs (numpy.array): state of the environment returned by the step method
        state_shape (tuple): shape of the pre-processed entry of the neural network
        previous_state (torch.tensor): last state used as entry of the neural network
        device : device on which the algorithm is running

    Returns:
        torch.tensor: pre-processed new entry of the neural network
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