"""Common functions for optimizers"""

from torch import nn, optim

def create_optimizer(network: nn.Module, learning_rate: float, opt_type: str='adam'):
    """Create and returns an optimizer for a given network.

    Args:
        network (torch.nn.Module): neural network to associate with a new optimizer. 
        learning_rate (float): learning rate.
        opt_type (str, optional): optimizer type. Defaults to 'adam'.

    Returns:
        torch.optim: optimizer of the given neural network.
    """
    if opt_type == 'adam':
        return optim.Adam(network.parameters(), lr=learning_rate)
    if opt_type == 'rms_prop':
        return optim.RMSProp(network.parameters(), lr=learning_rate)
    raise ValueError("Optimizer type is not valid")
