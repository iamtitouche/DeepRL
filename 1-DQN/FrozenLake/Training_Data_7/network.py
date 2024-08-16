import torch

network = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4)
    )