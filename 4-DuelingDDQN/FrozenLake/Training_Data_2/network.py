import torch

network = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32)
    )

network_value = torch.nn.Sequential(
        torch.nn.Linear(32, 1)
    )

network_advantage = torch.nn.Sequential(
        torch.nn.Linear(32, 4)
    )