import torch

network = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(8, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128)
    )

network_value = torch.nn.Sequential(
        torch.nn.Linear(128, 1)
    )

network_advantage = torch.nn.Sequential(
        torch.nn.Linear(128, 4)
    )