import torch

network_actor = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(3, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
        torch.nn.Tanh()
    )

network_critic_action = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.ReLU()
    )

network_critic_state = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(3, 32),
        torch.nn.ReLU()
    )

network_critic_final = torch.nn.Sequential(
        torch.nn.Linear(64, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1)
    )