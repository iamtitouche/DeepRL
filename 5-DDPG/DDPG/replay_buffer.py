import torch
import numpy as np

class ReplayBuffer:
    """Class ReplayBuffer"""

    def __init__(self, capacity, state_size):
        self.capacity = capacity

        self.states = torch.empty((capacity,) + state_size, dtype=torch.float32)
        self.actions = torch.empty((capacity, 1), dtype=torch.float32)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.not_dones = torch.empty((capacity, 1), dtype=torch.float32)
        self.next_states = torch.empty((capacity,) + state_size, dtype=torch.float32)

        self.oldest_data_index = 0
        self.length = 0

    def __len__(self):
        """Get the number of transitions stored in the buffer.

        Returns:
            int: number of items stored
        """
        return self.length

    def __str__(self):
        result = "Replay Buffer :\n"
        result += f"    States : {self.states}\n"
        result += f"    Actions : {self.actions}\n"
        result += f"    Rewards : {self.rewards}\n"
        result += f"    Not Done : {self.not_dones}\n"
        result += f"    Next States : {self.next_states}\n"
        return result

    def store_transition(self, state: torch.tensor, action: int, reward: float, next_state: torch.tensor, done: bool):
        if self.length < self.capacity:
            self.states[self.length] = state
            self.actions[self.length] = action
            self.rewards[self.length] = reward
            self.next_states[self.length] = next_state
            self.not_dones[self.length] = 1 - done
            self.length += 1

        else:
            self.states[self.oldest_data_index] = state
            self.actions[self.oldest_data_index] = action
            self.rewards[self.oldest_data_index] = reward
            self.next_states[self.oldest_data_index] = next_state
            self.not_dones[self.oldest_data_index] = 1 - done
            self.oldest_data_index = (self.oldest_data_index + 1) % self.capacity


    def sample(self, batch_size, device):
        indexes = np.random.choice(len(self), size=batch_size, replace=False)

        states = self.states[indexes].to(device)
        actions = self.actions[indexes].to(device)
        rewards = self.rewards[indexes].to(device)
        not_dones = self.not_dones[indexes].to(device)
        next_states = self.next_states[indexes].to(device)

        return states, actions, rewards, not_dones, next_states