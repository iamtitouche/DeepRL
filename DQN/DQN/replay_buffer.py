import torch
import torch.nn.functional as F
import numpy as np
import random
import time

"""Code de la classe ReplayBuffer"""


class ReplayBuffer:
    """Class ReplayBuffer"""

    def __init__(self, capacity, state_size, device):
        self.capacity = capacity

        self.states = torch.empty((capacity,) + state_size, dtype=torch.float32)
        self.actions = torch.empty((capacity, 1), dtype=torch.int64)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32)
        self.next_states = torch.empty((capacity,) + state_size, dtype=torch.float32)

        self.oldest_data_index = 0
        self.length = 0

    def __len__(self):
        """Get the number of transitions stored in the buffer.

        Returns:
            int: nulmber of items stored
        """
        return self.length

    def __str__(self):
        result = "Replay Buffer :\n"
        result += f"    States : {self.states}\n"
        result += f"    Actions : {self.actions}\n"
        result += f"    Rewards : {self.rewards}\n"
        result += f"    Done : {self.dones}\n"
        result += f"    Next States : {self.next_states}\n"
        result += f"    Capacity : {self.capacity}"
        return result

    def store_transition(self, state: torch.tensor, action: int, reward: float, next_state: torch.tensor, done: bool):
        """Store the transition in the replay buffer.

        Args:
            state (array): current state
            action (int): action taken
            reward (float): reward received
            next_state (array): next state
            done (bool): whether the episode is done
        """
        if self.length < self.capacity:
            self.states[self.length] = state
            self.actions[self.length] = action
            self.rewards[self.length] = reward
            self.next_states[self.length] = next_state
            self.dones[self.length] = done
            self.length += 1

        else:
            self.states[self.oldest_data_index] = state
            self.actions[self.oldest_data_index] = action
            self.rewards[self.oldest_data_index] = reward
            self.next_states[self.oldest_data_index] = next_state
            self.dones[self.oldest_data_index] = done

            self.oldest_data_index = (self.oldest_data_index + 1) % self.capacity

    def sample(self, batch_size, device):
        indexes = np.random.choice(len(self), size=batch_size, replace=False)
        indexes = torch.tensor(indexes, dtype=torch.int64)

        states = self.states[indexes].to(device)
        actions = self.actions[indexes].to(device)
        rewards = self.rewards[indexes].to(device)
        dones = self.dones[indexes].to(device)
        next_states = self.next_states[indexes].to(device)

        return states, actions, rewards, dones, next_states

    def rewards_not_all_null(self):
        for i in range(len(self)):
            if self.rewards[i] != 0:
                return True
        return False