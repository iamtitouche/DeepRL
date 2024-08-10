"""Deep Q-Learning Algorithm Implementation"""
import time

import sys
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import gymnasium as gm
from replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

from optimizer import create_optimizer

DEBUG = False


def debug_log(entry: str):
    """
    Print an entry only if the constant DEBUG is set to True

    Args:
        entry (str): printable entry
    """
    if DEBUG:
        print(entry)


class AgentDQN:
    """Class DQN for Deep Q-Learning Algorithm"""

    def __init__(self, hyperparams_dict):
        """Class AgentDQN Constructor

        Args:
            hyperparams_dict (dict): dictionnay of the hyperparameters
        """
        torch.autograd.set_detect_anomaly(True)
        self.env = hyperparams_dict["environment"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.__str__() == "cuda":
            print(f"Device used : {torch.cuda.get_device_name(0)}")
        else:
            print("Device used : cpu")

        self.number_actions = hyperparams_dict["action_space_size"]
        self.state_shape = hyperparams_dict["state_shape"]

        self.network_policy = hyperparams_dict["network"].to(self.device)

        if hyperparams_dict['mode_training']:
            self.network_target = copy.deepcopy(self.network_policy).to(self.device)
            self.exploration_mode = hyperparams_dict["exploration_mode"]
            if self.exploration_mode == 'softmax':
                self.tau_softmax = hyperparams_dict["tau_softmax"]
            elif self.exploration_mode == 'epsilon-greedy':
                self.epsilon = hyperparams_dict["epsilon"]
                self.epsilon_min = hyperparams_dict["epsilon_min"]
                self.epsilon_decay = hyperparams_dict["epsilon_decay"]

            self.update_mode = hyperparams_dict["update_mode"]
            if self.update_mode == "soft_update":
                if hyperparams_dict["tau"] == 1:
                    self.update_mode = "hard_update"
                else:
                    self.tau = hyperparams_dict["tau"]
            elif self.update_mode == "t-soft_update":
                self.tau = hyperparams_dict["tau"]
                self.nu = hyperparams_dict["nu"]
                self.W_i = (1 - self.tau) / self.tau
                self.tau_i = None
                self.tau_sigma_i = None
                self.w_i = None
                self.sigma_i_squared = 1e-2

            self.network_sync_rate = hyperparams_dict["update_frequency"]
            self.batch_size = hyperparams_dict["batch_size"]
            self.max_episodes = hyperparams_dict["max_episodes"]

            self.opt_type = hyperparams_dict['optimizer_type']

            self.discount_factor = hyperparams_dict["discount_factor"]
            self.optimizer = create_optimizer(
                self.network_policy,
                hyperparams_dict["learning_rate"],
                self.opt_type
            )

            self.grad_clipping_method = hyperparams_dict["grad_clipping_method"]
            if self.grad_clipping_method is not None:
                self.grad_clipping_threshold = hyperparameters["grad_clipping_threshold"]

            self.replay_buffer = ReplayBuffer(
                hyperparams_dict["memory_capacity"],
                self.state_shape,
                "cpu"
            )

        self.running_loss = 0
        self.state_preprocess = hyperparams_dict["state_preprocess"]
        self.get_initial_state = hyperparams_dict["get_initial_state"]

        self.losses = []

        self.timestep = 1

        self.create_summary(hyperparams_dict)

        self.working_directory = hyperparams_dict["working_directory"]

        if not os.path.exists(f"{self.working_directory}/checkpoints"):
            os.makedirs(f"{self.working_directory}/checkpoints")
            print("Checkpoints directory created")


    def create_summary(self, hyperparams_dict):
        """Create a summary of the learning process with TensorBoardX

        Args:
            hyperparams_dict (dict): dictionnay of the hyperparameters
        """
        self.writer = SummaryWriter()
        update_mode_mapping = {
            "hard_update": 0,
            "soft_update": 1,
            "t-soft_update": 2
        }
        hp = {
            'epsilon': hyperparams_dict['epsilon'],
            'epsilon_min': hyperparams_dict["epsilon_min"],
            'epsilon_decay': hyperparams_dict["epsilon_decay"],
            'update_mode': update_mode_mapping[self.update_mode],
            'batch_size': hyperparams_dict["batch_size"],
            'buffer_size': hyperparams_dict["memory_capacity"]
        }

        self.writer.add_hparams(
            hparam_dict=hp,
            metric_dict={}
        )



    def __str__(self):
        """Converstion to string method

        Returns:
            str: description of the Agent
        """
        result = "DQN Agent :\n"
        result += "  Data parameters :\n"
        result += f"    State Shape                         : {self.state_shape}\n"
        result += f"    Size of action space                : {self.number_actions}\n\n"
        result += "  Learning process hyperparameters :\n"
        result += f"    Learning rate                       : {self.optimizer.defaults['lr']}\n"
        result += f"    Optimizer Type                      : {self.opt_type}\n"
        result += f"    Exploration mode                    : {self.exploration_mode}\n"
        result += f"    Starting epsilon value              : {getattr(self, 'epsilon', None)}\n"
        result += f"    Minimum epsilon                     : {getattr(self, 'epsilon_min', None)}\n"
        result += f"    Epsilon decay rate                  : {getattr(self, 'epsilon_decay', None)}\n"
        result += f"    Discount factor (\u03B3)            : {self.discount_factor}\n"
        result += f"    Soft update parameter (\u03C4)      : {getattr(self, 'tau', None)}\n"
        result += f"    T-Soft update parameter (\u03BD)    : {getattr(self, 'nu', None)}\n"
        result += f"    Gradient clipping method            : {self.grad_clipping_method}\n"
        result += f"    Gradient clipping threshold         : {getattr(self, 'grad_clipping_threshold', None)}\n"
        result += f"    Batch size                          : {self.batch_size}\n"
        result += f"    Network synchronistation rate       : {self.network_sync_rate}\n"
        result += f"    Network Update mode                 : {self.update_mode}\n"
        result += f"    Buffer capacity                     : {self.replay_buffer.capacity}\n"
        result += f"    Maximum number of episodes          : {self.max_episodes}\n\n"
        result += "  Network Details :\n"
        result += f"    Network architecure                 : {self.network_policy}\n\n"
        return result


    def epoch(self):
        """Episode of training. In the case of the Deep Q-Learning Algorithm, an episode is done when the current game in the environnnemnt is done.

        Returns:
            float: reward obtained during the full episode
        """
        debug_log("Start of Episode : ")
        state = self.get_initial_state(self.env, self.state_shape, self.device)
        debug_log(f"Initial state : {state}")
        # Assertions for DEBUG
        #assert self.state_shape == state.shape
        #for i in range(1, self.state_shape[0]):
        #    assert torch.equal(state[i], state[0])

        debug_log(f"Shape should be : {self.state_shape} and is {state.shape}")

        total_reward = 0
        i = 1
        done = False
        truncated = False

        while not truncated and not done:
            step_result = self.step_training(state)
            next_state, reward, done, truncated, action = step_result

            next_state = self.state_preprocess(next_state, self.state_shape, state, self.device)

            total_reward += reward

            self.replay_buffer.store_transition(state, action, reward, next_state, done)

            state = next_state
            i += 1

            if len(self.replay_buffer) >= self.batch_size:
                loss = self.replay_experience()
                self.losses.append(loss.item())

                if len(self.losses) >= 200:
                    moving_average_loss = np.mean(self.losses[-200:])
                else:
                    moving_average_loss = np.mean(self.losses)

                self.writer.add_scalar('Moving_Average_Loss', moving_average_loss, self.timestep)

                if i % self.network_sync_rate == 0:
                    if self.update_mode == "hard_update":
                        self.hard_update()
                    elif self.update_mode == "soft_update":
                        self.soft_update()
                    elif self.update_mode == "t-soft_update":
                        self.t_soft_update()

            self.timestep += 1

        if self.exploration_mode == 'epsilon-greedy':
            self.epsilon_update()


        return total_reward

    def hard_update(self):
        """Full Synchronisation of the target network"""
        #debug_log("Hard update")
        self.network_target.load_state_dict(self.network_policy.state_dict())
        for mod1, mod2 in zip(self.network_policy, self.network_target):
            assert (type(mod1) == type(mod2))

            for param1, param2 in zip(mod1.parameters(), mod2.parameters()):
                assert param1.shape == param2.shape
                assert torch.equal(param1, param2)


    def soft_update(self):
        """Partial Synchronisation of the target network by weighted average"""
        debug_log("Soft update")
        for param_target, param_policy in zip(
            self.network_target.parameters(),
            self.network_policy.parameters()):
            param_target.data.copy_(self.tau * param_policy.data + (1 - self.tau) * param_target.data)


    def t_soft_update(self):
        """Partial Synchronisation of the target network with the t-soft update algorithm"""
        debug_log("t-soft update")
        policy_params = torch.cat([param.view(-1) for param in self.network_policy.parameters()])
        target_params = torch.cat([param.view(-1) for param in self.network_target.parameters()])

        delta_i_squared = F.mse_loss(policy_params, target_params).item()
        self.w_i = (self.nu + 1) / (self.nu + delta_i_squared / self.sigma_i_squared)
        self.tau_i = self.w_i / (self.W_i + self.w_i)

        for param_target, param_policy in zip(self.network_target.parameters(), self.network_policy.parameters()):
            param_target.data.copy_(self.tau_i * param_policy.data + (1 - self.tau_i) * param_target.data)

        self.tau_sigma_i = self.tau * self.w_i * self.nu / (self.nu + 1)

        self.sigma_i_squared = (1 - self.tau_sigma_i) * self.sigma_i_squared + self.tau_sigma_i * delta_i_squared
        self.W_i = (1 - self.tau) * (self.W_i + self.w_i)


    def choose_action_training(self, state: torch.tensor, policy: torch.nn.Sequential) -> int:
        """Choose the action according to the exploration methode of the agent

        Args:
            state (torch.tensor): state of the game

        Returns:
            int: action taken
        """
        debug_log(f"Choosing action : {self.exploration_mode}")
        if self.exploration_mode == "epsilon-greedy":

            if np.random.rand() < self.epsilon:
                debug_log("Choosing Randomly...")
                return np.random.randint(0, self.number_actions)

            debug_log("Choosing By Policy...")
            with torch.no_grad():
                q_values = policy(state.unsqueeze(0))
            return torch.argmax(q_values[0]).item()


        elif self.exploration_mode == "softmax":
            with torch.no_grad():
                q_values = policy(state.unsqueeze(0))

            p = torch.softmax(q_values * self.tau_softmax, dim=1)
            rand = np.random.rand()
            sum = p[0, 0].item()
            for i in range(1, p.shape[1]):
                if rand <= sum:
                    return i - 1
                sum += p[0, i].item()
            return p.shape[1] - 1

        raise ValueError(f"Exploration_mode {self.exploration_mode} is not valid")


    def step_training(self, state: torch.tensor, policy: torch.nn.Sequential = None) -> tuple:
        """Advance the game using the exploration method for choosing the action.

        Args:
            state (torch.tensor): state of the game
            policy (torch.nn.Sequential): policy network

        Returns:
            tuple: tuple containing the next state of the game, the reward, and a
            boolean describing if the game is finished
        """
        if policy is None:
            policy = self.network_policy

        action = self.choose_action_training(state, policy)

        action_result = self.env.step(action)

        return action_result[0], action_result[1], action_result[2], action_result[3], action


    def step_testing(self, state: torch.tensor) -> tuple:
        """Advance the game by following the agent policy.

        Args:
            state (torch.tensor): state of the game

        Returns:
            tuple: tuple containing the next state of the game, the reward, and a
            boolean describing if the game is finished
        """
        q_values = self.network_policy(state.unsqueeze(0))
        action = torch.argmax(q_values[0]).item()

        action_result = self.env.step(action)

        return action_result[0], action_result[1], action_result[2]

    def epsilon_update(self):
        """Update the value of epsilon if it is still above the set minimum."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        debug_log(f"Updating Espilon : {self.epsilon}")

    def replay_experience(self):
        """Replay and learn from the experience stored in the replay buffer."""
        debug_log("Starting Experience Replay...")
        states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.batch_size, self.device)

        debug_log(f"States : {states}")
        assert tuple(states.shape) == (self.batch_size,) + self.state_shape
        assert not states.requires_grad
        debug_log(f"Actions : {actions}")
        assert tuple(actions.shape) == (self.batch_size, 1)
        assert not actions.requires_grad
        debug_log(f"Q-Values from states : {self.network_policy(states)}")
        q_values = self.network_policy(states).gather(1, actions)
        assert tuple(q_values.shape) == (self.batch_size, 1)
        assert q_values.requires_grad
        debug_log(f"Q-Values for taken actions : {q_values}")

        with torch.no_grad():
            debug_log(f"Next states : {next_states}")
            debug_log(f"Target Q-Values from next states : {self.network_target(next_states)}")
            next_q_values = self.network_target(next_states).max(dim=1, keepdim=True)[0]

            debug_log(f"Best Target Q-Values from next states : {next_q_values}")
            expected_q_value = rewards + self.discount_factor * next_q_values * (1 - dones)

        assert expected_q_value.shape == (self.batch_size, 1)
        assert not expected_q_value.requires_grad

        loss = F.mse_loss(q_values, expected_q_value)
        self.running_loss += loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clipping_method == "norm":
            torch.nn.utils.clip_grad_norm_(self.network_policy.parameters(), self.grad_clipping_threshold)
        elif self.grad_clipping_method == "component":
            torch.nn.utils.clip_grad_value_(self.network_policy.parameters(), self.grad_clipping_threshold)
        
        self.optimizer.step()

        return loss

    def save_model(self, file_path: str):
        """Sauvegarde les poids et biais du modèle dans un fichier."""
        torch.save(self.network_policy.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path: str):
        """Charge les poids et biais du modèle depuis un fichier."""
        self.network_policy.load_state_dict(torch.load(file_path))
        self.network_target.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")

    def test(self):
        """Test the current policy network by playing the environment
        """
        state = get_initial_state(self.env)

        done = False
        count_step = 0
        while not done and count_step < 100:
            next_state, reward, done = self.step_testing(state)

            next_state = self.preprocess_func(next_state)
            state = next_state
            count_step += 1

    def train(self, checkpoint_step: int, visualize: bool = True):
        """Train the network

        Args:
            rewards_file (str): save file for rewards at each episode
            checkpoint_directory (str): directory for saving models
            checkpoint_step (int): model saving frequency
            visualize (bool, optional): Sets up the training visualisation. Defaults to True.
        """
        debug_log(self)
        debug_log("Start of training...")
        episode = 1
        rewards = []
        epsilons = []

        while episode <= self.max_episodes:
            #epsilons.append(self.epsilon)

            reward = self.epoch()
            print(episode, " reward : ", reward)
            rewards.append(reward)

            if episode >= 20:
                moving_average_reward = np.mean(rewards[-20:])
            else:
                moving_average_reward = np.mean(rewards)

            self.writer.add_scalar('Moving_Average_Reward', moving_average_reward, episode)

            rewards_file = f"{self.working_directory}/rewards.txt"
            checkpoint_directory = f"{self.working_directory}/checkpoints"

            with open(rewards_file, mode="a", encoding="utf-8") as file:
                file.write(f"Epoch {episode} : Reward : {reward:.8f} / Loss : {self.running_loss:.8f} ;\n")
                if (episode % checkpoint_step == 0):
                    self.save_model(f"{checkpoint_directory}/cp_{episode}.pth")
            episode += 1

            self.running_loss = 0

    def train_from_saved(self, model, nb_episode, rewards_file: str, checkpoint_directory: str, checkpoint_step: int, visualize: bool = True):
        """Train the network

        Args:
            rewards_file (str): save file for rewards at each episode
            checkpoint_directory (str): directory for saving models
            checkpoint_step (int): model saving frequency
            visualize (bool, optional): Sets up the training visualisation. Defaults to True.
        """
        self.load_model(model)

        debug_log(self)
        debug_log("Start of training...")
        episode = 1
        rewards = []
        epsilons = []

        if self.exploration_mode == 'epsilon-greedy':
            for i in range(nb_episode):
                self.epsilon_update()
        episode += 1

        while episode <= self.max_episodes:
            epsilons.append(self.epsilon)

            reward = self.epoch()
            print(episode, " reward : ", reward)
            rewards.append(reward)

            with open(rewards_file, mode="a", encoding="utf-8") as file:
                file.write(f"Epoch {episode} : Reward : {reward:.8f} / Loss : {self.running_loss:.8f} ;\n")
                if (episode % checkpoint_step == 0):
                    self.save_model(f"{checkpoint_directory}/cp_{episode}.pth")
            episode += 1

            self.running_loss = 0




   