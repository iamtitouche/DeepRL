import copy
import enum
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import time

from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

from optimizer import clip_gradients


class AgentDDPG:

    def __init__(self, hyperparams_dict):
        print("Initializing the agent...")
        torch.autograd.set_detect_anomaly(True)
        self.env = hyperparams_dict["environment"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.__str__() == "cuda":
            print(f"Working on : {torch.cuda.get_device_name(0)}")
        else:
            print("Working on : cpu")

        self.n_actions = hyperparams_dict['n_actions']

        self.state_shape = hyperparams_dict["state_shape"]
        if hyperparams_dict['mode_training']:
            self.max_episodes = hyperparams_dict["max_episodes"]
            self.discount_factor = hyperparams_dict['discount_factor']

            self.batch_size = hyperparams_dict['batch_size']
            self.buffer_size = hyperparams_dict['memory_capacity']
            self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_shape)

        self.actor = ActorNetwork(hyperparams_dict, self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = CriticNetwork(hyperparams_dict, self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.update_mode = hyperparams_dict['update_mode']
        self.update_frequency = hyperparams_dict['update_frequency']

        if self.update_mode == 'soft_update':
            if hyperparams_dict['tau'] == 1:
                self.update_mode = "hard_update"
            self.tau = hyperparams_dict['tau']

        self.action_lower_bound = hyperparams_dict['action_lower_bound'].to(self.device)
        self.action_upper_bound = hyperparams_dict['action_upper_bound'].to(self.device)

        self.get_initial_state = hyperparams_dict["get_initial_state"]
        self.state_preprocess = hyperparams_dict["state_preprocess"]

        self.working_directory = hyperparams_dict["working_directory"]

        if not os.path.exists(f"{self.working_directory}/checkpoints"):
            os.makedirs(f"{self.working_directory}/checkpoints")
            print("Checkpoints directory created")

        self.critic_grad_clipping_method = hyperparams_dict['critic_grad_clipping_method']
        self.critic_grad_clipping_threshold = hyperparams_dict['critic_grad_clipping_threshold']
        self.actor_grad_clipping_method = hyperparams_dict['actor_grad_clipping_method']
        self.actor_grad_clipping_threshold = hyperparams_dict['actor_grad_clipping_threshold']

    def __str__(self):
        """Converstion to string method

        Returns:
            str: description of the Agent
        """
        result = "DDPG Agent :\n"
        result += "  Data parameters :\n"
        result += f"    State Shape                         : {self.state_shape}\n"
        result += f"    Size of actions                     : {self.n_actions}\n"
        result += f"    Min Values for actions              : {self.action_lower_bound}\n"
        result += f"    Max Values for actions              : {self.action_upper_bound}\n\n"
        result += "  Learning process hyperparameters :\n"
        result += "    Actor Network parameters:\n"
        result += f"      Learning rate                       : {self.actor.optimizer.defaults['lr']}\n"
        result += f"      Optimizer Type                      : {self.actor.opt_type}\n"
        result += "    Critic Network parameters:\n"
        result += f"      Learning rate                       : {self.critic.optimizer.defaults['lr']}\n"
        result += f"      Optimizer Type                      : {self.critic.opt_type}\n"
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
        """Episode of training

        Returns:
            float: reward obtained during the full episode
        """
        state = self.get_initial_state(self.env, self.state_shape, self.device)


        total_reward = 0
        i = 1
        done = False
        truncated = False

        while not truncated and not done and i < 1000:

            step_result = self.step_training(state)
            next_state, reward, done, truncated, action = step_result

            if done and next_state is None:
                next_state = self.get_initial_state(self.env, self.state_shape, self.device)
            else:
                next_state = self.state_preprocess(next_state, self.state_shape, state, self.device)


            total_reward += reward

            self.replay_buffer.store_transition(state, action, reward, next_state, done)

            state = next_state
            i += 1

            if len(self.replay_buffer) >= self.batch_size:
                if i % self.update_frequency == 0:
                    self.replay_experience(True)
                    if self.update_mode == "hard_update":
                        self.hard_update()
                    elif self.update_mode == "soft_update":
                        self.soft_update()
                else:
                    self.replay_experience(False)

        return total_reward

    def hard_update(self):
        self.actor_target.network.load_state_dict(self.actor.network.state_dict())
        self.critic_target.network.load_state_dict(self.critic.network.state_dict())

    def soft_update(self):
        for param_target, param_policy in zip(self.actor_target.parameters(),
                                              self.actor.parameters()):
            param_target.data.copy_(self.tau * param_policy.data + (1 - self.tau) * param_target.data)

        for param_target, param_policy in zip(self.critic_target.parameters(),
                                              self.critic.parameters()):
            param_target.data.copy_(self.tau * param_policy.data + (1 - self.tau) * param_target.data)


    def choose_action_training(self, state) -> torch.Tensor:
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0))

            action += torch.from_numpy(np.sqrt(0.2) * np.random.normal(size=np.zeros(1).shape)).to(self.device)

            action = self.action_lower_bound + (self.action_upper_bound - self.action_lower_bound) * (action + 1) / 2
            action = torch.max(torch.min(action, self.action_upper_bound),
                               self.action_lower_bound)
        return action

    def step_training(self, state):
        action = self.choose_action_training(state).squeeze(0)
        result = self.env.step(np.array(action.cpu()))
        return result[0], result[1], result[2], result[3], action

    def step_testing(self, state):
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0)).squeeze(0)

        result = self.env.step(np.array(action.cpu()))

        return result[0], result[1], result[2], result[3]

    def replay_experience(self, update_actor) -> None:
        states, actions, rewards, not_dones, next_states = self.replay_buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)

            critic_target = self.critic_target(next_states, next_actions)

            target = rewards + self.discount_factor * critic_target * not_dones

        critic_value = self.critic(states, actions)
        critic_loss = nn.MSELoss()(target, critic_value)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        clip_gradients(self.critic, self.critic_grad_clipping_method, self.critic_grad_clipping_threshold)
        self.critic.optimizer.step()

        if update_actor:
            actions = self.actor(states)
            critic_value = self.critic(states, actions)
            actor_loss = - critic_value.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            clip_gradients(self.critic, self.actor_grad_clipping_method, self.actor_grad_clipping_threshold)
            self.actor.optimizer.step()

    def load_model(self, dir_path: str, model_nb):
        """Sauvegarde les poids et biais du modèle dans un fichier."""
        self.critic.network.load_state_dict(torch.load(f'{dir_path}/cp_critic_{model_nb}.pth'))
        self.actor.network.load_state_dict(torch.load(f'{dir_path}/cp_actor_{model_nb}.pth'))
        print("Actor Model loaded")

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

    def save_model(self, dir_path: str, model_nb):
        """Sauvegarde les poids et biais du modèle dans un fichier."""
        torch.save(self.critic.network.state_dict(), f'{dir_path}/cp_critic_{model_nb}.pth')
        print(f"Critic Model saved to {f'{dir_path}/cp_critic_{model_nb}.pth'}")

        torch.save(self.actor.network.state_dict(), f'{dir_path}/cp_actor_{model_nb}.pth')
        print(f"Actor Model saved to {f'{dir_path}/cp_actor_{model_nb}.pth'}")

    def train(self, checkpoint_step: int, visualize: bool = True):
        """Train the network

        Args:
            rewards_file (str): save file for rewards at each episode
            checkpoint_directory (str): directory for saving models
            checkpoint_step (int): model saving frequency
            visualize (bool, optional): Sets up the training visualisation. Defaults to True.
        """

        episode = 1
        rewards = []
        rewards_file = f"{self.working_directory}/rewards.txt"
        checkpoint_directory = f"{self.working_directory}/checkpoints"

        while episode <= self.max_episodes:
            reward = self.epoch()
            print("Episode ", episode, " - Reward : ", reward)
            rewards.append(reward)

            with open(rewards_file, mode="a", encoding="utf-8") as file:
                file.write(f"Epoch {episode} : Reward : {reward:.8f} ;\n")
                if episode % checkpoint_step == 0:
                    self.save_model(checkpoint_directory, episode)
            episode += 1


