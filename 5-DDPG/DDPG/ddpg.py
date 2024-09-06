import copy
import enum
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import time

from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork


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

        self.get_initial_state = hyperparams_dict["get_initial_state"]
        self.state_preprocess = hyperparams_dict["state_preprocess"]

        self.working_directory = hyperparams_dict["working_directory"]

        if not os.path.exists(f"{self.working_directory}/checkpoints"):
            os.makedirs(f"{self.working_directory}/checkpoints")
            print("Checkpoints directory created")

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
            action = torch.max(torch.min(action, torch.tensor([[1]]).to(self.device)),
                               torch.tensor([[-1]]).to(self.device))
        return action

    def step_training(self, state):
        action = self.choose_action_training(state).squeeze(0)
        result = self.env.step(np.array(action.cpu()))
        return result[0], result[1], result[2], result[3], action

    def step_testing(self, state):
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0)).squeeze(0)

        result = self.env.step(np.array(action.cpu()))

        return result[0], result[1], result[2]

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
        self.critic.optimizer.step()

        if update_actor:
            actions = self.actor(states)
            critic_value = self.critic(states, actions)
            actor_loss = - critic_value.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

    def load_model(self, dir_path: str, model_nb):
        """Sauvegarde les poids et biais du modèle dans un fichier."""
        self.critic.network.load_state_dict(torch.load(f'{dir_path}/cp_critic_{model_nb}.pth'))
        self.actor.network.load_state_dict(torch.load(f'{dir_path}/cp_actor_{model_nb}.pth'))
        print("Actor Model loaded to")

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


