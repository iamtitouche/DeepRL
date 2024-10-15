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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '5-DDPG', 'DDPG')))
from ddpg import AgentDDPG
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

from optimizer import clip_gradients


class AgentTD3(AgentDDPG):

    def __init__(self, hyperparams_dict):
        super().__init__(hyperparams_dict)

        self.critic_twin = CriticNetwork(hyperparams_dict, self.device)
        self.critic_twin_target = copy.deepcopy(self.critic_twin)

        
    
    def replay_experience(self, update_actor) -> None:
        states, actions, rewards, not_dones, next_states = self.replay_buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)

            target = self.critic_target(next_states, next_actions)
            target = torch.min(critic_target, self.critic_twin_target(next_states, next_actions))

            target = rewards + self.discount_factor * critic_target * not_dones

        critic_value = self.critic(states, actions)
        critic_twin_value = self.critic_twin(states, actions)
        critic_loss = nn.MSELoss()(target, critic_value) + nn.MSELoss()(target, critic_twin_value)

        self.critic.optimizer.zero_grad()
        self.critic_twin.optimizer.zero_grad()
        critic_loss.backward()
        clip_gradients(self.critic, self.critic_grad_clipping_method, self.critic_grad_clipping_threshold)
        clip_gradients(self.critic_twin, self.critic_grad_clipping_method, self.critic_grad_clipping_threshold)
        self.critic.optimizer.step()
        self.critic_twin.optimizer.step()

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
        self.critic_twin.network.load_state_dict(torch.load(f'{dir_path}/cp_critic_twin_{model_nb}.pth'))
        self.actor.network.load_state_dict(torch.load(f'{dir_path}/cp_actor_{model_nb}.pth'))
        print("Actor Model loaded")

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_twin_target = copy.deepcopy(self.critic_twin)

    def save_model(self, dir_path: str, model_nb):
        """Sauvegarde les poids et biais du modèle dans un fichier."""
        torch.save(self.critic.network.state_dict(), f'{dir_path}/cp_critic_{model_nb}.pth')
        print(f"Critic Model saved to {f'{dir_path}/cp_critic_{model_nb}.pth'}")

        torch.save(self.critic_twin.network.state_dict(), f'{dir_path}/cp_critic_twin_{model_nb}.pth')
        print(f"Critic Twin Model saved to {f'{dir_path}/cp_critic_twin_{model_nb}.pth'}")

        torch.save(self.actor.network.state_dict(), f'{dir_path}/cp_actor_{model_nb}.pth')
        print(f"Actor Model saved to {f'{dir_path}/cp_actor_{model_nb}.pth'}")


