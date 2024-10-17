import os
import sys
import torch
import hydra
from omegaconf import DictConfig
import gymnasium as gym
from wrapper import Pendulum
from Pendulum_Processing import state_preprocess, get_initial_state
import argparse
from hydra import initialize, compose

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DDPG')))
from ddpg import AgentDDPG

def parse_args():
    parser = argparse.ArgumentParser(description="DDPG Configuration with Hydra")
    parser.add_argument('--config_path', type=str, default="Training_Data_1", help="Path to the configuration directory")
    return parser.parse_args()

def main(cfg: DictConfig, config_path: str):
    env = Pendulum(cfg.environment.render_mode, cfg.environment.reward_mode)

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))
    from network import network_actor, network_critic_state, network_critic_action, network_critic_final

    hyperparameters = {
        'optimizer_type': cfg.hyperparameters.optimizer_type,
        'mode_training': cfg.hyperparameters.mode_training,
        'environment': env,
        'n_actions': 1,
        'action_space_size': cfg.hyperparameters.n_actions,
        'get_initial_state': get_initial_state,
        'state_preprocess': state_preprocess,
        'state_shape': tuple(cfg.hyperparameters.state_shape),
        'actor_network': network_actor,
        'critic_network_state': network_critic_state,
        'critic_network_action': network_critic_action,
        'critic_network_final': network_critic_final,
        'learning_rate_actor': cfg.hyperparameters.learning_rate_actor,
        'learning_rate_critic': cfg.hyperparameters.learning_rate_critic,
        'discount_factor': cfg.hyperparameters.discount_factor,
        'max_episodes': cfg.hyperparameters.max_episodes,
        'memory_capacity': cfg.hyperparameters.memory_capacity,
        'batch_size': cfg.hyperparameters.batch_size,
        'update_frequency': cfg.hyperparameters.update_frequency,
        'update_mode': cfg.hyperparameters.update_mode,
        'tau': cfg.hyperparameters.tau,
        'nu': cfg.hyperparameters.nu,
        'working_directory': config_path,
        'critic_grad_clipping_method': cfg.hyperparameters.critic_grad_clipping_method,
        'critic_grad_clipping_threshold': cfg.hyperparameters.critic_grad_clipping_threshold,
        'actor_grad_clipping_method': cfg.hyperparameters.actor_grad_clipping_method,
        'actor_grad_clipping_threshold': cfg.hyperparameters.actor_grad_clipping_threshold,
        'action_lower_bound': torch.tensor(cfg.hyperparameters.action_lower_bound),
        'action_upper_bound': torch.tensor(cfg.hyperparameters.action_upper_bound)
    }

    agent = AgentDDPG(hyperparameters)
    agent.train(cfg.hyperparameters.checkpoint_step)

if __name__ == "__main__":
    args = parse_args()
    
    # Utilisez Hydra pour initialiser et composer la configuration
    with initialize(config_path=args.config_path):
        cfg = compose(config_name="config")
        main(cfg, args.config_path)
