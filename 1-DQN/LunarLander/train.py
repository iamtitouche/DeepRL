import os
import sys
import torch
import hydra
from omegaconf import DictConfig
import gymnasium as gym
import gymnasium as gym
from LunarLander_Processing import state_preprocess, get_initial_state
import argparse
from hydra import initialize, compose

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DQN')))
from dqn import AgentDQN

def parse_args():
    parser = argparse.ArgumentParser(description="DQN Configuration with Hydra")
    parser.add_argument('--config_path', type=str, default="Training_Data_1", help="Path to the configuration directory")
    return parser.parse_args()

def main(cfg: DictConfig, config_path: str):
     # Utilisez config_path pour accéder au chemin
    env = gym.make("LunarLander-v2", render_mode=cfg.environment.render_mode)


    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))
    from network import network

    hyperparameters = {
        'optimizer_type': cfg.hyperparameters.optimizer_type,
        'mode_training': cfg.hyperparameters.mode_training,
        'environment': env,
        'action_space_size': cfg.network.output_dim,
        'get_initial_state': get_initial_state,
        'state_preprocess': state_preprocess,
        'state_shape': tuple(cfg.hyperparameters.state_shape),
        'network': network,
        'learning_rate': cfg.hyperparameters.learning_rate,
        'clip_grad_norm': cfg.hyperparameters.clip_grad_norm,
        'discount_factor': cfg.hyperparameters.discount_factor,
        'max_episodes': cfg.hyperparameters.max_episodes,
        'memory_capacity': cfg.hyperparameters.memory_capacity,
        'batch_size': cfg.hyperparameters.batch_size,
        'update_frequency': cfg.hyperparameters.update_frequency,
        'update_mode': cfg.hyperparameters.update_mode,
        'exploration_mode': cfg.hyperparameters.exploration_mode,
        'tau': cfg.hyperparameters.tau,
        'nu': cfg.hyperparameters.nu,
        'tau_softmax': cfg.hyperparameters.tau_softmax,
        'epsilon': cfg.hyperparameters.epsilon,
        'epsilon_min': cfg.hyperparameters.epsilon_min,
        'epsilon_decay': cfg.hyperparameters.epsilon_decay,
        'working_directory': config_path,
        'grad_clipping_method': cfg.hyperparameters.grad_clipping_method,
        'grad_clipping_threshold': cfg.hyperparameters.grad_clipping_threshold
    }

    dqn = AgentDQN(hyperparameters)
    dqn.train(cfg.hyperparameters.max_episodes)

if __name__ == "__main__":
    args = parse_args()
    
    # Utilisez Hydra pour initialiser et composer la configuration
    with initialize(config_path=args.config_path):
        cfg = compose(config_name="config")
        main(cfg, args.config_path)
