import os
import sys
import torch
import hydra
from omegaconf import DictConfig
import gymnasium as gym
from wrapper import CartPole
from Cartpole_Processing import state_preprocess, get_initial_state
import argparse
from hydra import initialize, compose

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DuelingDQN')))
from dueling_dqn import AgentDuelingDQN

def parse_args():
    parser = argparse.ArgumentParser(description="Dueling DQN Configuration with Hydra")
    parser.add_argument('--config_path', type=str, default="Training_Data_1", help="Path to the configuration directory")
    return parser.parse_args()

def main(cfg: DictConfig, config_path: str):
     # Utilisez config_path pour accéder au chemin
    env = CartPole(render_mode=cfg.environment.render_mode, reward_mode=cfg.environment.reward_mode)


    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))
    from network import network, network_value, network_advantage

    hyperparameters = {
        'optimizer_type': cfg.hyperparameters.optimizer_type,
        'mode_training': cfg.hyperparameters.mode_training,
        'environment': env,
        'action_space_size': cfg.network.output_dim,
        'get_initial_state': get_initial_state,
        'state_preprocess': state_preprocess,
        'state_shape': tuple(cfg.hyperparameters.state_shape),
        'cnn': network,
        'value_network': network_value,
        'advantage_network': network_advantage,
        'advantage_mode': 'mean',
        'learning_rate': cfg.hyperparameters.learning_rate,
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

    ddqn = AgentDuelingDQN(hyperparameters)
    ddqn.train(int(cfg.hyperparameters.max_episodes/10))

if __name__ == "__main__":
    args = parse_args()
    
    # Utilisez Hydra pour initialiser et composer la configuration
    with initialize(config_path=args.config_path):
        cfg = compose(config_name="config")
        main(cfg, args.config_path)
