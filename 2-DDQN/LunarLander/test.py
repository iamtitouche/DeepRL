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
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DDQN')))
from ddqn import AgentDDQN

def test(agent, path, n_games, generate_gif=False):
    state = get_initial_state(agent.env, agent.state_shape, agent.device)

    seq = [agent.env.render()]
    i = 1

    done = False
    while not done or i < n_games:
        if done:
            state = get_initial_state(agent.env, agent.state_shape, agent.device)
            seq.append(agent.env.render())
            done = False
            i += 1

        next_state, reward, done = agent.step_testing(state)
        seq.append(agent.env.render())
        state = state_preprocess(next_state, agent.state_shape, state, agent.device)
    
    pil_images = [Image.fromarray(image) for image in seq]
    pil_images[0].save(f'{path}/lunarlander.gif',
                   save_all=True,
                   append_images=pil_images[1:],
                   duration=20,  # Durée en ms entre les frames
                   loop=0)

def parse_args():
    parser = argparse.ArgumentParser(description="DDQN Configuration with Hydra")
    parser.add_argument('--config_path', type=str, default="Training_Data_1", help="Path to the configuration directory")
    return parser.parse_args()

def main(cfg: DictConfig, config_path: str):
     # Utilisez config_path pour accéder au chemin
    env = gym.make("LunarLander-v2", render_mode="rgb_array")


    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))
    from network import network

    hyperparameters = {
        'optimizer_type': cfg.hyperparameters.optimizer_type,
        'mode_training': False,
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

    dqn = AgentDDQN(hyperparameters)
    dqn.load_model("Training_Data_1/checkpoints/cp_2500.pth")
    test(dqn, config_path, 3)

if __name__ == "__main__":
    args = parse_args()
    
    # Utilisez Hydra pour initialiser et composer la configuration
    with initialize(config_path=args.config_path):
        cfg = compose(config_name="config")
        main(cfg, args.config_path)
