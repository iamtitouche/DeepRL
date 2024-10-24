import os
import sys
import torch
import hydra
from omegaconf import DictConfig
import gymnasium as gym
from Pendulum_Processing import state_preprocess, get_initial_state
import argparse
from hydra import initialize, compose
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DDPG')))
from ddpg import AgentDDPG

def test(agent, path, n_games, generate_gif=False):
    state = get_initial_state(agent.env, agent.state_shape, agent.device)

    seq = [agent.env.render()]
    i = 1

    done, truncated = False, False
    while (not done and not truncated) or i < n_games:
        if done or truncated:
            state = get_initial_state(agent.env, agent.state_shape, agent.device)
            seq.append(agent.env.render())
            done = False
            truncated = False
            i += 1

        next_state, reward, done, truncated = agent.step_testing(state)
        print(done, truncated)
        seq.append(agent.env.render())
        state = state_preprocess(next_state, agent.state_shape, state, agent.device)
    
    pil_images = [Image.fromarray(image) for image in seq]
    pil_images[0].save(f'{path}/pendulum.gif',
                   save_all=True,
                   append_images=pil_images[1:],
                   duration=20,  # Durée en ms entre les frames
                   loop=0)
def parse_args():
    parser = argparse.ArgumentParser(description="DDPG Configuration with Hydra")
    parser.add_argument('--config_path', type=str, default="Training_Data_1", help="Path to the configuration directory")
    return parser.parse_args()

def main(cfg: DictConfig, config_path: str):
     # Utilisez config_path pour accéder au chemin
     env = gym.make(cfg.environment.name,
         render_mode="rgb_array"
     )

     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))
     from network import network_actor, network_critic_state, network_critic_action, network_critic_final

     hyperparameters = {
         'optimizer_type': cfg.hyperparameters.optimizer_type,
         'mode_training': cfg.hyperparameters.mode_training,
         'environment': env,
         'n_actions': 1,
         'action_space_size': cfg.network.output_dim,
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

     agent = AgentDDPG(hyperparameters)
     agent.load_model("Training_Data_2/checkpoints", 500)
     test(agent, config_path, 1)

if __name__ == "__main__":
    args = parse_args()
    
    # Utilisez Hydra pour initialiser et composer la configuration
    with initialize(config_path=args.config_path):
        cfg = compose(config_name="config")
        main(cfg, args.config_path)
