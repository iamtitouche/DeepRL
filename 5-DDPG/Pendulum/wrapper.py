import gymnasium as gym


class Pendulum:

    def __init__(self, render_mode="human",reward_mode='original'):
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        self.reward_mode = reward_mode

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        return next_state, reward,  done, truncated, info



    