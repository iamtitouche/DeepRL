import gymnasium as gym


class CartPole:

    def __init__(self, render_mode="human", reward_mode='original'):
        self.env = gym.make("CartPole-v0", render_mode=render_mode)
        self.reward_mode = reward_mode

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        #if self.reward_mode == 'penalty_for_death' and done and reward == 0:
        #    reward = -1
        return next_state, reward,  done, truncated, info




    