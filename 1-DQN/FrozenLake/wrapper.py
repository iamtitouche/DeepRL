import gymnasium as gym


class FrozenLake:

    def __init__(self, render_mode="human", is_slippery=False, map_name='4x4', reward_mode='original'):
        self.env = gym.make("FrozenLake-v1", render_mode=render_mode, is_slippery=is_slippery, map_name=map_name)
        self.reward_mode = reward_mode

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        if self.reward_mode == 'penalty_for_death' and done and reward == 0:
            reward = -1
        return next_state, reward,  done, truncated, info


if __name__ == "__main__":
    f = FrozenLake()

    f.reset()

    print(f.step(1))

    