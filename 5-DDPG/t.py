import gymnasium as gym

import numpy as np

env = gym.make('Pendulum-v1', render_mode="rgb_array")

s = env.reset()
img = env.render()

print(img)
i = 0
while True:
    print(i)
    i += 1
    s = env.step(np.array([0.1]))
    print(s[2], " ", s[3])
    if s[3]:
        break

    img = env.render()
