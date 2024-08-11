# Lunar Lander Solved With DQN

### What is Lunar Lander ?

![LunarLander](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/DQN/LunarLander/lunar_lander.gif)

LunarLander is a classic rocket trajectory optimization problem.

#### Action Space

There are four discrete actions available:

- $0$ : do nothing
- $1$ : fire left orientation engine
- $2$ : fire main engine
- $3$ : fire right orientation engine

#### Rewards

For each step, the reward:

- is increased/decreased the closer/further the lander is to the landing pad.

- is increased/decreased the slower/faster the lander is moving.

- is decreased the more the lander is tilted (angle not horizontal).

- is increased by $10$ points for each leg that is in contact with the ground.

- is decreased by $0.03$ points each frame an engine is firing.


### Results

![FrozenLake](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/DQN/LunarLander/Training_Data_1/rewards.png)
