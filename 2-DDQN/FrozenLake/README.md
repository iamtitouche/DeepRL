# Frozen Lake Solved With DQN

### What is Frozen Lake ?

![FrozenLake](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/FrozenLake/frozen_lake.gif)

FrozenLake is a game where the player must cross a frozen lake and reach the gift without fallling in one of the holes.

#### Action Space

The agent takes a 1-element vector for actions. The action space is (dir), where dir decides direction to move in which can be:

- 0: LEFT
- 1: DOWN
- 2: RIGHT
- 3: UP

#### Rewards

After each action, the agent receive the following reward :

- $1$ : if it reaches the goal
- $0$ : in every other situation

#### Slippery

The game can be played in two different modes, depending on the value of the boolean is_slippery.

If is_slippery is True, the agent will move in intended direction with probability of $1/3$ else will move in either perpendicular direction with equal probability of $1/3$ in both directions.

#### Improvements

I created a environment wrapper for FrozenLake. This allowed me to add an argument : reward_mode.

If set to "original", the agent will be receiving the original FrozenLake rewards.

If set to "penalty_for_death", the agent will receive the following reward :

- $1$ : if it reaches the goal
- $-1$ : if it falls in a hole
- $0$ : in every other situation


