# Pendulum Solved With DDPG

### What is Pendulum ?

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


### DQN Training Results for the non-slippery Mode

Here are the results of the training on the 4x4 non-slippery map using both the original and modified reward structures. To ensure the data from both training scenarios are comparable, we plotted the original rewards that the agent would have received during the training with the modified rewards.

| Original Rewards | Modified Rewards |
|:---------------------:|:-------------------:|
| ![4x4-false-original](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/FrozenLake/Training_Data_1/rewards.png) | ![4x4-false-modified](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/FrozenLake/Training_Data_5/rewards.png) |

From these results, we observe that for this specific configuration, the modified reward structure does not lead to improved training outcomes compared to the original rewards.

Next, we examine the data obtained from training on the 8x8 non-slippery map.

| Original Rewards | Modified Rewards |
|:---------------------:|:-------------------:|
| ![8x8-false-original](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/FrozenLake/Training_Data_3/rewards.png) | ![8x8-false-modified](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/FrozenLake/Training_Data_7/rewards.png) |

In this case, it appears that the modified rewards enable the agent to learn more quickly, suggesting an improvement in training efficiency with the modified reward structure.

### Analysis of the Slippery 4x4 map

The FrozenLake environment being very simple we can find the ideal policy by analysing the map, and so even in the mode slippery.

To begin with, the following map shows in yellow the choices of actions that cannot lead to death and in red the safest choices of actions when no choice is completely safe.

| Mapping of the safe actions |
|:---------------------:|
|![safest_choices](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/FrozenLake/safest_choices.png)|

We can see for almost every tile a safe choice exist, for example on the tile bellow the staring point we can purposely choose to go into the left border and by doing so we are sure to avoid death and we have a little chance to come closer to the objective tile. We can notice that only one tile can lead to death, so our new objective is to limit our choices to actions that will not lead to death neither this dangerous tile if some other choice is available.

| Mapping of best policy |
|:----------------------:|
|![safest_choices](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/FrozenLake/best_policy_by_hand.png)|

We can notice that for the tile directly above the dangerous one, chosing to go up give us the guarantee to avoid death.
But for the tile directly bellow, going left is the only safe choice so even if it can lead us to the dangerous tile it 
is still better than every other choice that could lead to a direct death. So now we should also make changes to avoid this
tile, that is why when on the tile just before the ending we chose to go down. We also can change, the starting action, 
we notice that if we go up we are condemned to stay on the upper row and never end the game, so we have to choose 
between one of the three other choice. We choose to go left because, event if all three choices are safe, it is the only one that prevent us to go farther
into the upper row and so farther from the ending.

#### Actions Map of a Trained Model