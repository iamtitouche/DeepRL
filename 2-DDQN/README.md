# The Double Deep Q-Network Algorithm

## Prerequisites

Before reading this explanation of the DDQN algorithm, you should have read the description of the DQN algorithm available [here](https://github.com/iamtitouche/DeepRL/tree/main/1-DQN/DQN) or at least, have a good understanding of the functionning of this learning algorithm.

## The Double DQN improvement

The only difference between DQN and DDQN resides in the way the target is calculated.

| | DQN     | DDQN                                 |
|----|------------------|--------------------------------------------------------|
| **Target Formula** | $$r + \gamma (1 - d) \max_{a' \in \mathcal{A}}(Q_{target}(s', a'))$$ | $$r + \gamma (1 - d)Q_{target}(s', Argmax_{a' \in \mathcal{A}}(Q(s', a')))$$     |
| **Advantages** |- Simpler and faster to compute<br><br> | - More stable, requiring less precise tuning of the hyperparameters<br>- More efficient on complex environments<br><br> |
| **Disadvantages** |- Overestimation of the Q function which can lead to non-optimal policies<br><br> | - Slower to compute without major improvements of the results on simple environments<br><br> |

In DQN : $\max_{a' \in \mathcal{A}}(Q_{target}(s', a'))$ can be understood as the value of the best next action according to the Target-Network

Whereas in DDQN : $Q_{target}(s', Argmax_{a' \in \mathcal{A}}(Q(s', a')))$ can be interpreted as the value according to the Target-Network of the best next action according to the current policy