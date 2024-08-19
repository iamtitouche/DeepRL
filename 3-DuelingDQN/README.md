# The Dueling Deep Q-Network Algorithm

## Prerequisites

Before reading this explanation of the Dueling DQN algorithm, you should have read the description of the DQN algorithm available [here](https://github.com/iamtitouche/DeepRL/tree/main/1-DQN/DQN) or at least, have a good understanding of the functionning of this learning algorithm.

## The Dueling DQN improvement

The difference between DQN and Dueling DQN resides in the Q-Network Architecture : 

| **Dueling Network Architecture**      |
|---------------------------------------|
|![Network Architecture](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/3-DuelingDQN/DuelingDQN/dueling_network_architecture.png)|

In this new architecture, the network is divided in two parts one giving a vector as output and the other giving a simple scalar.
The vector $A(s)$ is called vector of advantages and $V(s)$ is called the value of the state $s$.

In this new architecture, $Q(s,a)$ is defined as follows :

$$Q(s, a) = V(s) + A(s, a) - max_{a' \in \mathcal{A}}\left(A(s, a')\right)$$

*Note : instead of using the max function we can use the average of the components of $A(s)$.*