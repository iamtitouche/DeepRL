# The Twin Delayed Deep Deterministic Policy Gradient

## Prerequisites

Before reading this explanation of the TD3 algorithm, you should have read the description of the DDPG algorithm available [here](https://github.com/iamtitouche/DeepRL/tree/main/5-DDPG) or at least, have a good understanding of the functionning of this learning algorithm.

## The TD3 improvement

The difference between DDPG and TD3 resides in the use of a third network that we will note $V^2$. This network is a twin of the critic network whose have its own target network.

This network not used in the calculation of the actor loss but only in the critic one.

Now we have : $$T(s, a, r, d, s') = r + \gamma (1 - d) min \left(V_{target}(s', \pi_{target}(s')), V^2_{target}(s', \pi_{target}(s'))\right)$$

The loss of the critic is still given by : 

$$
L_{V}(\mathcal{B}) = \dfrac{1}{n}\sum_{(s, a, r, d, s') \in \mathcal{B}} \left(V(s, a) - T(s, a, r, d, s')\right)^2
$$

And we use a similar loss for the twin network :
$$
L_{V^2}(\mathcal{B}) = \dfrac{1}{n}\sum_{(s, a, r, d, s') \in \mathcal{B}} \left(V^2(s, a) - T(s, a, r, d, s')\right)^2
$$

*Note : If both critic networks were initialized with the exact same weights the algorithm would be equivalent to the DDPG algorithm.*