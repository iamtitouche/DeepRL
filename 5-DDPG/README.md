# The Deep Deterministic Policy Gradient Algorithm

## Introduction to Notations

#### Environment and set of actions

Consider an environment $\mathcal{E}$ with a continuous action space. At each moment, the agent must choose an action from the action space $\mathcal{A}$. This continuous action space is the main difference between the environments on which DDPG and the previously seen algorithms such as DQN can work. Like in the previously seen algorithms, after executing this action, the state of $\mathcal{E}$ is modified. It should be noted that $\mathcal{E}$ is stochastic, and naturally, the algorithm will be capable of solving deterministic environments as they are simply a special case of stochastic environments.

In this type of environment, an action $a_t$ is a vector where each component is a floating number in a defined interval.

$a_t = \left(a_{1, t}, ...,a_{n, t} \right)$ where each $a_{i, t} \in \left[a_{{min}_i}, a_{{max}_i}\right]$ 

## Networks
In this algorithm, two types of neural networks are used: one for determining the action to take, and the other, used only during training, to evaluate the quality of a state-action pair.

The actor network, denoted as $\pi$, is responsible for selecting an action based on the current state. The critic network, denoted as $V$, estimates the value of the chosen action in the given state, providing feedback to improve the actor's training.

The actor network $\pi$ uses an activation function that ensures its outputs remain within a predefined range, such as tanh or sigmoid, constraining the scale of the actions produced. This will allow to easily rescaled every component of the action vector in the desired range :

$$a_{rescaled}(s) = a_{min} + \left(\pi(s) - activation_{min}\right) \dfrac{a_{max} - a_{min}}{activation_{max} - activation_{min}}$$

The critic network takes two inputs, a state $s$ and an action $a$, and it outputs a scalar. To do so, this network is decomposed in three subnetworks, on taking the state $s$ as input and the other taking the action $a$. both output vectors that we concatenate befor using the result as input for the third network returning the desired scalar.