# The Deep Deterministic Policy Gradient Algorithm

## Introduction to Notations

#### Environment and set of actions

Consider an environment $\mathcal{E}$ with a continuous action space. At each moment, the agent must choose an action from the action space $\mathcal{A}$. This continuous action space is the main difference between the environments on which DDPG and the previously seen algorithms such as DQN can work. Like in the previously seen algorithms, after executing this action, the state of $\mathcal{E}$ is modified. It should be noted that $\mathcal{E}$ is stochastic, and naturally, the algorithm will be capable of solving deterministic environments as they are simply a special case of stochastic environments.

In this type of environment, an action $a$ is a vector where each component is a floating number in a defined interval.

$a_t = \left(a_{1, t}, ...,a_{n, t} \right)$ where each $a_{i, t} \in \left[a_{{min}_i}, a_{{max}_i}\right]$ 
