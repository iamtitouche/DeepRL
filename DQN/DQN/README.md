# The Deep Q-Learning Algorithm

## Introduction to Notations

#### Environment and set of actions

Consider an environment $\mathcal{E}$ with a discrete action space. At each moment, the agent must choose an action from the discrete space of $K$ possible actions $\mathcal{A} = \lbrace 1, \dots, K \rbrace$. After executing this action, the state of $\mathcal{E}$ is modified. It is noted that $\mathcal{E}$ is stochastic, and naturally, the algorithm will be capable of solving deterministic environments as they are simply a special case of stochastic environments.

#### Observations and states

At each state, the environment will return an observation noted $x_t$. Depending on the environment, this observation can be of different types.

##### Types of Observations Across Different Environments

| Environment      | Type of Observation                                    | Example Details                                                                 |
|------------------|--------------------------------------------------------|---------------------------------------------------------------------------------|
| **FrozenLake**   | Integer                                                | The observation is an integer representing the tile number where the agent is located. |
| **LunarLander**  | Vector                                                 | The observation is a vector containing the angle, speed, rotational speed, and other relevant quantities. |
| **Super Mario Bros** | Set of Pixels                                          | The observation is the set of pixels on the screen, which represents the current frame of the game. |


It can be noted that by only receiving the last observation as input, crucial information will be missing for our agent. For instance, in Super Mario Bros, if the agent is provided with only the current frame, it cannot deduce Mario's speed and direction. Similarly, in LunarLander, while the agent has access to speed as part of the observation, it lacks information about acceleration, which is crucial for counteracting gravity. To address this issue, we provide the agent with a sequence of observations rather than just a single observation. It should be noted that in environments like FrozenLake, where previous states of the game do not influence the future, it is unnecessary to provide a sequence as input.

Thus, the sequence leading to the observation $x_t$ will be denoted as $s_t$ and called "state". We will then have $s_t = \left(\phi_{t-k}, \dots, \phi_{t-1}, \phi_t\right)$ where $\phi_t = \phi(x_t)$ with $\phi$ being the data preprocessing function. For example, in an environment like Super Mario Bros, it is common practice to reduce the input image resolution and convert it to grayscale as is it to allows improve the speed of the learning process by reducing the size of the entries.

#### Policy

The policy refers to a strategy or a mapping from states of the environment to actions to be taken when in those states. More formally, a policy, denoted by $\pi$, is a function associating an action to a state.

In the Deep Q-Network (DQN) algorithm, the policy is implicitly defined through a Q-function wich gives its name the algorithm. This Q-function evaluates the expected cumulative reward of taking action $a$ in state $s$ and following the optimal policy thereafter. Instead of directly mapping states to actions, DQN uses this Q-function to decide which action to take.

$$
Q(s_t) = \left(Q(s_t, a)\right)_{a \in \mathcal{A}} = \left(\mathbb{E}\left[\sum_{k=t}^{\infty} \gamma^k r_k \mid s_t, a_t = a\right]\right)_{a \in \mathcal{A}}
$$

Here $\gamma$ is the discount factor and $r_k$ is the reward returned by the environnement after choosing the action $a_k$.

For a given state $s$, the Q-function returns a vector of values, where each component of the vector corresponds to a specific action $a$ from the set of possible actions $\mathcal{A}$. The action chosen by the policy is typically the one with the highest Q-value :

$$\pi(s) = Argmax_{a \in \mathcal{A}}\left(Q(s, a)\right)$$

#### Q-Network

The main challenge in DQN is that the Q-function is typically too complex to compute exactly. To overcome this, we use a deep neural network to approximate the Q-function. The parameters of this neural network are denoted by $\theta$. Consequently, we refer to the policy and the Q-function that rely on these parameters as $\pi_{\theta}$ and $Q_{\theta}$​, respectively.

The learning process in DQN involves adjusting the parameters so that the Q-network accurately approximates the true Q-function. This allows the agent to make decisions that maximize its expected cumulative reward.

#### Replay Buffer

In Deep Q-Learning, one of the key innovations that significantly enhances the learning process is the Replay Buffer. The replay buffer is a crucial component that stores past experiences of the agent, which are then reused during training. This helps to improve the stability and efficiency of the learning process.

##### What is a Replay Buffer?

A replay buffer is a finite-sized memory that stores tuples of the form $(s_{t},a_t,r_t, d_t, s_{t+1})$, where:

- $s_t​$ is the state observed at time step $t$
- $a_t$ is the action taken by the agent at time step $t$
- $r_t$​ is the reward received after taking action $a_t​$
- $d_t$ is a boolean value indicating if the gmae is over (True) or not (False) 
- $s_{t+1}$​ is the resulting state after taking action $a_t$​

These tuples, often referred to as "experiences" or "transitions", are stored in the buffer during the agent's interaction with the environment. When the buffer reaches its capacity, the oldest experiences are discarded to make room for new ones.

Note : in my implementation of the DQN algorithm I chose to remember $1 - d_t$ instead of $d_t$



