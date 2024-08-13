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

Thus, the sequence leading to the observation $x_t$ will be denoted as $s_t$ and called "state". We will then have $s_t = \left(\phi_{t-k}, \dots, \phi_{t-1}, \phi_t\right)$ where $\phi_t = \phi(x_t)$ with $\phi$ being the data preprocessing function. For example, in an environment like Super Mario Bros, it is common practice to reduce the input image resolution and convert it to grayscale as is it allows improve the speed of the learning process by reducing the size of the entries.

