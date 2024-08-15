# The Deep Q-Network Algorithm

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

In the Deep Q-Network (DQN) algorithm, the policy is implicitly defined through a Q-function wich gives its name the algorithm. This Q-function evaluates the expected cumulative discounted reward of taking action $a$ in state $s$ and following the optimal policy thereafter. Instead of directly mapping states to actions, DQN uses this Q-function to decide which action to take.

$$
Q(s_t) = \left(Q(s_t, a)\right)_{a \in \mathcal{A}} $$

$$ Q(s, a) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{k+t} \mid s_t, a_t = a\right]
$$

Here $\gamma \in [0, 1]$ is the discount factor and $r_k$ is the reward provided by the environnement after taking the action $a_k$.

The discount factor determines the importance of future rewards. A higher $\gamma$ indicates that future rewards are valued more, while a lower $\gamma$ places more emphasis on immediate rewards.

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

These tuples, often referred to as "experiences" or "transitions", are stored in the buffer during the agent's interaction with the environment. When the buffer reaches its capacity $c$, the oldest experiences are discarded to make room for new ones.

Note : in my implementation of the DQN algorithm, I chose to remember $1 - d_t$ instead of $d_t$

##### Why Use a Replay Buffer?

The replay buffer serves multiple purposes that are essential for the success of the Deep Q-Network:

- Breaking Correlations:
When an agent learns from consecutive experiences, the data points are highly correlated. Learning directly from such correlated data can lead to poor generalization and slow learning. The replay buffer addresses this by allowing the agent to learn from a batch of random past experiences, thereby breaking the correlations and improving the robustness of the learning process.


- Efficient Use of Data:
The replay buffer allows the agent to reuse experiences multiple times, which leads to more efficient use of the data and faster convergence especially  when gathering new experiences is costly.

- Stabilizing Training:
Training neural networks with non-stationary data (data distribution that changes over time) can be challenging and may result in unstable learning. The replay buffer helps to stabilize training by providing a more stationary distribution of experiences to the neural network. Since experiences are sampled randomly from the buffer, the distribution of training data becomes more consistent over time.

## The Learning Algorithm

The pseudo-code below broadly outlines the functioning of the learning process of the DQN algorithm.

```
Begin
    Initialise Q-Network and buffer

    For episode 1 to max_episode
        state = environment.reset()
        While not done
            action = choose_action(state)

            next_state, reward, done = environment.step(action)

            buffer.store(state, action, reward, done, next_state)

            If buffer.size >= batch_size
                batch = buffer.sample(batch_size)
                replay_experience(batch)
            End If
        End While
    End For
End
```

#### The Experience Replay

After sampling a random batch $\mathcal{B}$ of $n$ experiences the algorithm learns from this selected batch through a process called the experience replay.

Through this process the algorithm will evaluate, for each experience $(s, a, , r, d, s')$ how far is $Q(s, a)$ from an expected value called the target $T(s, a, r, d, s')$. We calculate a loss for the full batch by using a loss function such as the MSE (Mean Squared Error) :

$$
L(\mathcal{B}) = \dfrac{1}{n}\sum_{(s, a, r, d, s') \in \mathcal{B}} \left(Q(s, a)  - T(s, a, r, d, s')\right)^2
$$

Note : Other loss functions such as Huber Loss or others can also be used, but the MSE is the most commonly used one and the choice of the loss function does not really impact the learning process

After calculating this loss we use the gradient descent algorithm to optimize the parameters of the Q-Network.

#### The Target 

To find the formula of the target, we have to think about what we want the Q-function to represent. We want $Q(s, a)$ to be the expected cumulative discounted reward of taking action $a$ in state $s$ and following the optimal policy thereafter.

$$Q(s, a) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{k+t} \mid s, a\right]$$

$$Q(s, a) = \mathbb{E}\left[r_t + \gamma (1 - d) max_{a' \in \mathcal{A}}(Q(s', a')) \mid s, a\right]$$

Ainsi pour faire converger les paramètres de Q vers le paramétrage idéal, on prendra :

$$T(s, a, r, d, s') = r + \gamma (1 - d) max_{a' \in \mathcal{A}}(Q(s', a'))$$

In practice, a target network is often used to stabilize the training process. This target network is initialized with the same parameters as the Q-network but is updated less frequently. By holding the target network's parameters constant for several training steps before updating them, this technique reduces oscillations and divergence during training. As a result, it provides a significant improvement in the learning process, making it more stable and allowing the algorithm to converge more effectively. We end up with the following loss function :

$$
L(\mathcal{B}) = \dfrac{1}{n}\sum_{(s, a, r, d, s') \in \mathcal{B}} \left(Q(s, a)  - r - \gamma (1 - d) max_{a' \in \mathcal{A}}(Q_{target}(s', a'))\right)^2
$$

##### Updating the Target-Network

We define an update frequency $f$, which determines how often the target network is updated. Specifically, the target network's parameters are updated every $f$ actions taken by the agent. There are different methods for performing this update. One common approach is the hard update, where the target network's parameters are completely replaced by the Q-network's parameters. Another approach is the soft update, where the target network's parameters are gradually updated by blending a small portion of the Q-network's parameters into the existing target network parameters. Both methods aim to stabilize the learning process, with the hard update providing a clear distinction between updates, while the soft update offers smoother transitions.

- Hard Update:
$\theta_{target} \leftarrow \theta$


- Soft Update:
$\theta_{target} \leftarrow \tau \theta + (1 - \tau)\theta_{target}$ with $\tau \in \left]0, 1\right[$  


![Parameter_Evolution](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/DQN/parameter_evolution.png)

This graph illustrates the evolution of a Q-Network weight compared to its corresponding weight in the Target-Network, using both hard updates (with $f=10$) and soft updates (with $f=2,\tau=0.1$).

A third method exists and is implemented in my code. It is called t-soft update and it basically consist in a soft update where the parameter $\tau$ evolves through time.


#### How to choose the action to take during training ?

The training process aims to find the optimal policy to solve an environment. So how can the agent explore its environment to gather diverse experiences which it can learn from ? The training process will be divided in two parts, the exploration during which the agent will not follow its policy because it is still far from optimal and then the exploitation during which the agent will follow its policy to some certain extend. To do so, there are two main methods :

##### The $\epsilon$-greedy exploration

At each step the agent will choose its next action, either completly randomly with a probability $\epsilon$ or by following its policy with a probability $1-\epsilon$.

The value of epsilon is initialized at a high probability value (often $1$), and is slowly decreased after each game until it reaches a minimum value.

This method allows, to choose mostly random actions in the first games, and slowly following more and more policy as it becomes more and more accurate.

##### The softmax exploration

With the softmax method, we take the output of the Q-Network and convert each component of the output vector to a probability being the probability of the corresponding action to be chosen. To convert these values to probabilities we use the softmax function : 

$$\forall a \in \mathcal{A}, P(a) = \dfrac{exp(Q(s, a))}{\sum_{a' \in \mathcal{A}} exp(Q(s, a'))}$$

We can also, introduce a parameter $\lambda > 0$ called the inverse temperature (because $1/\lambda$ is called the temperature), that will limit or encourage the exploration depending on its chosen value.

$$\forall a \in \mathcal{A}, P(a) = \dfrac{exp(\lambda Q(s, a))}{\sum_{a' \in \mathcal{A}} exp(\lambda Q(s, a'))}$$

The lower $\lambda$, the higher the entropy of the probability distribution of the actions, meaning the closer this distribution gets to a uniform distribution.

![Network Output](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/DQN/q_network_output.png)

|:------------------:|:------------------:|:-------------------------:|
| ![Proba 1](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/DQN/proba_1.png) | ![Proba 2](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/DQN/proba_2.png) | ![Proba 3](https://raw.githubusercontent.com/iamtitouche/DeepRL/main/1-DQN/DQN/proba_3.png) |

Note : the softmax function cannot be used as an activation function at the end of the Q-Network because we still need the output of this network to approximate the value of the expected cumulated and discounted rewards