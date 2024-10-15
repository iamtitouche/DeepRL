## The Deep Deterministic Policy Gradient Algorithm

### Introduction to Notations

#### Environment and Set of Actions

Consider an environment $\mathcal{E}$ with a continuous action space. At each moment, the agent must choose an action from the action space $\mathcal{A}$. This continuous action space is the main difference between the environments on which DDPG and previously seen algorithms, such as DQN, can work. As in previously seen algorithms, after executing this action, the state of $\mathcal{E}$ is modified. It should be noted that $\mathcal{E}$ is stochastic, and naturally, the algorithm will be capable of solving deterministic environments, as they are simply a special case of stochastic environments.

In this type of environment, an action $a_t$ is a vector where each component is a floating number within a defined interval:

$a_t = \left( a_{1,t}, a_{2,t}, \dots, a_{n,t} \right)$
where each component $a_{i,t}$ belongs to the interval $\left[ a_{{\text{min}}_i}, a_{{\text{max}}_i} \right]$.


#### Networks
In this algorithm, two types of neural networks are used: one for determining the action to take, and the other, used only during training, to evaluate the quality of a state-action pair.

The actor network, denoted as $\pi$, is responsible for selecting an action based on the current state. The critic network, denoted as $V$, estimates the value of the chosen action in the given state, providing feedback to improve the actor's training.

The actor network $\pi$ uses an activation function that ensures its outputs remain within a predefined range, such as tanh or sigmoid, constraining the scale of the actions produced. This will allow to easily rescaled every component of the action vector in the desired range :

$$a_{rescaled}(s) = a_{min} + \left(\pi(s) - activation_{min}\right) \dfrac{a_{max} - a_{min}}{activation_{max} - activation_{min}}$$

The critic network takes two inputs, a state $s$ and an action $a$, and it outputs a scalar. To do so, this network is decomposed in three subnetworks, on taking the state $s$ as input and the other taking the action $a$. both output vectors that we concatenate before using the result as input for the third network returning the desired scalar.

It is common practice for the actor and critic networks to share a subnetwork in their architecture. This subnetwork, which forms the initial part of the actor, takes the state of the environment as input. Its output is then used to produce the action in the actor network, while also being passed into the critic network to evaluate the state-action pair. This allows the critic to use the same feature representation of the state as the actor, while still maintaining distinct layers for the action evaluation.

#### Replay Buffer

The replay buffer in the DDPG algorithm is the same as the one in DQN, containing for each experience the action before and after the action, the action, the done (or not done in my implementation) boolean and the reward.

## The Learning Algorithm

The pseudo-code below broadly outlines the functioning of the learning process of the DDPG algorithm.

```
Begin
    Initialise Actor and Critic Networks and buffer

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


As in the DQN algorithm, target networks are used in DDPG. These target networks are copies of the main actor and critic networks, but their weights are updated more slowly. They are used to calculate the loss functions and stabilize training by providing more consistent targets. During training, a random batch $\mathcal{B}$ of $n$ experiences is sampled, and the algorithm learns from this batch using experience replay. 

The actor's loss function is defined as:

$$
L_{\pi}(\mathcal{B}) = -\dfrac{1}{n} \sum_{(s, a, r, d, s') \in \mathcal{B}} V(s, \pi(s))
$$

The goal of gradient descent is to adjust the network's parameters to minimize a loss function. However, in this case, minimizing the actor's loss involves maximizing the critic’s output, which evaluates how good the selected actions are. Thus, the actor network is updated to produce actions that the critic considers more valuable. This aligns with the critic’s purpose: to evaluate the quality of state-action pairs. By optimizing the actor to maximize the critic's value estimates, we encourage the agent to take actions that the critic deems beneficial.

The critic’s loss function is defined as:

$$
L_{V}(\mathcal{B}) = \dfrac{1}{n}\sum_{(s, a, r, d, s') \in \mathcal{B}} \left(V(s, a) - T(s, a, r, d, s')\right)^2
$$

where the target $T(s, a, r, d, s')$ is given by:

$$
T(s, a, r, d, s') = r + \gamma (1 - d) V_{target}(s', \pi_{target}(s'))
$$

In this equation, the target value is computed using the target networks. The critic network learns to minimize the difference between its predicted value $V(s, a)$ and the target value $T(s, a, r, d, s')$, where $V_{target}$ and $\pi_{target}$ are the outputs of the target critic and actor networks, respectively. The term $(1 - d)$ ensures that no value is propagated if the episode has ended, as $d$ represents the done signal.
