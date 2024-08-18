# The Dueling Double Deep Q-Network Algorithm

## Prerequisites

Before reading this explanation of the Dueling DDQN algorithm, you should have a good understanding of the functionning of both DDQN and Dueling DQN.

## The Dueling Double DQN improvement

If you understand DDQN and Dueling DQN, you probably already have the right idea about how Dueling DDQN works. The Dueling Double DQN consists in an improvement af the DQN algorithm in which we combine the target calculation from DDQN and the network architecture from Dueling DQN.

So Q is defined as follows : $$Q(s, a) = V(s) + A(s, a) - max_{a' \in \mathcal{A}}\left(A(s, a')\right)$$

and the target is : $$r + \gamma (1 - d)Q_{target}(s', Argmax_{a' \in \mathcal{A}}(Q(s', a')))$$ 