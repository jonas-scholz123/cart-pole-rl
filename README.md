# cart-pole-rl
Using Q-learning on a discretised version of the cart pole problem as a basic reinforcement learning implementation. In the second rendition, I use a NN to predict Q-values on the continuous state-space (deep reinforcement learning).

![The result](cartpole.gif)

## Usage:

Install requirements.txt:

`cd` into root directory and run:

```pip install -r requirements.txt```

once all requirements are installed run

 `python3 deep_q_learning.py` for the deep q-learning version

 `python3 discrete_q_learning.py` for the discretised q-learning version
