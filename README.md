# DeepRLAlgos
This repository contains deep reinforcement learning algorithm implementations.  It is an implementation that does not utilise mutliple processes but instead focusses on simplicity and transparency.  It is easy to follow the algorithm and understand it since it is not hidden away behind multiple abstractions.

The package expects an environment and a policy to be provided which offers more control to the user over the models.

# Supported Implementations
1. [Proximal Policy Optimisation](https://arxiv.org/pdf/1707.06347.pdf)
2. [Soft Actor Critic](https://arxiv.org/pdf/1801.01290.pdf)

# TODO
## Algorithm Implementations
A list of supported algorithms to become available in the future in order of implementation.
1. [DDPG](https://arxiv.org/pdf/1509.02971.pdf)

## Experiments
* Train several different agents using different hyperparams and plot results.

# Contributing
If you would like to contribute then feel free to either open an issue, a pull request or even contact me.

# Experiments
To check the implementations and get a feel of what works and where I'll perform at least 3 different to compare the following:

* shared vs not-shared layers for policy and value functions
* GAE vs advantage