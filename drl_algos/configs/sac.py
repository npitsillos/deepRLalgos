import torch
import drl_algos

from drl_algos.networks import critics
from drl_algos.networks import policies

default = {
    # Algorithm specific parameters
    "algorithm_kwargs" : {
        "discount": .99,
        "reward_scale": 1.0,
        "policy_lr": 1e-3,
        "qf_lr": 1e-3,
        "optimizer_class": torch.optim.Adam,
        "soft_target_tau": 1e-2,
        "target_update_period": 1,
        "use_automatic_entropy_tuning": True,
        "target_entropy": None,
    },

    # Network parameters
    "critic": drl_algos.networks.critics.MlpCritic,
    "critic_kwargs": {
        "hidden_sizes": [256,256],
        "base_kwargs": {
            "layer_activation": torch.nn.functional.relu,
            "layer_init": "orthogonal",
        },
    },
    "policy": drl_algos.networks.policies.MlpGaussianPolicy,
    "policy_kwargs": {
        "hidden_sizes": [256,256],
        "base_kwargs": {
            "layer_activation": torch.nn.functional.relu,
            "layer_init": "orthogonal",
        },
    },

    # Replay buffer
    "replay_buffer": drl_algos.data.replay_buffer.ReplayBuffer,
}

pendulum_v0 = {
    "env": "pendulum-v0",
    "device": "cuda:0",

    "critic_kwargs": {
        "hidden_sizes": [64,64],
    },
    "policy_kwargs": {
        "hidden_sizes": [64,64],
    },

    "replay_buffer_kwargs": {
        "size": 1e5,
    },

    # New implementation should look something like this
    "trainer_kwargs": {
        "timesteps": 1e5,
        "train_freq": 1, # Every 1 expl step do 1 train step
                         # to learn slower make this value larger than gradients
                         # updates
                         # this should be default
        "gradient_updates": 1, # Every 1 train step do 1 gradient update,
                               # to learn faster make this value larger than
                               # train freq
                               # this should be default, or -1
        "eval_freq": 5000, # Every 5000 train steps evaluate the agent. This
                           # should be logged against the expl step
        "log_freq": 1000, # Every 1000 train steps log the training stats. This
                          # should be logged against the expl step. Where
                          # relevant, this should be averaged over the 1000
                          # training steps. E.g., SAC or Dreamer stats, path
                          # collector stats, while replay buffer etc only needs
                          # the most recent value. I think pathcollector already
                          # handles this to an extent, since it keeps its own
                          # array which stores all the info which is later
                          # processed. I think SAC etc could have something
                          # similar. The logger just needs an array of values
                          # and a name for logging, e.g. ("Q1 Loss",
                          # [1.5,2.3...]). Every step just append to the array
                          # and empty when logging
    },
}
