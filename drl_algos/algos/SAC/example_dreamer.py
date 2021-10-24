import gym
import torch
import numpy as np

from drl_algos.networks import Dreamer
from drl_algos.data import EpisodicReplayBuffer as ReplayBuffer
from drl_algos.data import MdpPathCollector
from drl_algos import utils

torch.manual_seed(0)
np.random.seed(0)

# Device for the networks
DEVICE = "cuda:1"

# Environment info
ENV_NAME = "Pendulum-v0"
# ENV_NAME = "BipedalWalkerHardcore-v3"

# Hyperparams
BUFFER_SIZE = 100000
MAX_PATH_LEN = 200
BATCH_SIZE = 50
SEQUENCE_LEN = 50
GAMMA = .995
IMAGINATION_LEN = 15

# Create and seed envs
env = gym.make(ENV_NAME).env
eval_env = gym.make(ENV_NAME).env
env.seed(0)
eval_env.seed(1)

# Env dimensions
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

# Create dreamer
dreamer = Dreamer(GAMMA, obs_dim, action_dim, DEVICE)

# Create replay buffer
replay_buffer = ReplayBuffer(
                    BUFFER_SIZE,
                    env,
                    MAX_PATH_LEN
                )

# Create exploration and evaluation path collectors
expl_path_collector = MdpPathCollector(
                          env,
                          policy,
                      )
eval_path_collector = MdpPathCollector(
                          eval_env,
                          eval_policy,
                      )

# Create algorithm
algorithm = SAC(
                env=env,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                target_qf1=target_qf1,
                target_qf2=target_qf2,

                policy_lr=POLICY_LR,
                qf_lr=CRITIC_LR,

                soft_target_tau=TAU
            )

paths = expl_path_collector.collect_new_paths(200,3, False)
replay_buffer.add_paths(paths)
replay_buffer.random_batch(3, 3, True)


# Create training routine
trainer = BatchRLAlgorithm(
              algorithm=algorithm,
              exploration_env=env,
              evaluation_env=eval_env,
              exploration_path_collector=expl_path_collector,
              evaluation_path_collector=eval_path_collector,
              replay_buffer=replay_buffer,
              batch_size=BATCH_SIZE,
              max_path_length=MAX_PATH_LEN,
              num_epochs=100,
              num_eval_steps_per_epoch=MAX_PATH_LEN*10,
              num_train_loops_per_epoch=1,
              num_trains_per_train_loop=MAX_PATH_LEN,
              num_expl_steps_per_train_loop=MAX_PATH_LEN,
              min_num_steps_before_training=1000
          )

# Set up logging
utils.setup_logger('pendulum')
print()

# Move onto GPU and start training
trainer.to(DEVICE)
trainer.train()
