import gym
import torch
import numpy as np

from drl_algos.algos import SAC2 as SAC
from drl_algos.networks import critics
from drl_algos.networks import policies
from drl_algos.data.path_collector_parallel import ParallelPathCollector
from drl_algos.data import ReplayBuffer
from drl_algos.data import Logger2 as Logger
from drl_algos.trainers.trainer3 import BatchRLAlgorithm2 as Trainer
from drl_algos import utils

"""
Seems to converge as fast as impl22 and is as stable.

Not sure how much faster this is for gathering data than a single worker on
computationally inexpensive environments but its definetly a speed up on more
intensive environments.
"""

# Device for the networks
DEVICE = "cuda:1"

# Environment info
ENV_NAME = "Pendulum-v0"
SEED = 0
EVAL_SEED = 100
N_WORKERS = 5
N_EVAL_WORKERS = 10

# Hyperparams
BUFFER_SIZE = 50000
MAX_PATH_LEN = 200
BATCH_SIZE = 256
TAU = 0.005
POLICY_LR = 3e-4
CRITIC_LR = 3e-4
ACTOR_HIDDEN = [64,64]
CRITIC_HIDDEN = [64,64]

def create_env():
    return gym.make(ENV_NAME).env

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # Env dimensions
    dummy_env = create_env()
    obs_dim = dummy_env.observation_space.low.size
    action_dim = dummy_env.action_space.low.size

    # Create critics
    qf1 = critics.MlpCritic(
                hidden_sizes=CRITIC_HIDDEN,
                input_size=obs_dim+action_dim,
                output_size=1,
          )
    qf2 = critics.MlpCritic(
                hidden_sizes=CRITIC_HIDDEN,
                input_size=obs_dim+action_dim,
                output_size=1,
          )
    target_qf1 = critics.MlpCritic(
                     hidden_sizes=CRITIC_HIDDEN,
                     input_size=obs_dim+action_dim,
                     output_size=1,
                 )
    target_qf2 = critics.MlpCritic(
                     hidden_sizes=CRITIC_HIDDEN,
                     input_size=obs_dim+action_dim,
                     output_size=1,
                 )

    # Create actors
    policy = policies.MlpGaussianPolicy(
                 hidden_sizes=ACTOR_HIDDEN,
                 input_size=obs_dim,
                 output_size=action_dim,
             )
    eval_policy = policies.MakeDeterministic(policy)

    # Create buffer
    replay_buffer = ReplayBuffer(
            BUFFER_SIZE,
            dummy_env,
        )

    # Create exploration and evaluation path collectors
    expl_path_collector = ParallelPathCollector(
                              create_env,
                              policy,
                              MAX_PATH_LEN,
                              N_WORKERS,
                              SEED
                          )
    eval_path_collector = ParallelPathCollector(
                              create_env,
                              eval_policy,
                              MAX_PATH_LEN,
                              N_EVAL_WORKERS,
                              EVAL_SEED
                          )

    # Create algorithm
    algorithm = SAC(
                    env=dummy_env,
                    policy=policy,
                    qf1=qf1,
                    qf2=qf2,
                    target_qf1=target_qf1,
                    target_qf2=target_qf2,

                    policy_lr=POLICY_LR,
                    qf_lr=CRITIC_LR,

                    soft_target_tau=TAU,
                )

    dummy_env.close()

    # Create training routine
    logger = Logger("pendulum/sac2/trainer2/parallel/")
    trainer = Trainer(
                  algorithm=algorithm,
                  exploration_path_collector=expl_path_collector,
                  evaluation_path_collector=eval_path_collector,
                  replay_buffer=replay_buffer,
                  logger=logger,
                  batch_size=BATCH_SIZE,
                  num_epochs=200,
                  num_eval_eps_per_epoch=10,
                  num_train_loops_per_epoch=200,
                  num_trains_per_train_loop=5,
                  num_expl_steps_per_train_loop=1,
                  min_num_steps_before_training=200,
                  checkpoint_freq=5,
              )
    trainer.to("cuda:1")
    trainer.train()
