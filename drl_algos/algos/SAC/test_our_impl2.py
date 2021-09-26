import gym
import torch
import numpy as np

from drl_algos.algos import SAC
from drl_algos.networks import critics
from drl_algos.networks import policies
from drl_algos.data import MdpPathCollector2 as MdpPathCollector
from drl_algos.data import ReplayBuffer
from drl_algos.data import Logger2 as Logger
from drl_algos.trainers import BatchRLAlgorithm2 as Trainer
from drl_algos import utils


torch.manual_seed(0)
np.random.seed(0)

# Device for the networks
DEVICE = "cuda:0"

# Environment info
ENV_NAME = "Pendulum-v0"

# Hyperparams
BUFFER_SIZE = 50000
MAX_PATH_LEN = 200
BATCH_SIZE = 512
TAU = 0.01
POLICY_LR = 3e-4
CRITIC_LR = 3e-4
ACTOR_HIDDEN = [64,64]
CRITIC_HIDDEN = [64,64]

# Create and seed envs
env = gym.make(ENV_NAME).env
eval_env = gym.make(ENV_NAME).env
env.seed(0)
eval_env.seed(1)

# Env dimensions
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

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
        env,
    )

# Create exploration and evaluation path collectors
expl_path_collector = MdpPathCollector(
                          env,
                          policy,
                          MAX_PATH_LEN
                      )
eval_path_collector = MdpPathCollector(
                          eval_env,
                          eval_policy,
                          MAX_PATH_LEN
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

                soft_target_tau=TAU,
            )

# Create training routine
logger = Logger("name-of-experiment-2")
trainer = Trainer(
              algorithm=algorithm,
              exploration_env=env,
              evaluation_env=eval_env,
              exploration_path_collector=expl_path_collector,
              evaluation_path_collector=eval_path_collector,
              replay_buffer=replay_buffer,
              logger=logger,
              batch_size=BATCH_SIZE,
              num_epochs=180,
              num_eval_eps_per_epoch=1,
              num_train_loops_per_epoch=2,
              num_trains_per_train_loop=100,
              num_expl_steps_per_train_loop=100,
              min_num_steps_before_training=1000,
              checkpoint_freq=5,
          )
trainer.to("cuda:0")
trainer.train()
