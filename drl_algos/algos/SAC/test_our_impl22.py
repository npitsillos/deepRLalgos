import gym
import torch
import numpy as np

from drl_algos.algos import SAC2 as SAC
from drl_algos.networks import critics
from drl_algos.networks import policies
from drl_algos.data import MdpPathCollector2 as MdpPathCollector
from drl_algos.data import ReplayBuffer
from drl_algos.data import Logger2 as Logger
from drl_algos.trainers import BatchRLAlgorithm2 as Trainer
from drl_algos import utils

"""
Compared to original RLKit SAC, the diagnostic data looks similar, its just more
smooth and when it comes to min/maxs, they might be offset a bit since its
doing it over more data.

Compared to the original RLKit pathcollector, the data is alot better. Reporting
over episodes instead of paths is alot more interpretable. It actually doesn't
look like the original converges for some reason.

Adding the with torch.no_grad() for the q_target computation has no effect on
learning.
Adding torch.no_grad() around the q_values for the policy ruins learning.
Setting require_grad_ like in REDQ makes no difference to learning.
"""

torch.manual_seed(0)
np.random.seed(0)

# Device for the networks
DEVICE = "cuda:1"

# Environment info
ENV_NAME = "Pendulum-v0"
SEED = 0
EVAL_SEED = 100

# Hyperparams
BUFFER_SIZE = 50000
MAX_PATH_LEN = 200
BATCH_SIZE = 256
TAU = 0.005
POLICY_LR = 3e-4
CRITIC_LR = 3e-4
ACTOR_HIDDEN = [64,64]
CRITIC_HIDDEN = [64,64]

# Create and seed envs
env = gym.make(ENV_NAME).env
eval_env = gym.make(ENV_NAME).env
env.seed(SEED)
eval_env.seed(EVAL_SEED)

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
logger = Logger("pendulum/sac2/nograd3/trainer2/")
trainer = Trainer(
              algorithm=algorithm,
              exploration_env=env,
              evaluation_env=eval_env,
              exploration_path_collector=expl_path_collector,
              evaluation_path_collector=eval_path_collector,
              replay_buffer=replay_buffer,
              logger=logger,
              batch_size=BATCH_SIZE,
              num_epochs=200,
              num_eval_eps_per_epoch=10,
              num_train_loops_per_epoch=1000,
              num_trains_per_train_loop=1,
              num_expl_steps_per_train_loop=1,
              min_num_steps_before_training=1000,
              checkpoint_freq=5,
          )
trainer.to("cuda:1")
trainer.train()
