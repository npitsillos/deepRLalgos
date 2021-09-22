import gym
import torch
import numpy as np

from drl_algos.algos import SAC
from drl_algos.networks import critics
from drl_algos.networks import policies
from drl_algos.data import ReplayBuffer, MdpPathCollector
from drl_algos.trainers import BatchRLAlgorithm as Trainer
from drl_algos import utils

"""Note that our implementation is equivalent to RLKit but that the training
does not proceed exactly the same. This is because in our implementation we
correctly calculate the fan_in for the first layer of the neural networks
whereas rlkit seems to actually calculate the fan_out. Another difference is
that in rlkit they initialise (seemingly by mistake) their biases for the
policy's std layer to the weight initialisation value instead of to 0. For
pendulum, these changes make a big difference with our implementation converging
significantly faster (actually didn't run rlkit to convergence but it showed no
sign of improving after training for twice as long). It's unclear to what extent
each factor influences the result or if this result is consistent across
different environments/training runs. When removing these differences, the
training runs are exactly the same."""

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

                soft_target_tau=TAU,
            )

# Create training routine
utils.setup_logger('name-of-experiment')
trainer = Trainer(
              algorithm=algorithm,
              exploration_env=env,
              evaluation_env=eval_env,
              exploration_path_collector=expl_path_collector,
              evaluation_path_collector=eval_path_collector,
              replay_buffer=replay_buffer,
              batch_size=BATCH_SIZE,
              max_path_length=MAX_PATH_LEN,
              num_epochs=500,
              num_eval_steps_per_epoch=MAX_PATH_LEN*1,
              num_train_loops_per_epoch=1,
              num_trains_per_train_loop=MAX_PATH_LEN,
              num_expl_steps_per_train_loop=MAX_PATH_LEN,
              min_num_steps_before_training=1000
          )
trainer.to("cuda:0")
trainer.train()
