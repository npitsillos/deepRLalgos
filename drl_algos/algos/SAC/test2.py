import gym
import torch
import numpy as np

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

torch.manual_seed(0)
ptu.set_gpu_mode(True)
np.random.seed(0)

# Environment info
ENV_NAME = "Pendulum-v0"

# Hyperparams
BUFFER_SIZE = 50000
MAX_PATH_LEN = 200
BATCH_SIZE = 512
TAU = 0.01
POLICY_LR = 3e-4
CRITIC_LR = 3e-4

# Create and seed envs
env = gym.make(ENV_NAME).env
eval_env = gym.make(ENV_NAME).env
env.seed(0)
eval_env.seed(1)

# Env dimensions
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

# Create critics
qf1 = ConcatMlp(
          input_size=obs_dim + action_dim,
          output_size=1,
          hidden_sizes=[64, 64],
      )
qf2 = ConcatMlp(
          input_size=obs_dim + action_dim,
          output_size=1,
          hidden_sizes=[64, 64],
      )
target_qf1 = ConcatMlp(
                 input_size=obs_dim + action_dim,
                 output_size=1,
                 hidden_sizes=[64, 64],
             )
target_qf2 = ConcatMlp(
                 input_size=obs_dim + action_dim,
                 output_size=1,
                 hidden_sizes=[64, 64],
             )

# Create actors
policy = TanhGaussianPolicy(
             obs_dim=obs_dim,
             action_dim=action_dim,
             hidden_sizes=[64, 64],
         )
eval_policy = MakeDeterministic(policy)

# Create buffer
replay_buffer = EnvReplayBuffer(
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
trainer = SACTrainer(
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

# Create training routine
setup_logger('name-of-experiment')
algorithm = TorchBatchRLAlgorithm(
                trainer=trainer,
                exploration_env=env,
                evaluation_env=eval_env,
                exploration_data_collector=expl_path_collector,
                evaluation_data_collector=eval_path_collector,
                replay_buffer=replay_buffer,
                batch_size=BATCH_SIZE,
                max_path_length=MAX_PATH_LEN,
                num_epochs=4,
                num_eval_steps_per_epoch=MAX_PATH_LEN*1,
                num_train_loops_per_epoch=1,
                num_trains_per_train_loop=MAX_PATH_LEN,
                num_expl_steps_per_train_loop=MAX_PATH_LEN,
                min_num_steps_before_training=1000
            )
algorithm.to("cuda:0")
print(trainer.log_alpha)
algorithm.train()
