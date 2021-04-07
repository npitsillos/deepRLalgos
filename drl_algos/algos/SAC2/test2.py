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
np.random.seed(0)
ptu.set_gpu_mode(True)

env = gym.make("Pendulum-v0").env
eval_env = gym.make("Pendulum-v0").env
env.seed(0)
eval_env.seed(1)
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

qf1 = ConcatMlp(
          input_size=obs_dim + action_dim,
          output_size=1,
          hidden_sizes=[64, 64],
      ).cuda()
qf2 = ConcatMlp(
          input_size=obs_dim + action_dim,
          output_size=1,
          hidden_sizes=[64, 64],
      ).cuda()
target_qf1 = ConcatMlp(
                 input_size=obs_dim + action_dim,
                 output_size=1,
                 hidden_sizes=[64, 64],
             ).cuda()
target_qf2 = ConcatMlp(
                 input_size=obs_dim + action_dim,
                 output_size=1,
                 hidden_sizes=[64, 64],
             ).cuda()
policy = TanhGaussianPolicy(
             obs_dim=obs_dim,
             action_dim=action_dim,
             hidden_sizes=[64, 64],
         ).cuda()
eval_policy = MakeDeterministic(policy)
replay_buffer = EnvReplayBuffer(
        100000,
        env,
    )
expl_path_collector = MdpPathCollector(
                          env,
                          policy,
                      )
eval_path_collector = MdpPathCollector(
                          eval_env,
                          eval_policy,
                      )

trainer = SACTrainer(
              env=env,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              target_qf1=target_qf1,
              target_qf2=target_qf2
          )

setup_logger('name-of-experiment')
algorithm = TorchBatchRLAlgorithm(
                trainer=trainer,
                exploration_env=env,
                evaluation_env=eval_env,
                exploration_data_collector=expl_path_collector,
                evaluation_data_collector=eval_path_collector,
                replay_buffer=replay_buffer,
                batch_size=64,
                max_path_length=200,
                num_epochs=10,
                num_eval_steps_per_epoch=5000,
                num_trains_per_train_loop=1000,
                num_expl_steps_per_train_loop=1000,
                min_num_steps_before_training=100
            )
algorithm.to("cuda:0")
algorithm.train()
