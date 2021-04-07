import gym
import torch
import numpy as np

import sac2
import policies
import rollout
import replay_buffer
import algo

ENV_NAME = "Pendulum-v0"

BUFFER_SIZE = 50000
BATCH_SIZE = 512
TAU = 0.01
POLICY_LR = 3e-4
CRITIC_LR = 3e-4

torch.manual_seed(0)
np.random.seed(0)

env = gym.make(ENV_NAME).env
eval_env = gym.make(ENV_NAME).env
env.seed(0)
eval_env.seed(1)
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

qf1 = policies.ConcatMlp(
          input_size=obs_dim + action_dim,
          output_size=1,
          hidden_sizes=[64, 64],
      ).cuda()
qf2 = policies.ConcatMlp(
          input_size=obs_dim + action_dim,
          output_size=1,
          hidden_sizes=[64, 64],
      ).cuda()
target_qf1 = policies.ConcatMlp(
                 input_size=obs_dim + action_dim,
                 output_size=1,
                 hidden_sizes=[64, 64],
             ).cuda()
target_qf2 = policies.ConcatMlp(
                 input_size=obs_dim + action_dim,
                 output_size=1,
                 hidden_sizes=[64, 64],
             ).cuda()
policy = policies.TanhGaussianPolicy(
             obs_dim=obs_dim,
             action_dim=action_dim,
             hidden_sizes=[64, 64],
         ).cuda()
eval_policy = policies.MakeDeterministic(policy)

replay_buffer = replay_buffer.ReplayBuffer(
                    BUFFER_SIZE,
                    env,
                )
expl_path_collector = rollout.MdpPathCollector(
                          env,
                          policy,
                      )
eval_path_collector = rollout.MdpPathCollector(
                          eval_env,
                          eval_policy,
                          render=True,
                      )

trainer = sac2.SAC(
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

algo = algo.BatchRLAlgorithm(
                        trainer=trainer,
                        exploration_env=env,
                        evaluation_env=eval_env,
                        exploration_data_collector=expl_path_collector,
                        evaluation_data_collector=eval_path_collector,
                        replay_buffer=replay_buffer,
                        batch_size=BATCH_SIZE,
                        max_path_length=200,
                        num_epochs=50,
                        num_eval_steps_per_epoch=2000,
                        num_trains_per_train_loop=10000,
                        num_expl_steps_per_train_loop=10000,
                        min_num_steps_before_training=1000
        )
algo.train()
