import gym
import torch
import numpy as np

from drl_algos.algos import SAC
from drl_algos.networks import critics, policies, MlpDreamer
from drl_algos.data import EpisodicReplayBuffer as ReplayBuffer
from drl_algos.data import MdpPathCollector
from drl_algos.trainers import BatchRLAlgorithm
from drl_algos import utils

torch.manual_seed(0)
np.random.seed(0)

# Device for the networks
DEVICE = "cuda:1"

# Environment info
# ENV_NAME = "Pendulum-v0"
ENV_NAME = "BipedalWalkerHardcore-v3"

# Hyperparams
BUFFER_SIZE = 1000
MAX_PATH_LEN = 200
BATCH_SIZE = 256
TAU = 0.005
POLICY_LR = 1e-3
CRITIC_LR = 1e-3
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

obs = torch.rand(1, 24)
dist = policy(obs)
new_obs_actions, log_pi = dist.rsample_and_logprob()
log_pi = log_pi.unsqueeze(-1)
alpha_loss = 0
alpha = 1
q_new_actions = torch.min(
    qf1(obs, new_obs_actions),
    qf2(obs, new_obs_actions),
)
policy_loss = (alpha*log_pi - q_new_actions).mean()
print(policy_loss)
algorithm.policy_optimizer.zero_grad()
policy_loss.backward(retain_graph=True)
print(policy.base.fc0.weight.grad.mean())
print(policy.base.fc0.weight.grad.max())
print(policy.base.fc0.weight.grad.min())

policy_loss = (alpha*log_pi - q_new_actions).mean()
print(policy_loss)
algorithm.policy_optimizer.zero_grad()
policy_loss.backward(retain_graph=True)
print(policy.base.fc0.weight.grad.mean())
print(policy.base.fc0.weight.grad.max())
print(policy.base.fc0.weight.grad.min())

policy_loss = (alpha*log_pi - q_new_actions.detach()).mean()
print(policy_loss)
algorithm.policy_optimizer.zero_grad()
policy_loss.backward(retain_graph=True)
print(policy.base.fc0.weight.grad.mean())
print(policy.base.fc0.weight.grad.max())
print(policy.base.fc0.weight.grad.min())

policy_loss *= 1e3
algorithm.policy_optimizer.zero_grad()
policy_loss.backward(retain_graph=True)
print(policy.base.fc0.weight.grad.mean())
print(policy.base.fc0.weight.grad.max())
print(policy.base.fc0.weight.grad.min())

algorithm.policy_optimizer.zero_grad()
policy_loss *= 0
policy_loss.backward(retain_graph=True)
print(policy.base.fc0.weight.grad.mean())
print(policy.base.fc0.weight.grad.max())
print(policy.base.fc0.weight.grad.min())



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
