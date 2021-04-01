import gym
import torch
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp

torch.manual_seed(0)
ptu.set_gpu_mode(True)

env = gym.make("Pendulum-v0")
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

trainer = SACTrainer(
              env=env,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              target_qf1=target_qf1,
              target_qf2=target_qf2
          )

dist = policy(torch.tensor([[0,0,0]]).float().cuda())
print(dist.rsample_and_logprob())
print(policy.get_action(torch.tensor([0,0,0]).float().cuda()))
print(eval_policy.get_action(torch.tensor([0,0,0]).float().cuda()))
