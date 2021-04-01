import gym
import torch
import sac2
import policies

torch.manual_seed(0)

env = gym.make("Pendulum-v0")
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

trainer = sac2.SAC(
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
