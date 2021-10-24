import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as td

import drl_algos
from drl_algos.networks.mydreamer import Model, ActorCritic
from drl_algos.networks import policies, critics, models
from drl_algos.distributions import Discrete
from drl_algos.data import MdpPathCollector2, EpisodicReplayBuffer, Logger2
from drl_algos import utils


"""
Notes:
    - This version trains on-policy only using dreamer to get the latent
    states.
    - Training with only the observation loss doesn't seem to make much of a
    difference to the model observation learning
        - It converges faster but it doesn't seem to be significantly more
        accurate
    - Some training runs have converged to somewhat stable solutions but they
    fall far short of the optimal solution while others have failed to make any
    progress
        -
"""


SEED = 10

torch.manual_seed(SEED)
np.random.seed(SEED)

# Create and seed envs
env = gym.make('CartPole-v0').env
env.seed(SEED)
eval_env = gym.make('CartPole-v0').env
eval_env.seed(SEED+1)

# Env dimensions
obs_dim = env.observation_space.low.size
action_dim = env.action_space.n

# Create networks
# model = Model(
#     stoch_size=11,
#     deter_size=128,
#     discrete_size=11,
#     action_size=action_dim,
#     rssm_hidden_size=128,
#     obs_size=obs_dim,
#     hidden_sizes=[64, 64, 64, 64],
# )
# model = Model(
#     stoch_size=16,
#     deter_size=256,
#     discrete_size=16,
#     action_size=action_dim,
#     rssm_hidden_size=256,
#     obs_size=obs_dim,
#     hidden_sizes=[128, 128, 128, 128],
# )
model = Model(
    stoch_size=23,
    deter_size=512,
    discrete_size=23,
    action_size=action_dim,
    rssm_hidden_size=512,
    obs_size=obs_dim,
    hidden_sizes=[256, 256, 256, 256],
    kl_scale=1.,
)
# model = Model(
#     stoch_size=32,
#     deter_size=1024,
#     discrete_size=32,
#     action_size=action_dim,
#     rssm_hidden_size=1024,
#     obs_size=obs_dim,
#     hidden_sizes=[400, 400, 400, 400],
# )
model.latent_size = 4
# actor_critic = ActorCritic(
#     model, [256, 256], horizon=10, policy_lr=0.0001, critic_lr=0.001
# )
actor_critic = ActorCritic(
    model, [256, 256], horizon=2, policy_lr=0.00005, critic_lr=0.0005
)
# actor_critic = ActorCritic(
#     model, [256, 256], horizon=10, policy_lr=0.00001, critic_lr=0.0001
# )

policy = actor_critic.policy
eval_policy = policies.MakeDeterministic2(policy)

path_collector = MdpPathCollector2(
    env=env,
    policy=policy,
    max_episode_length=200,
)
eval_path_collector = MdpPathCollector2(
    env=eval_env,
    policy=eval_policy,
    max_episode_length=200,
)

replay_buffer = EpisodicReplayBuffer(
    max_replay_buffer_size=1000000,
    env=env,
    max_path_len=200,
    replace=False,
)

logger = Logger2("dreamer_medModel_medActorCriticNormLR_imagineDecoded5Step")

# Move to device
DEVICE = "cuda:1"

actor_critic.to(DEVICE)
model.to(DEVICE)
policy.to(DEVICE)
eval_policy.to(DEVICE)

# Prefill the experience replay
paths = path_collector.collect_new_paths(10000)
replay_buffer.add_paths(paths)
path_collector.end_epoch(-1)

for epoch in range(1, 2001):
    value_losses = []
    policy_losses = []

    paths = path_collector.collect_new_paths(2000)
    replay_buffer.add_paths(paths)

    posteriors, discounts = model.train(replay_buffer.random_batch(250, 10))
    states, actions, rewards, discounts, weights = model.imagine(
        actor_critic.policy, posteriors, discounts, 5, actor_critic.discount
    )
    obs_loc = model.obs_decoder(states)
    obs_dist = td.Independent(td.Normal(obs_loc, 1), 1)
    obs = obs_dist.mean

    # Compute losses
    actor_critic._num_train_steps += 1
    losses = actor_critic.compute_loss(obs, actions, rewards, discounts, weights)

    # Backpropogate policy loss
    actor_critic.policy_optim.zero_grad()
    losses.policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.policy.parameters(), actor_critic.grad_clip)
    actor_critic.policy_optim.step()

    # Backpropogate critic loss
    actor_critic.critic_optim.zero_grad()
    losses.critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.critic.parameters(), actor_critic.grad_clip)
    actor_critic.critic_optim.step()

    # Update target critic
    actor_critic.try_update_target_networks()

    # perform eval
    _ = eval_path_collector.collect_new_episodes(10)

    stats = {}
    stats.update(
        utils.add_prefix(
            replay_buffer.get_diagnostics(),
            prefix="replay_buffer/"
        )
    )
    stats.update(
        utils.add_prefix(
            model.get_diagnostics(),
            prefix="model/"
        )
    )
    stats.update(
        utils.add_prefix(
            actor_critic.get_diagnostics(),
            prefix="actor_critic/"
        )
    )
    stats.update(
        utils.add_prefix(
            path_collector.get_diagnostics(),
            prefix="exploration/"
        )
    )
    stats.update(
        utils.add_prefix(
            eval_path_collector.get_diagnostics(),
            prefix="evaluation/"
        )
    )
    stats["Epoch"] = epoch
    logger.log(epoch*2000, stats)

    path_collector.end_epoch(epoch)
    eval_path_collector.end_epoch(epoch)
    replay_buffer.end_epoch(epoch)
    model.end_epoch(epoch)
    actor_critic.end_epoch(epoch)
