import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn, optim
import torch.nn.functional as F

import drl_algos
from drl_algos.networks.mydreamer import Model, ActorCritic
from drl_algos.networks import policies, critics, models
from drl_algos.distributions import Discrete
from drl_algos.data import ModelPathCollector2, EpisodicReplayBuffer, Logger2
from drl_algos import utils


"""
Notes:
    - There is definite signs of learning but it is unstable
        - I suspect that the sequence length is part of the problem
            - Using shorter sequences seems to hurt performance, using longer
            sequences seems to help but losing the short sequences seems to
            prevent it from learning what it did wrong on the short sequences
        - After correcting the imagined returns, it seems like the environment
        returns actually do correspond to the environment returns OUT OF DATE - I WAS NOT RESETTING STATISTICS AFTER EPOCH
            - For some reason, after good initial learning in imagination which
            also translates into the environment it's imagined returns then
            start to decrease which also translates to the environment
            - In one case, the imagined returns started to improve again which
            corresponded to an improvement in the environment but then
            performance in the environment decreased but imagined returns kept
            increasing
        - Maybe dreamer has a hard time with terminals but it doesn't really
        seem like that since Atari has alot of terminal states too
        - Maybe dreamer has a hard time because its not got high dimensional
        input so it's stochastic bottleneck isn't really working as intended
            - Dreamer did show that the image gradients are very important for
            learning its representation. Disabling reward gradients had no
            effect really but disabling image gradients had a massive effect
        - I tried replacing all the predicted rewards with 1s but it didn't
        seem to have much of an effect so I don't think reward prediction
        accuracy is the issue
        - Maybe it is due to the observation loss since with images I think the
        loss is alot larger. It seems to me that for each element in the
        observation space it gets it's own log_prob so with an image this loss
        gets really large whereas with low state it isn't so big. Maybe the
        loss just needs to be scaled up?
    - It seems that the first kl_loss term in a sequence can be quite big as
    removing that significantly shifts the kl_loss curve and seems to show more
    learning on the kl curve
        - Training on the first kl_loss seems silly to me because there is no
        way for the RSSM to learn what the next state would be since its input
        is a zerod state and action
        - It seems to help the observation loss curve and seems to have reduced
        the bumps in the reward and discount curves
        - Other than that its hard to say if it really helped learning
    - Zeroing out the deterministic state on the first observe doesn't seem
    to have helped performance
        - Not sure if it has helped learning
    - Evaluation performance is sometimes lower than exploration
        - This isn't due to chance
        - It seems like in general the noisy policy performs better but this
        isn't always the case and sometimes the evaluation significantly
        outperforms the exploration
        - With the deterministic policy, it does perform better with the
        determinisitic model so I believe the sampling is correct
        - I think the policy just isn't learning that well so the randomness
        helps it achieve better performance, but when it is learning the
        deterministic is clearly better

    - Compared to reinforce_lambda_truncated2.py the actor and critic loss
    curves are very different OUT OF DATE - I WAS NOT RESETTING STATISTICS AFTER EPOCH
        - the actor and critic losses start low and increase rather than start
        high and decrease
            - potentially this is due to the reward head etc not being trained
            yet but even with pretraining their losses are not that high to
            begin with and very quickly go down

        - there is no noticable jump in the actor and critic losses when the
        critic is hard updated every 100 steps

ToDo
    - Try training on an image based task like Atari to compare this
    implementation to the original
    - Try training a really small model
    - Train actor critic on on-policy trajectories
        - Use dreamer only to extract features
        - Then try switching in different heads to see their impact on
        performance
            - if discount is the issue, maybe it would be better to apply a
            hard clip on the terminals, e.g., if less than 50% sure the state
            is terminal then make it 0 so we don't learn from imagined states
            which we have low certainty about
                - probably fine to keep above 50% as their values but may be
                worth testing
        - If all of above works then the issue is with imagination
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
model = Model(
    stoch_size=4,
    deter_size=16,
    discrete_size=4,
    action_size=action_dim,
    rssm_hidden_size=16,
    obs_size=obs_dim,
    hidden_sizes=[64],
)
actor_critic = ActorCritic(model, [64], horizon=10)

policy = actor_critic.policy
eval_policy = policies.MakeDeterministic2(policy)

path_collector = ModelPathCollector2(
    env=env,
    policy=policy,
    model=model,
    max_episode_length=200,
)
eval_path_collector = ModelPathCollector2(
    env=eval_env,
    policy=eval_policy,
    model=model,
    max_episode_length=200,
    deterministic=True,
)

replay_buffer = EpisodicReplayBuffer(
    max_replay_buffer_size=1000000,
    env=env,
    max_path_len=200,
    replace=False,
)

logger = Logger2("dreamer_seq10_horizon10_ignoreFirstKl_zeroFirstDeter_vSmall")

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

# Pretrain model
# for i in range(1000):
#     _, _ = model.train(replay_buffer.random_batch(250, 10))

for epoch in range(2000):
    value_losses = []
    policy_losses = []

    # Only reports every 250 gradients steps, i.e. 1000 environment steps
    for i in range(1):
        # Get paths and process
        # paths = path_collector.collect_new_paths(4)
        paths = path_collector.collect_new_episodes(1)
        replay_buffer.add_paths(paths)

        # Train dreamer
        paths = replay_buffer.random_batch(250, 10)
        post, discounts = model.train(paths)
        actor_critic.train(post, discounts)

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
    logger.log(epoch*1000, stats)

    path_collector.end_epoch(epoch)
    eval_path_collector.end_epoch(epoch)
    replay_buffer.end_epoch(epoch)
    model.end_epoch(epoch)
    actor_critic.end_epoch(epoch)
