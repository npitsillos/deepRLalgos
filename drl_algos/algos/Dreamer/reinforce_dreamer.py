import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn, optim
import torch.nn.functional as F

from drl_algos.networks import policies, critics, models
from drl_algos.distributions import Discrete
from drl_algos.data import ModelPathCollector2, EpisodicReplayBuffer, Logger2
from drl_algos import utils

"""
    below implementation seems legit
    https://github.com/RajGhugare19/dreamerv2/blob/0c12a20c5608fdadc5aeb0cd335d6db3ba0cad0c/dreamerv2/training/config.py

    - note - I think you can add exploration noise to Dreamer because all
    on-policy training is done within imagination, so its okay to add exploration
    into the actual environment interactions

    - below seems to process imagination trajectories correctly, so any learning
    issues is probably due to the model
        - I think this implementation is correct and I've changed
        reinforce_lambda_truncated so it does it in parallel like this one and
        that seems to be equivalent to the old slow version
        - Reflecting on the Dreamer code I based my model implementation on,
        I don't really trust it
            - doesn't seem to have been tested or verified much
            - the training results don't really make much sense
        - Note that the best loss the reward and observation model can converge
        to is 0.9189 which would mean 100% accuracy
            - -log_prob(normal_dist.mean) = 0.9189 when using a scale of 1.0
            - the reward loss was converging to .96 when including the loss
            term for the first state but having now removed that it converges
            to .919 which isn't perfect buts its okay, +-.15, but why can't it
            learn to just predict 1 all the time?
            - the observation loss converges around about .92
        - the discount does seem to be learning well and from inspection
        it doesn't only predict 1, and actually when it predicts lower its
        often very close to 0. I've not confirmed if its actually correct
        in imagination but it seems promising
        - my original implementation of the kl balancing loss was incorrect I
        think

        - the imagined rewards don't seem accurate

    - it seems like the implementation I have copied has missed some stuff
        - for example, it seems that the state and action should be fed through
        a fully connected layer before being passed to the deterministic layer
            - I beleive this should output same as deterministic input

    - original implementations seems to discount next state by current state
    gamma but that doesn't seem right to me, this implementation has followed
    that anyway
    - original implementation used target critic for baseline but that doesn't
    seem right, this implementation has followed that anyway
    - original seemed to scale discount loss by 5, not done in this impl
    - original implementation did a hard critic update every 100 steps, not done
    in this impl
    - I should scale the loss by the cumulative gamma not just gamma but I've
    not done that yet
        - would it not make sense to actually scale the entire target by
        gamma? Otherwise, aren't Values calculated for terminal states being
        incorporated into earlier states
        - V_i = Y_i * (R_i+1 + Y_i+1 * v(S_i+1)) # this is my new function

    - final state is only ever used for bootstrap no loss calculation
        - .i.e., V_H-1 = r_H-1 + y_H-1 * v(s_H)
        - so reward, action and gamma are unneccessary but its easier to
        include them then not use them
    - first state is from experience replay, i.e., posterior
        - apparently shouldn't use for training policy and also lose two
        model_states at end of sequence but I'm not sure why
        - not sure I really understand why
            - I get last state is lost for bootstrap but action executed
            in posterior state is on-policy
            - and why lose second last model_state? I think thats maybe a
            miscommunication because it says its action leads nowhere but
            it leads to the bootstrap state and it looks like in its graph
            that it does calculate a loss for the second last state
                - but in code it does look like it chucks last two states
                but that would mean it does use first state (I think)
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

def lambda_returns(rewards, gammas, values, lam=0.95, discount=0.99):
    returns = []
    prev_target = None
    gammas = gammas*discount
    # Iterate backwards through list starting from last index
    for i in range(len(rewards[0])-2, -1, -1):
        reward = rewards[:, i]
        # NOTE - in the paper they say they discount the next state by the
        #        current state's gamma but I'm not sure this is correct but have
        #        left it as is for now
        gamma = gammas[:, i]
        next_value = values[:, i+1]

        if prev_target is None:
            target = reward + gamma * next_value
        else:
            target = reward + gamma * ((1.0-lam)*next_value + lam*prev_target)

        returns.append(target.detach())
        prev_target = target

    return torch.stack(returns, dim=1).flip(1)

# Create networks
model = models.MlpDreamer2(obs_dim, action_dim)
policy = policies.MlpDiscretePolicy(
    model.latent_size,
    action_dim,
    [400, 400, 400, 400],
    layer_activation=F.elu
)
eval_policy = policies.MakeDeterministic(policy)
critic = critics.SacMlpCritic(
    model.latent_size,
    1,
    [400, 400, 400, 400],
    layer_activation=F.elu
)
critic_target = critics.SacMlpCritic(
    model.latent_size,
    1,
    [400, 400, 400, 400],
    layer_activation=F.elu
)

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

logger = Logger2("dreamer-reinforce")

# Move to device
DEVICE = "cuda:1"

policy.to(DEVICE)
eval_policy.to(DEVICE)
critic.to(DEVICE)
critic_target.to(DEVICE)
model.to(DEVICE)

total_rewards = []
total_eval_rewards = []
total_value_losses = []
total_policy_losses = []
vf_criterion = nn.MSELoss(reduction="none")
policy_optim = optim.Adam(
    policy.parameters(),
    lr=0.00004,
    eps=1e-5,
    weight_decay=1e-6
)
critic_optim = optim.Adam(
    critic.parameters(),
    lr=0.0001,
    eps=1e-5,
    weight_decay=1e-6
)

# Prefill the experience replay
paths = path_collector.collect_new_paths(10000)
replay_buffer.add_paths(paths)
path_collector.end_epoch(-1)

# Below runs for 1M timesteps, with a total of 250k gradient updates
# the actor critic is trained on 8250M imagined steps
#   - each step samples 200*10 experiences giving 200*11=2200 model states
#   - each model step has 15 imagination steps giving 2200*15=33000 imagined steps
#   - there is 250000 updates giving 250000*33000=8250M total imagined steps
# for epoch in range(2500):
for epoch in range(200):
    value_losses = []
    policy_losses = []

    # Only reports every 250 gradients steps, i.e. 1000 environment steps
    for i in range(250):
    # for i in range(50):
        # Get paths and process
        paths = path_collector.collect_new_paths(4)
        replay_buffer.add_paths(paths)

        # Train dreamer
        paths = replay_buffer.random_batch(200, 10)
        model_states = model.train(paths)

        # Get imagination trajectories
        states, actions, log_pis, rewards, gammas = model.dream(
            model_states,
            policy,
            15,
        )

        if epoch > 5:
            print()
            print(rewards.mean(), rewards.max(), rewards.min())

        values = critic_target(states)
        returns = lambda_returns(rewards, gammas, values)

        reward_tensor = returns.flatten(0, 1)
        state_tensor = states[:, :-1].flatten(0, 1)
        action_tensor = actions[:, :-1].flatten(0, 1)
        action_tensor = torch.argmax(action_tensor, dim=1, keepdim=True)
        gamma_tensor = torch.cumprod(gammas[:, :-1], 1).flatten(0, 1)

        # Calculate critic loss
        value = critic(state_tensor)
        critic_loss = (vf_criterion(value, reward_tensor) * gamma_tensor).mean()
        value_losses.append(critic_loss.item())

        # Calculate policy loss
        baseline = (reward_tensor - critic_target(state_tensor)).detach()
        dist = policy(state_tensor)
        # Below line looks like a potential area where my implementation may be
        # incorrect, not flattening generates unexpected results
        reinforce = -dist.log_prob(action_tensor.flatten()) * baseline.flatten()
        entropy = 1e-3 * dist.entropy()
        policy_loss = ((reinforce - entropy) * gamma_tensor).mean()
        policy_losses.append(policy_loss.item())

        # Backpropagate policy loss
        policy_optim.zero_grad()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 100)
        policy_loss.backward()
        policy_optim.step()

        # Backpropagate critic loss
        critic_optim.zero_grad()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 100)
        critic_loss.backward()
        critic_optim.step()

        # Soft update the target
        utils.soft_update(critic, critic_target, 5e-3)

        batch_rewards = []
        batch_states = []
        batch_actions = []

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
            {"policy_loss": np.mean(policy_losses),
            "critic_loss": np.mean(value_losses)},
            prefix="algorithm/"
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
