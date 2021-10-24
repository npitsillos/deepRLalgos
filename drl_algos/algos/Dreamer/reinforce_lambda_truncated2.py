import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as td


from drl_algos.networks import policies, critics
from drl_algos.distributions import Discrete
from drl_algos.data import MdpPathCollector2, EpisodicReplayBuffer
from drl_algos import utils

"""Notes
    - this is practically identical to how dreamer is trained
    - to make this work had to append a dummy state onto the end along with
    its action, reward and discount but they are never utilised
    - it seems to work
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

def lambda_returns(rewards, discounts, values, lam=.95, discount=.99):
    targets = []
    prev_target = None
    discounts *= discount

    for i in range(len(values)-2, -1, -1):
        if prev_target == None:
            target = rewards[i] + discounts[i] * values[i+1]
        else:
            target = rewards[i] + discounts[i] * (
                (1.-lam) * values[i+1] + lam * prev_target
            )
        targets.append(target)
        prev_target = target
    targets = torch.stack(targets).flip(0)

    return targets

# Create networks
policy = policies.MlpDiscretePolicy2(
    obs_dim,
    action_dim,
    [16],
    layer_activation=F.elu
)
eval_policy = policies.MakeDeterministic2(policy)
critic = critics.SacMlpCritic(
    obs_dim,
    1,
    [256,256],
    layer_activation=F.elu
)
critic_target = critics.SacMlpCritic(
    obs_dim,
    1,
    [256,256],
    layer_activation=F.elu
)

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

device = torch.device("cpu")
policy.to(device)
eval_policy.to(device)
critic.to(device)
critic_target.to(device)

total_rewards = []
total_eval = []
value_losses = []
policy_losses = []
advantages = []
log_probs = []
targets_list = []
values_list = []
slow_values_list = []
vf_criterion = nn.MSELoss(reduction="none")
policy_optim = optim.Adam(policy.parameters(),
                       lr=0.001)
critic_optim = optim.Adam(critic.parameters(),
                       lr=0.01)
for epoch in range(2000):
    # Get paths and add to buffer
    paths = path_collector.collect_new_paths(2000)
    replay_buffer = EpisodicReplayBuffer(
        max_replay_buffer_size=200*10,
        env=env,
        max_path_len=200,
        replace=False,
    )
    replay_buffer.add_paths(paths)

    # Sample paths
    paths = replay_buffer.random_batch(200, 10)
    batch = utils.to_tensor_batch(paths, device)

    # Preprocess data into required sequences
    batch_size = batch["observations"].shape[1]
    states = torch.cat(
        (batch["observations"][0:1], batch["next_observations"], torch.zeros(1,200,4).to(device))
    )
    dummy_action = torch.zeros(
        1, batch_size, action_dim
    ).to(device)
    actions = torch.cat((dummy_action, batch["actions"], dummy_action))
    dummy_reward = torch.zeros(1, batch_size, 1).to(device)
    rewards = torch.cat((dummy_reward, batch["rewards"], torch.zeros(1,200,1).to(device)))
    dummy_terminal = torch.zeros(1, batch_size, 1).to(device)
    discounts = 1. - torch.cat((dummy_terminal, batch["terminals"], torch.zeros(1,200,1).to(device)))
    weights = torch.cumprod(
        torch.cat((torch.ones_like(discounts[0:1]), discounts[1:])), 0
    )

    # Calculate returns
    values = critic_target(states)
    targets = lambda_returns(rewards, discounts, values)

    # Calculate actor loss
    policy_dist = policy(states.detach()[:-2])
    baseline_loc = critic_target(states[:-2])
    baseline_dist = td.Normal(baseline_loc, 1)
    baseline = baseline_dist.mean
    advantage = (targets[1:] - baseline).detach()
    log_prob = policy_dist.log_prob(actions[1:-1])
    objective = log_prob * advantage.squeeze(2)
    entropy = policy_dist.entropy()
    objective += entropy * 2e-3
    actor_loss = -(weights[:-2].squeeze(2).detach() * objective).mean()

    # Calculate critic loss
    critic_loc = critic(states[:-1].detach())
    critic_dist = td.Normal(critic_loc, 1)
    critic_loss = -(
        critic_dist.log_prob(targets.detach())
        * weights[:-1].detach()
    ).mean()

    value_losses.append(critic_loss.item())
    policy_losses.append(actor_loss.item())
    advantages.append(advantage.mean().item())
    log_probs.append(log_prob.mean().item())
    targets_list.append(targets.mean().item())
    values_list.append(baseline.mean().item())
    slow_values_list.append(values.mean().item())

    # Backpropagate policy loss
    policy_optim.zero_grad()
    actor_loss.backward()
    policy_optim.step()

    # Backpropagate critic loss
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # Update the target
    if epoch % 100 == 0:
        utils.soft_update(critic, critic_target, 1)

    # Evaluate
    _ = eval_path_collector.collect_new_episodes(10)

    # Print running average
    total_rewards.append(path_collector.get_diagnostics()["Returns Mean"])
    total_eval.append(eval_path_collector.get_diagnostics()["Returns Mean"])
    path_collector.end_epoch(epoch)
    eval_path_collector.end_epoch(epoch)
    print("\rEpoch: {} Expl: {:.2f} Eval: {:.2f}".format(
         epoch, total_rewards[-1], total_eval[-1]), end="")

plt.plot(range(len(total_rewards)), targets_list)
plt.savefig("reinforce_lambda_truncated2_targets.png")
plt.clf()
plt.plot(range(len(total_rewards)), values_list)
plt.savefig("reinforce_lambda_truncated2_value.png")
plt.clf()
plt.plot(range(len(total_rewards)), slow_values_list)
plt.savefig("reinforce_lambda_truncated2_value_slow.png")
plt.clf()
plt.plot(range(len(total_rewards)), advantages)
plt.savefig("reinforce_lambda_truncated2_advantages.png")
plt.clf()
plt.plot(range(len(total_rewards)), log_probs)
plt.savefig("reinforce_lambda_truncated2_log_prob.png")
plt.clf()
plt.plot(range(len(total_rewards)), total_rewards)
plt.savefig("reinforce_lambda_truncated2_expl.png")
plt.clf()
plt.plot(range(len(total_eval)), total_eval)
plt.savefig("reinforce_lambda_truncated2_eval.png")
plt.clf()
plt.plot(range(len(policy_losses)), policy_losses)
plt.savefig("reinforce_lambda_truncated2_policy_loss.png")
plt.clf()
plt.plot(range(len(value_losses)), value_losses)
plt.savefig("reinforce_lambda_truncated2_critic_loss.png")
