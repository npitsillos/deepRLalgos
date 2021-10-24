import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn, optim
import torch.nn.functional as F

from drl_algos.networks import policies, critics
from drl_algos.distributions import Discrete
from drl_algos.data import MdpPathCollector2, EpisodicReplayBuffer
from drl_algos import utils

SEED = 10

torch.manual_seed(SEED)
np.random.seed(SEED)

# Create and seed envs
env = gym.make('CartPole-v0').env
env.seed(SEED)

# Env dimensions
obs_dim = env.observation_space.low.size
action_dim = env.action_space.n


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r

# Create networks
policy = policies.MlpDiscretePolicy(
    obs_dim,
    action_dim,
    [16],
    layer_activation=F.relu
)
critic = critics.SacMlpCritic(
    obs_dim,
    1,
    [256,256]
)

path_collector = MdpPathCollector2(
    env=env,
    policy=policy,
    max_episode_length=200,
)

# replay_buffer = EpisodicReplayBuffer(
#     max_replay_buffer_size=200*10,
#     env=env,
#     max_path_len=200,
#     replace=False,
# )


total_rewards = []
value_losses = []
policy_losses = []
vf_criterion = nn.MSELoss()
policy_optim = optim.Adam(policy.parameters(),
                       lr=0.01)
critic_optim = optim.Adam(critic.parameters(),
                       lr=0.001)
for epoch in range(4000):
    # Get paths and process
    paths = path_collector.collect_new_episodes(10)
    batch_rewards = []
    batch_states = []
    batch_actions = []
    for path in paths:
        final_state = path["next_observations"][-1]
        returns = discount_rewards(
            path["rewards"])
        batch_rewards.extend(returns)
        batch_states.extend(path["observations"])
        batch_actions.extend(path["actions"])
        total_rewards.append(sum(path["rewards"]))
    batch_rewards = np.expand_dims(np.array(batch_rewards), 1)
    batch_states = np.array(batch_states)
    batch_actions = np.array(batch_actions)

    reward_tensor = torch.FloatTensor(batch_rewards)
    state_tensor = torch.FloatTensor(batch_states)
    action_tensor = torch.LongTensor(batch_actions)

    # Calculate critic loss
    value = critic(state_tensor)
    critic_loss = vf_criterion(value, reward_tensor)
    value_losses.append(critic_loss.item())

    # Calculate policy loss
    baseline = (reward_tensor - value).detach()
    dist = policy(state_tensor)
    # Below line looks like a potential area where my implementation may be
    # incorrect, not flattening generates unexpected results
    reinforce = -dist.log_prob(action_tensor.flatten()) * baseline.flatten()
    entropy = 1e-3 * dist.entropy()
    policy_loss = (reinforce - entropy).mean()
    policy_losses.append(policy_loss.item())

    # Backpropagate policy loss
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    # Backpropagate critic loss
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    batch_rewards = []
    batch_states = []
    batch_actions = []

    # Print running average
    avg_rewards = np.mean(total_rewards[-100:])
    print("\rEp: {} Average of last 100: {:.2f}".format(
         (epoch + 1)*10, avg_rewards), end="")

plt.plot(range(1, len(total_rewards)+1), total_rewards)
plt.savefig("reinforce_results.png")
plt.clf()
plt.plot(range(10, (len(policy_losses)+1)*10, 10), policy_losses)
plt.savefig("reinforce_policy.png")
plt.clf()
plt.plot(range(10, (len(value_losses)+1)*10, 10), value_losses)
plt.savefig("reinforce_values.png")
