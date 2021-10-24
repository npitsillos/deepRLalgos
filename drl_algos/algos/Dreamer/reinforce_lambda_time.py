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


"""
NOTES:
    - at higher policy learning rate this seems to be more stable than
    reinforce_lambda and equivalent to reinforce
"""

SEED = 10

torch.manual_seed(SEED)
np.random.seed(SEED)

class TimeWrapper:

    def __init__(self, env, timelimit):
        self.env = env
        self.timelimit = timelimit
        self.stepCounter = 0

    def reset(self):
        self.stepCounter = 0
        o = self.env.reset()
        return np.concatenate((o, self._timefeature()))

    def step(self, action):
        self.stepCounter += 1
        o, r, d, env_info = self.env.step(action)
        o = np.concatenate((o, self._timefeature()))
        if self.stepCounter == self.timelimit:
            d = True
        return o, r, d, env_info

    def _timefeature(self):
        return [((self.stepCounter / self.timelimit) * 2) - 1]

# Create and seed envs
env = gym.make('CartPole-v0').env
env.seed(SEED)

# Env dimensions
obs_dim = env.observation_space.low.size + 1
action_dim = env.action_space.n
env = TimeWrapper(env, 200)


def lambda_returns(rewards, states, final_state, dones, critic, lam=0.95, gamma=0.99):
    returns = []
    prev_target = None
    # Iterate backwards through list starting from last index
    for i in range(len(rewards)-1, -1, -1):
        reward = torch.tensor(rewards[i])
        done = dones[i]
        if done:
            discount = 0.0
        else:
            discount = gamma

        if prev_target is None:
            next_state = torch.FloatTensor(final_state).unsqueeze(0)
            target = reward + discount * critic(torch.FloatTensor(next_state))
        else:
            next_state = torch.FloatTensor(states[i+1]).unsqueeze(0)
            target = reward + discount * ((1.0-lam)*critic(torch.FloatTensor(next_state)) + lam*prev_target)

        returns.append(target.detach().numpy()[0][0])
        prev_target = target

    return np.array(returns)[::-1]

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
critic_target = critics.SacMlpCritic(
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
                       lr=0.001)
critic_optim = optim.Adam(critic.parameters(),
                       lr=0.01)
for epoch in range(4000):
    # Get paths and process
    paths = path_collector.collect_new_episodes(10)
    batch_rewards = []
    batch_states = []
    batch_actions = []
    for path in paths:
        final_state = path["next_observations"][-1]
        returns = lambda_returns(
            path["rewards"], path["observations"], final_state, path["terminals"], critic_target)
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

    # Soft update the target
    utils.soft_update(critic, critic_target, 5e-3)

    batch_rewards = []
    batch_states = []
    batch_actions = []

    # Print running average
    avg_rewards = np.mean(total_rewards[-100:])
    print("\rEp: {} Average of last 100: {:.2f}".format(
         (epoch + 1)*10, avg_rewards), end="")

plt.plot(range(1, len(total_rewards)+1), total_rewards)
plt.savefig("reinforce_lambda_timed_results.png")
plt.clf()
plt.plot(range(10, (len(policy_losses)+1)*10, 10), policy_losses)
plt.savefig("reinforce_lambda_timed_policy.png")
plt.clf()
plt.plot(range(10, (len(value_losses)+1)*10, 10), value_losses)
plt.savefig("reinforce_lambda_timed_values.png")
