import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn
from torch import optim

from drl_algos.networks import critics, policies, models
from drl_algos.distributions import Discrete

"""
NOTES:
    - seems roughly equivalent to original's performance
    - if you are instansiating the Categorical distribution with probabilities
    then output should be fed through softmax
    - if you are instansiating with logits then output of fully connected should
    be fed straight into distribution
"""

SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

class policy_estimator():
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs))

    def predict(self, state):
        logits = self.network(torch.FloatTensor(state))
        return Discrete(logits)

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def reinforce(env, policy_estimator, num_episodes=10000,
              batch_size=10, gamma=0.99):    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(policy_estimator.parameters(),
                           lr=0.0001)

    ep = 0
    while ep < num_episodes:
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action, _ = policy_estimator.get_action(s_0)
            s_1, r, done, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(
                    rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    # Actions are used as indices, must be
                    # LongTensor
                    action_tensor = torch.LongTensor(
                       batch_actions)

                    # Calculate loss
                    dist = policy_estimator(state_tensor)
                    loss = (-dist.log_prob(action_tensor) * reward_tensor).mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("\rEp: {} Average of last 100: {:.2f}".format(
                     ep + 1, avg_rewards), end="")
                ep += 1

    return total_rewards

env = gym.make('Pendulum-v0')
env.seed(SEED)
policy = policies.MlpGaussianPolicy(
             hidden_sizes=[64],
             input_size=3,
             output_size=1,
         )
rewards = reinforce(env, policy)

plt.plot(range(1, len(rewards)+1), rewards)
plt.savefig("results_distribution_pendulum.png")
