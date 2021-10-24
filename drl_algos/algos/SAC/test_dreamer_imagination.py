import gym
import torch
from torch import nn
import numpy as np

from drl_algos.algos import SAC
from drl_algos.networks import critics
from drl_algos.networks import policies
from drl_algos.networks import models
from drl_algos.data import MdpPathCollector2 as MdpPathCollector
from drl_algos.data import EpisodicReplayBuffer as ReplayBuffer
from drl_algos.data import Logger2 as Logger
from drl_algos.distributions import Delta
from drl_algos.trainers import BatchRLAlgorithm2 as Trainer
from drl_algos import utils

"""
Notes to remind myself:
    when encoding a sequence:
        if the sequence is 50 actions long then each array will be 51 in length
            - initial observation plus 50 next observations
                - so index 0 and 1 both correspond to the experience at index 0
                - index 2 is the next obs of the experience at index 1
            - first action is a 0 concatenated on and should be ignored
                - should maybe be at the end not the beginning?
            - first reward should be ignored
                - there has been no action yet to trigger a reward
            - first gamma should be ignored
                - but in practice never start at a terminal so it is fine

    when dreaming:
        if horizon is 15 then sequence will be 16 in length
        the first reward and gamma corresponds to the starting model_state,
            i.e., the posterior
        the first model state is the posterior
        the final action is not imagined
"""

SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

# Device for the networks
DEVICE = "cuda:0"

replay_buffer = torch.load("buffer.pkl")
model = torch.load("model.pkl")

sequence = replay_buffer.random_batch(
    1,
    2,
)
print(sequence["observations"].shape)
print(sequence["terminals"])
"""
burnin = {
    "observations": np.array([sequence["observations"][0][:50].tolist()]),
    "actions": np.array([sequence["actions"][0][:50].tolist()]),
    "next_observations": np.array([sequence["next_observations"][0][:50].tolist()]),
}

ground_truth = {
    "observations": np.array([sequence["observations"][0][50:65].tolist()]),
    "actions": np.array([sequence["actions"][0][50:65].tolist()]),
    "next_observations": np.array([sequence["next_observations"][0][50:65].tolist()]),
}

model_states, actions, rewards, gammas = model.encode(sequence)

decoded_obs = []
for i in range(50):
    decoded_obs.append(model.obs_decoder(model_states[0][i]).detach())

print("Ground truth observations")
print(sequence["observations"][0][:50])
print("Predicted observations")
print(torch.stack(decoded_obs))
print("Observation Errors")
print(torch.tensor(sequence["observations"][0][:50]) - torch.stack(decoded_obs))

print("Ground truth rewards")
print(sequence["rewards"][0][:50])
print("Predicted rewards")
print(rewards[1:51])
print("Reward Errors")
print(torch.tensor(sequence["rewards"][0][:50]) - rewards[1:51])

# Pick 25th model state as starting point for imagination
start_state = (model_states[0][25].unsqueeze(0), model_states[1][25].unsqueeze(0), model_states[2][25].unsqueeze(0))

# Create a policy for the 15 actions after the starting model state
class Policy(nn.Module):

    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.index = 0

    def forward(self, state):
        action = self.actions[self.index]
        self.index += 1
        return Delta(torch.tensor([[action]]))

policy = Policy(actions[26:42])

# Imagine running that policy
states_img, actions_img, log_pis_img, rewards_img, gammas_img = model.dream(start_state, policy, 15)

print("Ground truth rewards")
print(sequence["rewards"][0][24:40])
print("Imagined rewards")
print(rewards_img)
print("Reward error")
print(torch.tensor(sequence["rewards"][0][24:40]) - rewards_img[0])

print("Ground truth actions")
print(sequence["actions"][0][25:41])
print("Executed actions")
print(actions_img)
"""
