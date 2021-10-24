import gym
import torch
import numpy as np

from drl_algos.algos import SAC
from drl_algos.networks import critics
from drl_algos.networks import policies
from drl_algos.networks import models
from drl_algos.data import MdpPathCollector2 as MdpPathCollector
from drl_algos.data import EpisodicReplayBuffer as ReplayBuffer
from drl_algos.data import Logger2 as Logger
from drl_algos.trainers import BatchRLAlgorithm2 as Trainer
from drl_algos import utils


torch.manual_seed(0)
np.random.seed(0)

# Device for the networks
DEVICE = "cuda:0"

# Environment info
ENV_NAME = "Pendulum-v0"

BATCH_SIZE = 50
SEQUENCE_LEN = 50

replay_buffer = torch.load("buffer.pkl")

# Create and seed envs
env = gym.make(ENV_NAME).env
eval_env = gym.make(ENV_NAME).env
env.seed(0)
eval_env.seed(1)

# Env dimensions
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

# Create model
model = models.MlpDreamer(obs_dim, action_dim)

logger = Logger("dreamer_pendulum_model")
for i in range(500):
    train_data = replay_buffer.random_batch(
        BATCH_SIZE,
        SEQUENCE_LEN,
    )
    model.train(train_data)
    logger.log(i, model.get_diagnostics())
torch.save(model, "model.pkl")
