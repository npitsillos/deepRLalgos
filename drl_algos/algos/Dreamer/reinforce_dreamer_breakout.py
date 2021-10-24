import time
import threading
import sys

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn, optim
import torch.nn.functional as F

import drl_algos
from drl_algos.networks.mydreamer import Model, ActorCritic
from drl_algos.networks import policies, critics, models
from drl_algos.distributions import Discrete
from drl_algos.data import ModelPathCollector2, EpisodicReplayBuffer, Logger2
from drl_algos import utils


class Atari:

    LOCK = threading.Lock()

    def __init__(
                self, name, action_repeat=4, size=(64, 64), grayscale=True, noops=30,
                life_done=False, sticky=True, all_actions=True):
        assert size[0] == size[1]
        import gym.wrappers
        import gym.envs.atari
        if name == 'james_bond':
            name = 'jamesbond'
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name, obs_type='image', frameskip=1,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=all_actions)
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
        self._env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale)
        self._size = size
        self._grayscale = grayscale

        self.observation_space = gym.spaces.Box(
            np.float32(0), np.float32(1), (1,64,64)
        )
        self.action_space = self._env.action_space


    def seed(self, seed):
        self._env.seed(seed)

    @property
    def obs_space(self):
        shape = (1 if self._grayscale else 3,) + self._size
        return shape

    @property
    def act_space(self):
        return {'action': self._env.action_space}

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        if self._grayscale:
            image = image[None, ...]
        image = np.array(image) / 255
        return image, reward, done, info

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[None, ...]
        image = np.array(image) / 255
        return image

    def close(self):
        return self._env.close()


SEED = 10

torch.manual_seed(SEED)
np.random.seed(SEED)

# Create and seed envs
env = Atari("breakout")
env.seed(SEED)
eval_env = Atari("breakout")
eval_env.seed(SEED+1)

# Env dimensions
obs_dim = (1,64,64)
action_dim = 18

# Create networks
model = Model(
    stoch_size=32,
    deter_size=600,
    discrete_size=32,
    action_size=action_dim,
    rssm_hidden_size=600,
    obs_size=obs_dim,
    hidden_sizes=[400, 400, 400, 400],
)
actor_critic = ActorCritic(model, [400, 400, 400, 400], horizon=15)

policy = actor_critic.policy
eval_policy = policies.MakeDeterministic2(policy)

path_collector = ModelPathCollector2(
    env=env,
    policy=policy,
    model=model,
    max_episode_length=27000,
)
eval_path_collector = ModelPathCollector2(
    env=eval_env,
    policy=eval_policy,
    model=model,
    max_episode_length=27000,
    deterministic=True,
)

replay_buffer = EpisodicReplayBuffer(
    max_replay_buffer_size=1000,
    env=env,
    max_path_len=27000,
    replace=False,
)

logger = Logger2("dreamer_breakout")

# Move to device
DEVICE = "cuda:0"

actor_critic.to(DEVICE)
model.to(DEVICE)
policy.to(DEVICE)
eval_policy.to(DEVICE)

# Prefill the experience replay
paths = path_collector.collect_new_paths(50000)
replay_buffer.add_paths(paths)
path_collector.end_epoch(-1)

# Pretrain model once
_ = model.train(replay_buffer.random_batch(50, 49))

def get_snapshot():
    snapshot = {}
    for k, v in model.get_snapshot().items():
        snapshot['model/' + k] = v
    for k, v in actor_critic.get_snapshot().items():
        snapshot['actor_critic/' + k] = v

# Runs for a total of 50M steps which is 200M environment steps (action repeat)
# Each epoch being 50k environment steps
best_eval = -999999999
for epoch in range(1, 1001):
    # Only reports every 12500 gradients steps, i.e. 50k steps (200k env steps)
    for i in range(25):
        start = time.time()
        # Get paths and process
        start2 = time.time()
        paths = path_collector.collect_new_paths(4)
        # print("path collector", time.time()-start2)
        start2 = time.time()
        replay_buffer.add_paths(paths)
        # print("buffer storing", time.time()-start2)

        # Train dreamer
        start2 = time.time()
        paths = replay_buffer.random_batch(50, 49)
        # print("buffer sampling", time.time()-start2)
        start2 = time.time()
        post, discounts = model.train(paths)
        print("model training", time.time()-start2)
        start2 = time.time()
        actor_critic.train(post, discounts)
        # print("actor critic training", time.time()-start2)

        print("Finished iter", i)
        print("Took", time.time()-start)
        print()

    # perform eval
    _ = eval_path_collector.collect_new_episodes(10)

    # stats = {}
    # stats.update(
    #     utils.add_prefix(
    #         replay_buffer.get_diagnostics(),
    #         prefix="replay_buffer/"
    #     )
    # )
    # stats.update(
    #     utils.add_prefix(
    #         model.get_diagnostics(),
    #         prefix="model/"
    #     )
    # )
    # stats.update(
    #     utils.add_prefix(
    #         actor_critic.get_diagnostics(),
    #         prefix="actor_critic/"
    #     )
    # )
    # stats.update(
    #     utils.add_prefix(
    #         path_collector.get_diagnostics(),
    #         prefix="exploration/"
    #     )
    # )
    # stats.update(
    #     utils.add_prefix(
    #         eval_path_collector.get_diagnostics(),
    #         prefix="evaluation/"
    #     )
    # )
    # stats["Epoch"] = epoch
    # logger.log(epoch*50000, stats)
    #
    # eval_score = stats["evaluation/Returns Mean"]
    # if eval_score > best_eval:
    #     best_eval = eval_score
    #     logger.save_params("best", get_snapshot())

    path_collector.end_epoch(epoch)
    eval_path_collector.end_epoch(epoch)
    replay_buffer.end_epoch(epoch)
    model.end_epoch(epoch)
    actor_critic.end_epoch(epoch)

logger.save_params("final", get_snapshot())
