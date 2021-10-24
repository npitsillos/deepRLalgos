import time
import copy

import gym

from drl_algos.data.path_collector_parallel import ParallelPathCollector
from drl_algos.data.path_collector import MdpPathCollector2
from drl_algos.networks import policies
from behaviour_gym.scene import Basic as Scene
from behaviour_gym.primitive import ReachFarNoObj3 as Env
from drl_algos.data import Logger2 as Logger


# Past this point performance starts to saturate although it does still improve
# slightly
# 48 workers = 18.9x speed up
# 32 workers = 14x speed up
# 16 workers = 8.8x speed up
# 8 workers = 5x speed up
# 4 workers = 2.7x speed up
# 2 workers = 1.8x speed up

# Pendulum
# 8 workers = 3x speed up
# 2 workers = 1.2x speed up

def create_env(guiMode=False):
    # scene = Scene(guiMode=guiMode)
    # robot = scene.loadRobot("ur103f")
    # camera = None
    # scene.setInit()
    # return Env(scene=scene, robot=robot)
    return gym.make("Pendulum-v0").env

if __name__ == "__main__":
    env = create_env()
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    policy = policies.MlpGaussianPolicy(
                 hidden_sizes=[400,300],
                 input_size=obs_dim,
                 output_size=action_dim)

    n_envs = 2

    logger = Logger("test")

    para_collector = ParallelPathCollector(create_env, policy, 3, n_envs, 0)

    para_collector.collect_new_episodes(3)
    stats = para_collector.get_diagnostics()
    logger.log(2*n_envs, stats)
    para_collector.end_epoch(0)

    para_collector.collect_new_episodes(20)
    stats = para_collector.get_diagnostics()
    logger.log(4*n_envs, stats)
    para_collector.end_epoch(1)
