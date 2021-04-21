from collections import OrderedDict
import warnings

from gym.spaces import Discrete, Box
import numpy as np


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


class RolloutBuffer(object):

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        env_info_sizes = None,
        replace = True,
        gamma = 0.99,
        gae_lambda = 0.95
    ):
        self.env = env

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()
        self._env_info_sizes = env_info_sizes
        self._observation_dim = get_dim(env.observation_space)
        self._action_space = env.action_space
        self._action_dim = get_dim(env.action_space)
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._replace = replace

        self._top = 0
        self._size = 0

    def add_path(self, path):
        """
        Add a path to the replay buffer.
        This default implementation naively goes through every step, but you
        may want to optimize this.
        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.
        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        if isinstance(self._action_space, Discrete):
           new_action = np.zeros(self._action_dim)
           new_action[action] = 1
        else:
           new_action = action

        self._observations[self._top] = observation
        self._actions[self._top] = new_action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):

        num_states = self._size
        batch_start = np.arange(0, num_states, batch_size)
        indices = np.arange(num_states, dtype=np.int64)
        np.random.shuffle(indices)
        batch_indices = [indices[i:i+batch_size] for i in batch_start]
        batches = []
        for batch_index in batch_indices:

            batches.append(dict(
                observations=self._observations[batch_index],
                actions=self._actions[batch_index],
                rewards=self._rewards[batch_index],
                terminals=self._terminals[batch_index],
                next_observations=self._next_obs[batch_index],
            ))
        # batch = dict(
        #         observations=self._observations,
        #         actions=self._actions,
        #         rewards=self._rewards,
        #         terminals=self._terminals,
        #         next_observations=self._next_obs,
        #     )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batches

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def end_epoch(self, epoch):
        self._observations = np.zeros((self._max_replay_buffer_size,
                                       self._observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((self._max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((self._max_replay_buffer_size, self._action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((self._max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((self._max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in self._env_info_sizes.keys():
            self._env_infos[key] = np.zeros((self._max_replay_buffer_size, size))
        self._env_info_keys = self._env_info_sizes.keys()

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def get_snapshot(self):
        return {}