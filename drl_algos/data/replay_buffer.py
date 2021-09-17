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


class ReplayBuffer(object):

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        env_info_sizes = None,
        replace = True,
    ):
        """Note - it is standard practice to sample with replace=True even
        though it may sample a single experience multiple times within a single
        batch. It is more computationally efficient and should have negligible
        performance impact."""
        self.env = env

        # Get info sizes if provided
        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        # Get dimensions
        self._observation_dim = get_dim(env.observation_space)
        self._action_space = env.action_space
        self._action_dim = get_dim(env.action_space)
        self._max_replay_buffer_size = max_replay_buffer_size

        # Create local 2D buffers
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))

        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        # Set up env infos
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._replace = replace

        # Track for indexing
        self._top = 0
        self._size = 0

    def add_path(self, path):
        """Add a path to the replay buffer.

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
        # Handle discrete/continuous
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
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

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
        return

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def get_snapshot(self):
        return {}

class EpisodicReplayBuffer(ReplayBuffer):
    """This implementation is only valid for a SINGLE worker.

    Episode terminals automatically detected so episode can be added as multiple
    paths.

    Multiple workers could easily be implemented by only adding completed
    episodes although this may be impractical for long horizon environments.
    Best way may be to have the PathCollector return [workers, paths] in a
    consistent manner then have replay buffer track each worker's paths for
    terminals.
    """

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        max_path_len,
        env_info_sizes=None,
        replace=True,
    ):
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes,
            replace=replace
        )
        self.max_path_len = max_path_len
        # Stores array of indices for each episode in the buffer
        self.episodes = []

    def add_path(self, path):
        # Track index where each experience is stored
        trajectory = []

        # Continue last episode if not reached max_path_len or terminal
        new_ep = True
        if self._size > 0: # skip if first path ever added
            if len(self.episodes[-1]) < self.max_path_len:
                if not self._terminals[self._top - 1]:
                    new_ep = False

        # Update episodes indices to remove experiences about to be replaced
        path_len = len(path["actions"])
        clear_space = (self._size + path_len) - self._max_replay_buffer_size
        while clear_space > 0:
            ep = self.episodes[0]
            if len(ep) <= clear_space:
                # Free oldest episode
                self.episodes.pop(0)
                clear_space -= len(ep)
            else:
                # Free oldest experiences
                self.episodes[0] = self.episodes[0][clear_space:]
                break

        # Add each experience to the buffer
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
            trajectory.append(self._top)
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )

        if new_ep:
            # Add new trajectory to episodes array
            self.episodes.append(trajectory)
        else:
            # Append trajectory onto end of most recent episode
            self.episodes[-1] += trajectory
        self.terminate_episode()

    def random_batch(
        self,
        batch_size,
        sequence_len=1,
        strict_sequence_len=True
    ):
        # If sequence length is 1 then return default batch
        if sequence_len == 1:
            return super().random_batch(batch_size=batch_size)

        if strict_sequence_len:
            valid_eps = []
            ep_lens = []
            # Filter out episodes shorter than sequence_len
            for index in range(len(self.episodes)):
                ep_len = len(self.episodes[index])
                if ep_len >= sequence_len:
                    valid_eps.append(index)
                    ep_lens.append(ep_len)
        else:
            # Sample from all episodes
            valid_eps = [id for id in range(len(self.episodes))]
            ep_lens = [len(ep) for ep in self.episodes]

        # Probability of sampling an episode increases with episode length to
        # preserve uniform sampling of experiences
        # e.g., with uniform episode sampling experiences are more likely to be
        # sampled if they are contained within a shorter episode
        ep_sample_probs = np.array(ep_lens) / np.sum(ep_lens)
        ep_indices = np.random.choice(
            valid_eps,
            size=batch_size,
            replace=True,
            p=ep_sample_probs,
        )

        # Sample trajectory randomly from each episode
        episodes = [self.episodes[index] for index in ep_indices]
        obs_sequences = []
        action_sequences = []
        reward_sequences = []
        terminal_sequences = []
        next_obs_sequences = []
        for ep in episodes:
            ep_len = len(ep)
            index = np.random.choice(ep_len)
            if strict_sequence_len:
                # Clip index to ensure enough experiences sampled
                start_index = np.clip(index, None, ep_len-sequence_len)
                end_index = start_index + sequence_len
            else:
                start_index = np.clip(index-sequence_len, 0, None)
                end_index = index+1
            indices = np.array(ep[start_index: end_index])

            # Track sequences
            obs_sequences.append(self._observations[indices])
            action_sequences.append(self._actions[indices])
            reward_sequences.append(self._rewards[indices])
            terminal_sequences.append(self._terminals[indices])
            next_obs_sequences.append(self._next_obs[indices])

            # TODO - Add support for env_info_keys
            # for key in self._env_info_keys:
            #     assert key not in sequence.keys()
            #     sequence[key] = self._env_infos[key][indices]

        batch = dict(
            observations=np.stack(obs_sequences),
            actions=np.stack(action_sequences),
            rewards=np.stack(reward_sequences),
            terminals=np.stack(terminal_sequences),
            next_observations=np.stack(next_obs_sequences),
        )
        return batch
