import copy
from collections import deque, OrderedDict

import numpy as np

from drl_algos.data.rollouts import rollout
from drl_algos import utils, eval_util


class MdpPathCollector(object):

    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn

        self._num_steps_total = 0
        self._num_paths_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def collect_new_episodes(
            self,
            max_path_length,
            num_episodes,
    ):
        paths = []
        num_steps_collected = 0
        for ep in range(num_episodes):
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path['actions'])
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(utils.create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict


class MdpPathCollector2(object):
    """Key differences with MdpPathCollector:
        - diagnostics reports in terms of episodes rather than paths
        - diagnostics provides more information, not need to parse episodes in
        trainer class
        - rollout now a class function rather than an external function
        - now only resets environent when terminal encountered or hit maximum
        episode length rather than everytime rollout is called
        - simplified function calls

    ToDo:
        - Add new arg to ignore terminals due to exceeding
        max_episode_length, without a timefeature these terminals are
        effectively just noise. I think we may need our own custom wrapper to
        do this. Gym's timelimit does not account for the fact that there could
        be a valid terminal on the final timestep,
        https://github.com/openai/baselines/blob/d80acbb4d1f8b3359c143543dd3f21a3b12679c8/baselines/common/retro_wrappers.py#L6,
        we only want to know if the done was due to timelimit or not
    """

    def __init__(
            self,
            env,
            policy,
            max_episode_length,
            ignore_timeout_terminals=False,
            max_episodes_saved=None,
            render=False,
            render_kwargs=None,
            save_env_in_snapshot=True,
            reset_callback=None,
            preprocess_obs_for_policy_fn=None,
            action_kwargs=None,
            return_dict_obs=False,
            full_o_postprocess_func=None,

    ):
        if render_kwargs is None:
            render_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        if preprocess_obs_for_policy_fn is None:
            preprocess_obs_for_policy_fn = lambda x: x

        self._env = env
        self._policy = policy
        self._max_episode_length = max_episode_length
        self._ignore_timeout_terminals = ignore_timeout_terminals
        self._max_episodes_saved = max_episodes_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._save_env_in_snapshot = save_env_in_snapshot
        self._reset_callback = reset_callback
        self._preprocess_obs_for_policy_fn = preprocess_obs_for_policy_fn
        self._action_kwargs = action_kwargs
        self._return_dict_obs = return_dict_obs
        self._full_o_postprocess_func = full_o_postprocess_func

        self._epoch_episodes = deque(maxlen=self._max_episodes_saved)
        self._episode_step_counter = 0
        self._last_obs = None
        self._num_steps_total = 0
        self._num_episodes_total = 0

    def collect_new_paths(
            self,
            num_steps,
            force_reset=False,
    ):
        """Performs rollouts to gather the desired number of steps.

        Args:
            num_steps (int): total number of steps to collect
            force_reset (bool): if true will always reset environment before
                                any paths have been gathered. Potentially useful
                                for on-policy where may not want to continue
                                episode started by a stale policy? (Does this
                                matter?)

        Returns:
            list of paths where each path is a dictionary, each path is from
            a distinct episode, first and last paths may not cover an entire
            episode
        """
        paths = []
        num_steps_collected = 0

        # Resets environment even if not hit terminal or max_episode_length
        if force_reset:
            self._reset()

        # Perform rollouts until required number of steps gathered
        while num_steps_collected < num_steps:
            # Handle env resetting
            new_episode = False
            if self._should_reset():
                new_episode = True
                self._reset()

            # If completing an episode from previous epoch don't record it for
            # diagnostics
            skip_episode = False
            if (len(self._epoch_episodes) == 0 and
                self._episode_step_counter > 0):
                skip_episode = True

            # Don't gather more steps than maximum allowable in one episode
            # or than number required to meet num_steps arg
            max_path_length_this_loop = min(
                self._max_episode_length - self._episode_step_counter,
                num_steps - num_steps_collected,
            )

            # Gather a path
            path, path_len = self._rollout(
                max_path_length=max_path_length_this_loop,
            )
            num_steps_collected += path_len

            # Track episodes for later diagnostics
            if not skip_episode:
                if new_episode:
                    self._epoch_episodes.extend([path])
                else:
                    for key in path.keys():
                        self._epoch_episodes[-1][key].extend(path[key])

            # Convert path to numpy
            np_path = {}
            for key in path.keys():
                # In original impl, these keys were left as lists
                if key not in ["agent_infos", "env_infos", "full_observations",
                               "full_next_observations"]:
                    np_path[key] = np.array(path[key])
                else:
                    np_path[key] = path[key]
            paths.append(np_path)

        self._num_steps_total += num_steps_collected
        return paths

    def collect_new_episodes(
            self,
            num_episodes,
    ):
        """Performs rollouts to gather the desired number of episodes.

        Args:
            num_episodes (int): total number of episodes to collect

        Returns:
            list of paths where each path is a dictionary, each path is from
            a distinct episode
        """
        paths = []
        num_steps_collected = 0

        # Gather episodes
        for ep in range(num_episodes):
            # Reset the environment
            self._reset()

            # Gather a complete episode
            path, path_len = self._rollout(
                max_path_length=self._max_episode_length,
            )
            num_steps_collected += path_len

            # Track for diagnostics
            self._epoch_episodes.extend([path])

            # Convert path to numpy
            np_path = {}
            for key in path.keys():
                # In original impl, these keys were left as lists
                if key not in ["agent_infos", "env_infos", "full_observations",
                               "full_next_observations"]:
                    np_path[key] = np.array(path[key])
                else:
                    np_path[key] = path[key]
            paths.append(np_path)

        self._num_steps_total += num_steps_collected
        return paths

    def _rollout(
        self,
        max_path_length
    ):
        raw_obs = []
        raw_next_obs = []
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        next_observations = []
        path_length = 0

        # Perform rollout starting from last seen observation
        o = self._last_obs
        while path_length < max_path_length:
            raw_obs.append(o)

            # Preprocess obs for agent then get action
            o_for_agent = self._preprocess_obs_for_policy_fn(o)
            a, agent_info = self._policy.get_action(o_for_agent,
                                                    **self._action_kwargs)

            # Perform postprocessing of obs
            if self._full_o_postprocess_func:
                self._full_o_postprocess_func(self._env, self._policy, o)

            # Step through environment then render
            next_o, r, d, env_info = self._env.step(copy.deepcopy(a))
            self._episode_step_counter += 1
            if self._render:
                self._env.render(**self._render_kwargs)

            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            next_observations.append(next_o)
            raw_next_obs.append(next_o)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            o = next_o

            # If terminal then end rollout
            if d:
                break

        # Track final observation for next rollout call
        self._last_obs = o

        # TODO - avoid needing to convert to numpy
        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1).tolist()
        else:
            actions = actions.tolist()
        observations = observations
        next_observations = next_observations
        if self._return_dict_obs:
            observations = raw_obs
            next_observations = raw_next_obs
        # TODO - avoid needing to convert to numpy
        rewards = np.array(rewards)
        if len(rewards.shape) == 1:
            rewards = rewards.reshape(-1, 1).tolist()
        else:
            rewards = reward.tolist()
        # TODO - avoid needing to convert to numpy
        terminals = np.array(terminals).reshape(-1, 1).tolist()
        return dict(
            observations=observations,
            actions=actions, # no longer np array
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            agent_infos=agent_infos,
            env_infos=env_infos,
            full_observations=raw_obs,
            full_next_observations=raw_obs,
        ), path_length

    def end_epoch(self, epoch):
        self._epoch_episodes = deque(maxlen=self._max_episodes_saved)

    def get_epoch_episodes(self):
        """Returns epoch episodes used for diagnostics.

        Only call this function after calling get_diagnostics, as it removes
        incomplete episodes.
        """
        return self._epoch_episodes

    def get_diagnostics(self):
        """Returns diagnostic data as a dictionary.

        This should only be called once per epoch, after the point at which no
        more data will be collected because this function removes incomplete
        episodes. Removed episodes could be added back but why bother?
        """
        # Don't report with final episode if it is incomplete
        last_ep_len = len(self._epoch_episodes[-1]['actions'])
        if (last_ep_len < self._max_episode_length and
            not self._epoch_episodes[-1]['terminals'][-1][0]):
            last_ep = self._epoch_episodes.pop()

        episode_lens = [len(episode['actions']) for episode in self._epoch_episodes]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num episodes total', self._num_episodes_total),
        ])
        stats.update(utils.create_stats_ordered_dict(
            "episode length",
            episode_lens,
            always_show_all_stats=True,
        ))

        # Convert to numpy prior to getting generic information
        for episode in self._epoch_episodes:
            episode["rewards"] = np.array(episode["rewards"])
            episode["actions"] = np.array(episode["actions"])
        stats.update(eval_util.get_generic_episode_information(
            self._epoch_episodes)
        )

        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict

    def _reset(self):
        self._policy.reset()
        self._last_obs = self._env.reset()
        if self._reset_callback:
            self._reset_callback(self._env, self._policy, self._last_obs)
        if self._render:
            self._env.render(**self._render_kwargs)
        self._episode_step_counter = 0
        self._num_episodes_total += 1

    def _should_reset(self):
        # If reached max_episode_length then should reset
        if self._episode_step_counter == self._max_episode_length:
            return True
        # If no last_obs, i.e., very first episode, then should reset
        if self._last_obs is None:
            return True
        return False
