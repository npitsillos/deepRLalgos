import copy
from collections import deque, OrderedDict

from torch import multiprocessing as mp
import numpy as np
import torch.distributions as td # CHANGELOG - Added for decoder

from drl_algos.networks.policies import DiscretePolicy2
from drl_algos.data.rollouts import rollout
from drl_algos import utils, eval_util


def worker(remote, parent_remote, env_fn, seed):
    parent_remote.close()
    env = env_fn()
    env.seed(seed)

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == "step":
                remote.send(env.step(copy.deepcopy(data)))

            if cmd == "reset":
                remote.send((env.reset(),))

        except EOFError:
            break


class ProcessManager(object):

    def __init__(self, ctx, env_fn, n_envs, max_ep_len, seed):
        self._remotes, work_remotes = zip(
            *[ctx.Pipe() for _ in range(n_envs)])
        for i in range(n_envs):
            work_remote = work_remotes[i]
            remote = self._remotes[i]
            work_seed = seed+i
            args = (work_remote, remote, env_fn, work_seed)
            process = ctx.Process(target=worker, args=args, daemon=True)
            process.start()
            work_remote.close()
            remote.send(("reset", None))

        self._n_envs = n_envs
        self._max_ep_len = max_ep_len
        self._episodes = [[self._create_dict()] for i in range(n_envs)]
        self._step_counter = [0 for i in range(n_envs)]
        self._last_obs = [remote.recv()[0] for remote in self._remotes]

    def step(self, actions):
        for i in range(self._n_envs):
            self._send_step(i, actions[i])

        batch_next_obs = []
        batch_rewards = []
        batch_done = []
        batch_info = []
        for i in range(self._n_envs):
            obs, reward, done, info = self._recv_step(i, actions[i])
            batch_next_obs.append(obs)
            batch_rewards.append(reward)
            batch_done.append(done)
            batch_info.append(info)
            if self.is_done(i, done):
                self._send_reset(i)

        for i in range(self._n_envs):
            if self.is_done(i, batch_done[i]):
                self._recv_reset(i)

        return batch_next_obs, batch_rewards, batch_done, batch_info

    def collect_episodes(self, num_episodes, policy, action_kwargs):
        """Only made with intention of using with evaluation path collector.

        Doesn't return episodes since evaluation episodes only used for
        diagnostics. Assumes all environments reset before calling (since it
        automatically resets any finished environments for next call).
        """
        eps_complete = 0
        running_remotes = []

        while eps_complete < num_episodes:
            batch_actions, _ = policy.get_action(
                np.array(self._last_obs), **action_kwargs)

            for i in range(self._n_envs):
                if (i not in running_remotes and
                        len(running_remotes) + eps_complete < num_episodes):
                    running_remotes.append(i)
                if i in running_remotes:
                    self._send_step(i, batch_actions[i])

            reset_remotes = []
            for i in running_remotes:
                obs, reward, done, info = self._recv_step(i, batch_actions[i])
                if self.is_done(i, done):
                    reset_remotes.append(i)
                    self._send_reset(i)
                    eps_complete += 1

            for i in reset_remotes:
                running_remotes.remove(i)
                self._recv_reset(i)

    def end_epoch(self):
        # Always carry over last episode in progress
        for i in range(self._n_envs):
            self._episodes[i] = [self._episodes[i][-1]]

    def get_obs(self):
        return self._last_obs

    def get_episodes(self):
        # Last episode is never complete
        complete_episodes = []
        for i in range(self._n_envs):
            complete_episodes.extend(self._episodes[i][:-1])
        return complete_episodes

    def is_done(self, i, done):
        return (done or
            self._step_counter[i] == self._max_ep_len or
            self._step_counter[i] == 0)

    def _send_reset(self, i):
        self._remotes[i].send(("reset", None))
        self._step_counter[i] = 0

    def _send_step(self, i, action):
        self._remotes[i].send(("step", action))
        self._step_counter[i] += 1

    def _recv_reset(self, i):
        self._episodes[i].append(self._create_dict())
        self._last_obs[i] = self._remotes[i].recv()[0]

    def _recv_step(self, i, action):
        obs, reward, done, info = self._remotes[i].recv()
        self._episodes[i][-1]["rewards"].append(reward)
        self._episodes[i][-1]["actions"].append(action)
        self._episodes[i][-1]["env_infos"].append(info)
        self._last_obs[i] = obs
        return obs, reward, done, info

    def _create_dict(self):
        return {"rewards": [], "actions": [], "env_infos": []}


class ParallelPathCollector(object):

    def __init__(
            self, env_fn, policy, max_episode_length, n_envs, seed,
            action_kwargs={}, start_method=None):
        self._policy = policy
        self._n_envs = n_envs
        self._action_kwargs = action_kwargs
        self._num_steps_total = 0
        self._num_episodes_total = 0 # Num episodes completed

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
            ctx = mp.get_context(start_method)
        self._manager = ProcessManager(
            ctx, env_fn, n_envs, max_episode_length, seed)

    def collect_new_paths(self, num_steps):
        observations = []
        actions = []
        rewards = []
        terminals = []
        env_infos = []
        next_observations = []
        agent_infos = []

        for i in range(num_steps):
            batch_obs = np.array(self._manager.get_obs()) # np.array necessary for indexing with None as done in the policy get_action
            batch_actions, _ = self._policy.get_action(
                batch_obs, **self._action_kwargs)
            next_obs, reward, done, env_info = self._manager.step(batch_actions)

            observations.extend(batch_obs)
            actions.extend(batch_actions)
            rewards.extend(reward)
            terminals.extend(done)
            env_infos.extend(env_info)
            next_observations.extend(next_obs)
            agent_infos.extend([[] for i in range(self._n_envs)])

            self._num_steps_total += self._n_envs
            for i in range(len(done)):
                if self._manager.is_done(i, done[i]):
                    self._num_episodes_total += 1

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        rewards = np.array(rewards)
        if len(rewards.shape) == 1:
            rewards = rewards.reshape(-1, 1)
        terminals = np.array(terminals).reshape(-1, 1)
        return [dict(
            observations=np.array(observations),
            actions=actions,
            rewards=rewards,
            next_observations=np.array(next_observations),
            terminals=terminals,
            env_infos=env_infos,
            agent_infos=agent_infos)]

    def collect_new_episodes(self, num_episodes):
        self._manager.collect_episodes(
            num_episodes, self._policy, self._action_kwargs)
        self._num_episodes_total += num_episodes

    def end_epoch(self, epoch):
        self._manager.end_epoch()

    def get_epoch_episodes(self):
        """Returns epoch episodes used for diagnostics."""
        return self._manager.get_episodes()

    def get_diagnostics(self):
        """Returns diagnostic data as a dictionary."""
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num episodes total', self._num_episodes_total),
        ])

        episodes = self.get_epoch_episodes()
        episode_lens = [len(episode['actions']) for episode in episodes]
        stats.update(utils.create_stats_ordered_dict(
            "episode length",
            episode_lens,
            always_show_all_stats=True,
        ))

        for episode in episodes:
            episode["rewards"] = np.array(episode["rewards"])
            episode["actions"] = np.array(episode["actions"])
        stats.update(eval_util.get_generic_episode_information(episodes))

        # Need to add in placeholders for env info keys
        if len(episodes) == 0:
            for key in self._manager._episodes[0][0]["env_infos"][0].keys():
                stats.update(utils.create_stats_ordered_dict(
                    key,
                    [],
                    stat_prefix='env_infos/final/'))
                stats.update(utils.create_stats_ordered_dict(
                    key,
                    [],
                    stat_prefix='env_infos/initial/'))
                stats.update(utils.create_stats_ordered_dict(
                    key,
                    [],
                    stat_prefix='env_infos/'))

        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        return snapshot_dict

    def get_num_envs(self):
        return self._n_envs
