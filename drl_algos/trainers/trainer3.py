import math
import collections
from collections import OrderedDict

import numpy as np
import gtimer as gt

from drl_algos import utils, eval_util

"""
Compared to trainer 2:
    - doesn't take environments
        - doesn't call their diagnostics method
    - needs path collector to have get_num_envs function

"""


class Trainer2(object):
    """Key differences with Trainer
        - path parsing for statistics now handled by PathCollector2
        - uses new logger
            - passed in as parameter
            - reports stats in timesteps rather than epochs
            - now actually performs checkpointing
        - doesn't save PathCollectors in checkpoints
        - can set frequency of checkpointing
    """

    def __init__(
        self,
        algorithm,
        exploration_path_collector,
        evaluation_path_collector,
        replay_buffer,
        logger,
        checkpoint_freq,
        model=None,
    ):
        self.algo = algorithm
        self.expl_path_collector = exploration_path_collector
        self.eval_path_collector = evaluation_path_collector
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.checkpoint_freq = checkpoint_freq
        self.model = model

        self.best_eval = -math.inf
        self._timestep = 0
        self._start_epoch = 1
        self.post_epoch_funcs = []

    def train(self, start_epoch=1):
        self._start_epoch = start_epoch
        self._train()
        snapshot = self._get_snapshot()
        self.logger.save_params("final", snapshot)

    def _train(self):
        raise NotImplementedError

    def training_mode(self, mode):
        """Sets training mode."""
        pass

    def _end_epoch(self, epoch):
        if self.checkpoint_freq is not None and self.checkpoint_freq > 0:
            if epoch % self.checkpoint_freq == 0:
                snapshot = self._get_snapshot()
                self.logger.save_params(self._timestep, snapshot)
        self._log_stats(epoch)

        self.expl_path_collector.end_epoch(epoch)
        self.eval_path_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.algo.end_epoch(epoch)
        if self.model is not None:
            self.model.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.algo.get_snapshot().items():
            snapshot['algorithm/' + k] = v
        # for k, v in self.expl_path_collector.get_snapshot().items():
        #     snapshot['exploration/' + k] = v
        # for k, v in self.eval_path_collector.get_snapshot().items():
        #     snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        if self.model is not None:
            for k, v in self.model.get_snapshot().items():
                snapshot['model/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        stats = {}

        """
        Replay Buffer
        """
        stats.update(
            utils.add_prefix(
                self.replay_buffer.get_diagnostics(),
                prefix="replay_buffer/"
            )
        )

        """
        Algorithm
        """
        stats.update(
            utils.add_prefix(
                self.algo.get_diagnostics(),
                prefix="algorithm/"
            )
        )

        """
        Model
        """
        if self.model is not None:
            stats.update(
                utils.add_prefix(
                    self.model.get_diagnostics(),
                    prefix="model/"
                )
            )

        """
        Exploration
        """
        stats.update(
            utils.add_prefix(
                self.expl_path_collector.get_diagnostics(),
                prefix="exploration/"
            )
        )

        """
        Evaluation
        """
        stats.update(
            utils.add_prefix(
                self.eval_path_collector.get_diagnostics(),
                prefix="evaluation/"
            )
        )
        eval_returns = stats["evaluation/Returns Mean"]
        if eval_returns > self.best_eval:
            self.best_eval = eval_returns
        snapshot = self._get_snapshot()
        self.logger.save_params("best", snapshot)

        """
        Misc
        """
        gt.stamp('logging')
        stats.update(self._get_epoch_timings())
        stats["Epoch"] = epoch
        self.logger.log(self._timestep, stats)

    def to(self, device):
        raise NotImplementedError

    def _get_epoch_timings(self):
        times_itrs = gt.get_times().stamps.itrs
        times = OrderedDict()
        epoch_time = 0
        for key in sorted(times_itrs):
            time = times_itrs[key][-1]
            epoch_time += time
            times['time/{} (s)'.format(key)] = time
        times['time/epoch (s)'] = epoch_time
        times['time/total (s)'] = gt.get_times().total
        return times


class BatchRLAlgorithm2(Trainer2):
    """Key differences with BatchRLAlgorithm2:
        - designed to work with PathCollector2
            - environment is reset only when done or exceeded max_path_length.
            In old implementation, the environment was always reset before
            gathering more data so training episodes would never exceed
            num_expl_steps_per_training_loop in length.
        - no need to pass max_path_length
        - specify number of evaluation episodes instead of evaluation steps
        - evaluation now performed after training instead of before
    """
    def __init__(
            self,
            algorithm,
            exploration_path_collector,
            evaluation_path_collector,
            replay_buffer,
            logger,
            batch_size,
            num_epochs,
            num_eval_eps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            checkpoint_freq=1,
    ):
        super().__init__(
            algorithm=algorithm,
            exploration_path_collector=exploration_path_collector,
            evaluation_path_collector=evaluation_path_collector,
            replay_buffer=replay_buffer,
            logger=logger,
            checkpoint_freq=checkpoint_freq,
        )

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_eval_eps_per_epoch = num_eval_eps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_path_collector.collect_new_paths(
                self.min_num_steps_before_training
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_path_collector.end_epoch(-1)
            self._timestep += (self.min_num_steps_before_training
                * self.expl_path_collector.get_num_envs())

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_path_collector.collect_new_paths(
                    self.num_expl_steps_per_train_loop
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.algo.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self.eval_path_collector.collect_new_episodes(
                self.num_eval_eps_per_epoch
            )
            gt.stamp('evaluation sampling')

            self._timestep += (self.num_train_loops_per_epoch
                               * self.num_expl_steps_per_train_loop
                               * self.expl_path_collector.get_num_envs())
            self._end_epoch(epoch)

    def to(self, device):
        self.algo.set_device(device)
        self.eval_path_collector._policy.to(device)

    def training_mode(self, mode):
        for net in self.algo.get_networks():
            net.train(mode)
