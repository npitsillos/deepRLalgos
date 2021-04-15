import collections
from collections import OrderedDict

import numpy as np
import gtimer as gt

from drl_algos.utils.logging import logger
from drl_algos.utils import utils, eval_util


class Trainer(object):

    def __init__(
        self,
        algorithm,
        exploration_env,
        evaluation_env,
        exploration_path_collector,
        evaluation_path_collector,
        replay_buffer,
    ):
        self.algo = algorithm
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_path_collector = exploration_path_collector
        self.eval_path_collector = evaluation_path_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        raise NotImplementedError

    def training_mode(self, mode):
        """Sets training mode."""
        pass

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_path_collector.end_epoch(epoch)
        self.eval_path_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.algo.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.algo.get_snapshot().items():
            snapshot['algorithm/' + k] = v
        for k, v in self.expl_path_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_path_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        # logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
        print("Epoch {} finished".format(epoch))

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Algorithm
        """
        logger.record_dict(self.algo.get_diagnostics(), prefix='algorithm/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_path_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_path_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_path_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_path_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(self._get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

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


class BatchRLAlgorithm(Trainer):
    def __init__(
            self,
            algorithm,
            exploration_env,
            evaluation_env,
            exploration_path_collector,
            evaluation_path_collector,
            replay_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            algorithm=algorithm,
            exploration_env=exploration_env,
            evaluation_env=evaluation_env,
            exploration_path_collector=exploration_path_collector,
            evaluation_path_collector=evaluation_path_collector,
            replay_buffer=replay_buffer,
        )

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_path_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_path_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_path_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_path_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
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

            self._end_epoch(epoch)

    def to(self, device):
        self.algo.set_device(device)
        self.eval_path_collector._policy.to(device)

    def training_mode(self, mode):
        for net in self.algo.get_networks():
            net.train(mode)

class OnPolicyAlgorithm(Trainer):
    def __init__(
            self,
            algorithm,
            exploration_env,
            evaluation_env,
            exploration_path_collector,
            evaluation_path_collector,
            rollout_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1
    ):
        super().__init__(
            algorithm=algorithm,
            exploration_env=exploration_env,
            evaluation_env=evaluation_env,
            exploration_path_collector=exploration_path_collector,
            evaluation_path_collector=evaluation_path_collector,
            replay_buffer=rollout_buffer,
        )

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
    
    def _train(self):
        
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_path_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_path_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)
                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.algo.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def to(self, device):
        self.algo.set_device(device)
        self.eval_path_collector._policy.to(device)

    def training_mode(self, mode):
        for net in self.algo.get_networks():
            net.train(mode)