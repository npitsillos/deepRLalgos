import collections
from collections import OrderedDict

import numpy as np
import gtimer as gt
import utils

from poo import logger
import eval_util

class BatchRLAlgorithm(object):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
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
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
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
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def to(self, device):
        for net in self.trainer.get_networks():
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.get_networks():
            net.train(mode)

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
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
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
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
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
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

    def get_generic_path_information(paths, stat_prefix=''):
        """
        Get an OrderedDict with a bunch of statistic names and values.
        """
        statistics = OrderedDict()
        returns = [sum(path["rewards"]) for path in paths]

        rewards = np.vstack([path["rewards"] for path in paths])
        statistics.update(utils.create_stats_ordered_dict('Rewards', rewards,
                                                    stat_prefix=stat_prefix))
        statistics.update(utils.create_stats_ordered_dict('Returns', returns,
                                                    stat_prefix=stat_prefix))
        actions = [path["actions"] for path in paths]
        if len(actions[0].shape) == 1:
            actions = np.hstack([path["actions"] for path in paths])
        else:
            actions = np.vstack([path["actions"] for path in paths])
        statistics.update(utils.create_stats_ordered_dict(
            'Actions', actions, stat_prefix=stat_prefix
        ))
        statistics['Num Paths'] = len(paths)
        statistics[stat_prefix + 'Average Returns'] = get_average_returns(paths)

        for info_key in ['env_infos', 'agent_infos']:
            if info_key in paths[0]:
                all_env_infos = [
                    ppp.list_of_dicts__to__dict_of_lists(p[info_key])
                    for p in paths
                ]
                for k in all_env_infos[0].keys():
                    final_ks = np.array([info[k][-1] for info in all_env_infos])
                    first_ks = np.array([info[k][0] for info in all_env_infos])
                    all_ks = np.concatenate([info[k] for info in all_env_infos])
                    statistics.update(utils.create_stats_ordered_dict(
                        stat_prefix + k,
                        final_ks,
                        stat_prefix='{}/final/'.format(info_key),
                    ))
                    statistics.update(utils.create_stats_ordered_dict(
                        stat_prefix + k,
                        first_ks,
                        stat_prefix='{}/initial/'.format(info_key),
                    ))
                    statistics.update(utils.create_stats_ordered_dict(
                        stat_prefix + k,
                        all_ks,
                        stat_prefix='{}/'.format(info_key),
                    ))

        return statistics

    def get_average_returns(self, paths):
        returns = [sum(path["rewards"]) for path in paths]
        return np.mean(returns)

    def list_of_dicts__to__dict_of_lists(self, lst):
        """
        ```
        x = [
            {'foo': 3, 'bar': 1},
            {'foo': 4, 'bar': 2},
            {'foo': 5, 'bar': 3},
        ]
        ppp.list_of_dicts__to__dict_of_lists(x)
        # Output:
        # {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
        ```
        """
        if len(lst) == 0:
            return {}
        keys = lst[0].keys()
        output_dict = collections.defaultdict(list)
        for d in lst:
            assert set(d.keys()) == set(keys), (d.keys(), keys)
            for k in keys:
                output_dict[k].append(d[k])
        return output_dict

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
