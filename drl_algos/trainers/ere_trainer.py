import collections
from collections import OrderedDict

import numpy as np
import gtimer as gt

from drl_algos.trainers import Trainer
from drl_algos.data.logging import logger
from drl_algos import utils, eval_util


class ERETrainer(Trainer):
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
            n,
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

        self.n = n

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
                for k in range(1, self.num_trains_per_train_loop+1):
                    train_data = self.replay_buffer.ere_batch(
                        self.batch_size,
                        self.n,
                        k,
                        self.num_trains_per_train_loop+1
                    )
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
