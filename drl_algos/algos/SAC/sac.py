from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn
import gtimer as gt

from drl_algos.algos import Algorithm
from drl_algos import utils


SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss',
)


class SAC(Algorithm):
    """Class defining how to train a Soft Actor-Critic."""

    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        """Initialises SAC algorithm."""
        super().__init__()

        # Networks
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        # Automatic entropy tuning
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            # Create log_alpha variable and optimiser
            self.log_alpha = torch.zeros(1, requires_grad=True, device=qf1.device)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        # Set up optimisers
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        # Setup loss functions
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        # Hyperparameters
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.reward_scale = reward_scale

        # Stats
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        self._num_train_steps = 0

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

    def train(self, batch):
        self._num_train_steps += 1
        batch = utils.to_tensor_batch(batch, self.qf1.device)
        self.train_on_batch(batch)

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num train calls', self._num_train_steps),
        ])
        stats.update(self.eval_statistics)
        return stats

    def train_on_batch(self, batch):
        gt.blank_stamp()
        losses, eval_stats = self.compute_loss(batch,
            skip_statistics=not self._need_to_update_eval_statistics
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = eval_stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    def compute_loss(self, batch, skip_statistics=False):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(
                qf1_loss.to('cpu').detach().numpy()
            )
            eval_statistics['QF2 Loss'] = np.mean(
                qf2_loss.to('cpu').detach().numpy()
            )
            eval_statistics['Policy Loss'] = np.mean(
                policy_loss.to('cpu').detach().numpy()
            )
            eval_statistics.update(utils.create_stats_ordered_dict(
                'Q1 Predictions',
                q1_pred.to('cpu').detach().numpy(),
            ))
            eval_statistics.update(utils.create_stats_ordered_dict(
                'Q2 Predictions',
                q2_pred.to('cpu').detach().numpy(),
            ))
            eval_statistics.update(utils.create_stats_ordered_dict(
                'Q Targets',
                q_target.to('cpu').detach().numpy(),
            ))
            eval_statistics.update(utils.create_stats_ordered_dict(
                'Log Pis',
                log_pi.to('cpu').detach().numpy(),
            ))
            policy_statistics = utils.add_prefix(
                                    dist.get_diagnostics(),
                                    "policy/"
                                )
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )

        return loss, eval_statistics

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            utils.soft_update(self.qf1, self.target_qf1, self.soft_target_tau)
            utils.soft_update(self.qf2, self.target_qf2, self.soft_target_tau)

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def set_device(self, device):
        # Move networks to device
        for network in self.get_networks():
            network.to(device)

        # Copy log_alpha to device then create new optimizer
        if self.use_automatic_entropy_tuning:
            self.log_alpha = self.log_alpha.detach().clone().to(device).requires_grad_(True)
            self.alpha_optimizer = type(self.alpha_optimizer)(
                [self.log_alpha],
                lr=self.alpha_optimizer.param_groups[0]['lr'],
            )

    def get_networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
