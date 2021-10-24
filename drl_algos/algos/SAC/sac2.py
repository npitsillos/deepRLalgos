from collections import OrderedDict, namedtuple

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

            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class=optim.Adam,
            weight_decay=0,

            soft_target_tau=5e-3,
            target_update_period=1,
            grad_clip=None,

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
            weight_decay=weight_decay
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
            weight_decay=weight_decay
        )

        # Setup loss functions
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        # Hyperparameters
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.reward_scale = reward_scale
        self.grad_clip = grad_clip

        # Stats
        self._num_train_steps = 0
        self._reset_eval_stats()

    def train(self, batch):
        self._num_train_steps += 1
        batch = utils.to_tensor_batch(batch, self.qf1.device)
        self.train_on_batch(batch)

    def get_diagnostics(self):
        stats = {'num train calls': self._num_train_steps}
        for key, value in self.eval_statistics.items():
            stats.update(utils.get_stats(key, value))
        return stats

    def train_on_batch(self, batch):
        gt.blank_stamp()
        losses = self.compute_loss(batch)

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.log_alpha, self.grad_clip)
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.qf1.parameters(), self.grad_clip)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.qf2.parameters(), self.grad_clip)
        self.qf2_optimizer.step()

        self.try_update_target_networks()
        gt.stamp('sac training', unique=False)

    def compute_loss(self, batch):
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

        with torch.no_grad():
            next_dist = self.policy(next_obs)
            new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
            new_log_pi = new_log_pi.unsqueeze(-1)
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            ) - alpha * new_log_pi
            # i.e., reward for action plus future reward unless next_state is done
            q_target = self.reward_scale*rewards + (1.-terminals)*self.discount*target_q_values

        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        # This doesn't seem to have a big impact on training time, over the
        # course of 10000 updates it seemed to add about 5 seconds, from 78sec
        # to 83sec, including time taken in get_diagnositcs()
        self.eval_statistics['QF1 Loss'].append(
            utils.to_numpy(qf1_loss)
        )
        self.eval_statistics['QF2 Loss'].append(
            utils.to_numpy(qf2_loss)
        )
        self.eval_statistics['Policy Loss'].append(
            utils.to_numpy(policy_loss)
        )
        self.eval_statistics['Q1 Predictions'].append(
            utils.to_numpy(q1_pred)
        )
        self.eval_statistics['Q2 Predictions'].append(
            utils.to_numpy(q2_pred)
        )
        self.eval_statistics['Q Targets'].append(
            utils.to_numpy(q_target)
        )
        self.eval_statistics['Log Pis'].append(
            utils.to_numpy(log_pi)
        )
        if self.use_automatic_entropy_tuning:
            self.eval_statistics['Alpha'].append(alpha.item())
            self.eval_statistics['Alpha Loss'].append(alpha_loss.item())
        dist_diag = utils.add_prefix(dist.get_metrics(), "policy/")
        for key, value in dist_diag.items():
            if key in self.eval_statistics:
                self.eval_statistics[key].append(value)
            else:
                self.eval_statistics[key] = [value]

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )
        return loss

    def try_update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            utils.soft_update(self.qf1, self.target_qf1, self.soft_target_tau)
            utils.soft_update(self.qf2, self.target_qf2, self.soft_target_tau)

    def end_epoch(self, epoch):
        self._reset_eval_stats()

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

    def _reset_eval_stats(self):
        self.eval_statistics = {
            'QF1 Loss': [],
            'QF2 Loss': [],
            'Policy Loss': [],
            'Q1 Predictions': [],
            'Q2 Predictions': [],
            'Q Targets': [],
            'Log Pis': [],
        }
        if self.use_automatic_entropy_tuning:
            self.eval_statistics['Alpha'] = []
            self.eval_statistics['Alpha Loss'] = []
