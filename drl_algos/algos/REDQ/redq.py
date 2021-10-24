from collections import OrderedDict, namedtuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn
import gtimer as gt

from drl_algos.algos import Algorithm
from drl_algos import utils

"""
Compared to SAC, this algorithm has a much higher compute complexity but it does
seem to be much more stable and seems to learn faster. On simple problems
performance is likely to be similar, but on harder problems its likely to
outshine SAC.

The original REDQ paper built this algorithm specifically to target the sample
efficiency of model based methods. They used a sample:update ratio of 1:20,
which on plain SAC is unlikely to succeed. The higher you push the sample
efficiency the more likely you are to run into stability issues.

In particular, what this implementation aims to address is the overestimation
bias in Q-learning.
"""


REDQLosses = namedtuple(
    'REDQLosses',
    'policy_loss q_loss alpha_loss',
)


class REDQ(Algorithm):
    """Class defining how to train a Soft Actor-Critic."""

    def __init__(
            self,
            env,
            policy,
            critic_fn,
            ensemble_size=10,
            minimisation=2,

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
        """Initialises REDQ algorithm."""
        super().__init__()

        self.ensemble_size = ensemble_size
        self.minimisation = minimisation

        # Networks
        self.policy = policy
        self.critics = []
        self.target_critics = []
        for i in range(ensemble_size):
            self.critics.append(critic_fn())
            self.target_critics.append(critic_fn())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

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
            self.log_alpha = torch.zeros(1, requires_grad=True, device=policy.device)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        # Set up optimisers
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.critic_optims = []
        for i in range(ensemble_size):
            self.critic_optims.append(optimizer_class(
                self.critics[i].parameters(),
                lr=qf_lr,
                weight_decay=weight_decay
            ))

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
        batch = utils.to_tensor_batch(batch, self.policy.device)
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

        for critic_optim in self.critic_optims:
            critic_optim.zero_grad()
        losses.q_loss.backward()
        for i in range(self.ensemble_size):
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.grad_clip)
            self.critic_optims[i].step()

        self.try_update_target_networks()
        gt.stamp(' training', unique=False)

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

        q_preds_new_action = []
        for i in range(self.ensemble_size):
            q_preds_new_action.append(self.critics[i](obs, new_obs_actions))
        q_preds_new_action = torch.cat(q_preds_new_action, 1)
        avg_q = torch.mean(q_preds_new_action, dim=1, keepdim=True)
        policy_loss = (alpha*log_pi - avg_q).mean()

        """
        QF Loss
        """
        q_preds = []
        for i in range(self.ensemble_size):
            q_preds.append(self.critics[i](obs, actions))
        q_preds = torch.cat(q_preds, dim=1)

        with torch.no_grad():
            next_dist = self.policy(next_obs)
            new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
            new_log_pi = new_log_pi.unsqueeze(-1)
            critic_idxs = torch.multinomial(
                torch.ones(self.ensemble_size), self.minimisation)
            q_preds_next_obs = []
            for idx in critic_idxs:
                q_preds_next_obs.append(
                    self.target_critics[idx](next_obs, new_next_actions))
            q_preds_next_obs = torch.cat(q_preds_next_obs, 1)
            min_q, _ = torch.min(q_preds_next_obs, dim=1, keepdim=True)
            target_q_values = min_q - alpha * new_log_pi
            q_target = self.reward_scale*rewards + (1.-terminals)*self.discount*target_q_values

        # Torch complains about this, but as far as I can tell its okay
        q_loss = self.qf_criterion(q_preds, q_target.detach()) * self.ensemble_size

        """
        Save some statistics for eval
        """
        # This doesn't seem to have a big impact on training time, over the
        # course of 10000 updates it seemed to add about 5 seconds, from 78sec
        # to 83sec, including time taken in get_diagnositcs()
        self.eval_statistics['Q Loss'].append(
            utils.to_numpy(q_loss)
        )
        self.eval_statistics['Policy Loss'].append(
            utils.to_numpy(policy_loss)
        )
        self.eval_statistics['Q Predictions'].append(
            utils.to_numpy(q_preds)
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

        loss = REDQLosses(
            policy_loss=policy_loss,
            q_loss=q_loss,
            alpha_loss=alpha_loss,
        )
        return loss

    def try_update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            for i in range(self.ensemble_size):
                utils.soft_update(
                    self.critics[i], self.target_critics[i],
                    self.soft_target_tau)

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
        networks = [self.policy]
        networks.extend(self.critics)
        networks.extend(self.target_critics)
        return networks

    def get_optimizers(self):
        optims = [self.alpha_optimizer, self.policy_optimizer]
        optims.extend(self.critic_optims)
        return optims

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            critics=self.critics,
            target_critics=self.target_critics,
        )

    def _reset_eval_stats(self):
        self.eval_statistics = {
            'Q Loss': [],
            'Policy Loss': [],
            'Q Predictions': [],
            'Q Targets': [],
            'Log Pis': [],
        }
        if self.use_automatic_entropy_tuning:
            self.eval_statistics['Alpha'] = []
            self.eval_statistics['Alpha Loss'] = []
