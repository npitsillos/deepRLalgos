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

class SACDreamer(Algorithm):
    """Class defining how to train a Soft Actor-Critic."""

    """
    TODO - Add evaluation statistics
         - Check data is right shapes
         - Check actions and log pis all have gradients attached except for
           the very first action (off-policy)
         - Remove discount from dreamer and add back to sac
    """


    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=.99,
            lamb=0.95,
            grad_clip=100,
            reward_scale=1.0,
            gamma_target_scaling=False,

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
        self.qf_criterion = nn.MSELoss(reduction='none') # returns square errors

        # Hyperparameters
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.grad_clip = grad_clip
        self.lamb = lamb
        self.reward_scale = reward_scale
        self.discount = discount
        self.gamma_target_scaling = gamma_target_scaling

        # Stats
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        self._num_train_steps = 0

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

    def train(self, batch):
        self._num_train_steps += 1
        # batch = utils.to_tensor_batch(batch, self.qf1.device)
        self.train_on_batch(batch)

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num train calls', self._num_train_steps),
        ])
        stats.update(self.eval_statistics)
        return stats

    def train_on_batch(self, batch):
        """
        TODO - Check if gradient clipping is necessary for alpha_loss or needs tuned
             - Check if gradient clipping is necessary for SAC or needs tuned
        """
        gt.blank_stamp()
        losses, eval_stats = self.compute_loss(batch,
            skip_statistics=not self._need_to_update_eval_statistics
        )

        # if self.use_automatic_entropy_tuning:
        #     self.alpha_optimizer.zero_grad()
        #     losses.alpha_loss.backward()
        #     nn.utils.clip_grad_norm_(self.log_alpha, self.grad_clip)
        #     self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        nn.utils.clip_grad_norm_(self.qf1.parameters(), self.grad_clip)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        nn.utils.clip_grad_norm_(self.qf2.parameters(), self.grad_clip)
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = eval_stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    def compute_loss(self, batch, skip_statistics=False):
        """
        Notes
            - below implementation is using the TD-k trick from MVE. Only using
            the final target value for the off-policy action provides poor
            performance because the agent is only trained on environment data
            which doesn't match the distribution of data in dreaming.
        """
        states = batch['states']
        actions = batch['actions']
        log_pis = batch['log_pis']
        rewards = batch['rewards']
        gammas = batch['gammas']

        # Cumulative product of gamma for each sequence, [a,b,c] = [a,a*b,a*b*c]
        loss_weights = torch.cumprod(gammas, 1)

        """todo
            - track data for eval stats
            - tracking losses would enable smarter loss scaling
              - if downweighting losses by gamma, then should also take mean
              gamma for samples into account when averaging by sequence length
                - needs to be done for each sequence individually
        """
        # Iterate backwards through sequence
        old_q_target = None
        qf1_losses = []
        qf2_losses = []
        policy_losses = []
        sequence_len = len(rewards[0])
        for i in range(sequence_len-2, -1, -1):
            state = states[:,i]
            next_state = states[:,i+1]
            action = actions[:,i]
            log_pi = log_pis[:,i]
            reward = rewards[:,i+1]
            gamma = gammas[:,i+1]
            next_action = actions[:,i+1]
            next_log_pi = log_pis[:,i+1]

            if i == 0:
                # Replace off-policy action with on-policy
                dist = self.policy(state)
                new_action, new_log_pi = dist.rsample_and_logprob()
                new_log_pi = new_log_pi.unsqueeze(-1)
            else:
                # Else reuse on-policy action
                new_action = action
                new_log_pi = log_pi

            """
            Entropy
            """
            # Calculate alpha and alpha loss
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (new_log_pi + self.target_entropy).detach()).mean()
                alpha = self.log_alpha.exp().detach()
            else:
                alpha_loss = 0
                alpha = 1

            """
            Critic
            """
            # Predict Q-values for state-action pairs
            q1_pred = self.qf1(state, action.detach())
            q2_pred = self.qf2(state, action.detach())

            # Calculate n-step Q-target
            with torch.no_grad():
                target_q_value = torch.min(
                    self.target_qf1(next_state, next_action.detach()),
                    self.target_qf2(next_state, next_action.detach()),
                ) - alpha * next_log_pi.detach()
                if old_q_target is not None:
                    target_q_value = ((1-self.lamb)*target_q_value
                                       + self.lamb*old_q_target)
                q_target = self.reward_scale*reward + gamma*self.discount*target_q_value
                if self.gamma_target_scaling:
                    # Downweight target q-value by state gamma to softly
                    # account for episode end in the q-target
                    q_target *= gammas[:,i]
                old_q_target = q_target

            # Calculate critic loss
            qf1_loss = loss_weights[:,i] * self.qf_criterion(q1_pred, q_target.detach())
            qf1_losses.append(qf1_loss)
            qf2_loss = loss_weights[:,i] * self.qf_criterion(q2_pred, q_target.detach())
            qf2_losses.append(qf2_loss)

            """
            Policy
            """
            if i == 0:
                # If off-policy experience then calculate q-value for new action
                q_new_action = torch.min(
                    self.qf1(state, new_action),
                    self.qf2(state, new_action),
                )
            else:
                # Else use n-step target
                q_new_action = old_q_target

            # Calculate policy loss
            objective = alpha*new_log_pi - q_new_action
            policy_losses.append(loss_weights[:,i] * objective)

        # Divide loss by number of valid training samples per sequence
        valid_sequence_len = (sequence_len - 1) * loss_weights.mean(1)
        qf1_losses = torch.cat(qf1_losses, dim=1).mean(1)
        qf2_losses = torch.cat(qf2_losses, dim=1).mean(1)
        policy_losses = torch.cat(policy_losses, dim=1).mean(1)
        policy_loss = (policy_losses / valid_sequence_len).mean()
        qf1_loss = (qf1_losses / valid_sequence_len).mean()
        qf2_loss = (qf2_losses / valid_sequence_len).mean()

        """
        Save some statistics for eval
        TODO - Check evaluation still valid
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
        #     eval_statistics.update(utils.create_stats_ordered_dict(
        #         'Q1 Predictions',
        #         q1_pred.to('cpu').detach().numpy(),
        #     ))
        #     eval_statistics.update(utils.create_stats_ordered_dict(
        #         'Q2 Predictions',
        #         q2_pred.to('cpu').detach().numpy(),
        #     ))
        #     eval_statistics.update(utils.create_stats_ordered_dict(
        #         'Q Targets',
        #         q_target.to('cpu').detach().numpy(),
        #     ))
        #     eval_statistics.update(utils.create_stats_ordered_dict(
        #         'Log Pis',
        #         log_pi.to('cpu').detach().numpy(),
        #     ))
        #     policy_statistics = utils.add_prefix(
        #                             dist.get_diagnostics(),
        #                             "policy/"
        #                         )
        #     eval_statistics.update(policy_statistics)
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
