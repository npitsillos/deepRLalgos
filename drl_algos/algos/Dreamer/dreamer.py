from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn
import gtimer as gt

from drl_algos.algos import Algorithm
from drl_algos import utils


"""
Uses copy of critic for targets to stabilise learning
    - updated every 100 gradient steps

For continuous control I'm not sure if can do straight through gradients when
sampling actions

To start off, lets try and follow the paper as closely as possible and implement
it for discrete control. Once I know that works, I can move onto continuous
control which may require reparameterised gradients
"""

BehaviourLosses = namedtuple(
    'BehaviourLosses',
    'policy_loss v_loss',
)


class DreamerBehaviour(Algorithm):
    """Class defining how to train Dreamer Behaviours.

    Args:
        discount (float): normal discount for Rl
        lambda_target (float): for the multi-step target
        reinforce_mixing (float): how much weight to give reinforce gradients.
                                  for discrete (Atari) 1 is a good value, for
                                  continuous control dyanamics is better so
                                  use 1.
        """

    def __init__(
            self,
            policy,
            vf,
            target_vf,

            discount=0.995,
            lambda_target=0.95,
            reinforce_mixing=1.0,
            actor_entropy_scale=1e-3,
            actor_lr=4e-5,
            critic_lr=1e-4,
            target_update_period=100,
            grad_clip=100,

            optimizer_class=optim.Adam,
    ):
        """Initialises SAC algorithm."""
        super().__init__()

        # Networks
        self.policy = policy
        self.vf = vf
        self.target_vf = target_vf

        self.discount = discount
        self.lambda_target = lambda_target
        self.reinforce_mixing = reinforce_mixing
        self.actor_entropy_scale = actor_entropy_scale
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.target_update_period = target_update_period
        self.grad_clip = grad_clip

        # Set up optimisers
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=actor_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=critic_lr,
        )

        # Setup loss functions
        self.vf_criterion = nn.MSELoss()

        # Stats
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        self._num_train_steps = 0

    def train(self, states, actions, log_pis, rewards, gammas):
        self._num_train_steps += 1

        gt.blank_stamp()
        losses, eval_stats = self.compute_loss(
            states,
            actions,
            log_pis,
            rewards,
            gammas,
            skip_statistics=not self._need_to_update_eval_statistics
        )

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        self.vf_optimizer.zero_grad()
        losses.v_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.vf.parameters(), self.grad_clip)
        self.vf_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = eval_stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('dreamer behaviour training', unique=False)

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num train calls', self._num_train_steps),
        ])
        stats.update(self.eval_statistics)
        return stats

    def compute_loss(self, states, actions, log_pis, rewards, gammas,
                     skip_statistics=False):
        loss_weights = torch.cumprod(gammas, 1)

        # Iterate backwards through sequence
        old_v_target = None
        v_losses = []
        policy_losses = []
        sequence_len = len(rewards[0])
        for i in range(sequence_len-2, -1, -1):
            state = states[:,i]
            next_state = states[:,i+1]
            action = actions[:,i]
            next_action = actions[:,i+1]
            log_pi = log_pis[:,i]
            next_log_pi = log_pis[:,i+1]
            reward = rewards[:,i+1]
            gamma = gammas[:,i+1]

            """
            Critic
            """
            # Predict value for state
            v_pred = self.vf(state)

            # Calculate n-step target
            with torch.no_grad():
                target_v_value = self.target_vf(next_state)

                if old_v_target is not None:
                    target_v_value = ((1-self.lambda_target)*target_v_value
                                       + self.lambda_target*old_v_target)
                v_target = reward + gamma*self.discount*target_v_value
                old_v_target = v_target

            # Calculate critic loss while softly accounting for episode end
            v_loss = loss_weights[:,i] * self.vf_criterion(v_pred, v_target.detach())
            v_losses.append(v_loss)

            """
            Policy
            """
            dist = self.policy(state)

            reinforce = self.reinforce_mixing * dist.log_prob(action) * (v_target - v_pred).detach()
            dynamics = (1 - self.reinforce_mixing) * v_target
            entropy = self.actor_entropy_scale * dist.entropy()

            policy_loss = loss_weights[:,i] * (-reinforce - dynamics - entropy)
            policy_losses.append(policy_loss)

        # Divide loss by number of valid training samples per sequence
        valid_sequence_len = (sequence_len - 1) * loss_weights.mean(1)
        v_losses = torch.cat(v_losses, dim=1).mean(1)
        policy_losses = torch.cat(policy_losses, dim=1).mean(1)
        policy_loss = (policy_losses / valid_sequence_len).mean()
        v_loss = (v_losses / valid_sequence_len).mean()

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['value Loss'] = np.mean(
                v_loss.to('cpu').detach().numpy()
            )
            eval_statistics['Policy Loss'] = np.mean(
                policy_loss.to('cpu').detach().numpy()
            )
            eval_statistics.update(utils.create_stats_ordered_dict(
                'Value Predictions',
                v_pred.to('cpu').detach().numpy(),
            ))
            eval_statistics.update(utils.create_stats_ordered_dict(
                'Value Targets',
                v_target.to('cpu').detach().numpy(),
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

        loss = BehaviourLosses(
            policy_loss=policy_loss,
            v_loss=v_loss,
        )

        return loss, eval_statistics

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            utils.soft_update(self.vf, self.target_vf, 1)

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def set_device(self, device):
        # Move networks to device
        for network in self.get_networks():
            network.to(device)

    def get_networks(self):
        return [
            self.policy,
            self.vf,
            self.target_vf,
        ]

    def get_optimizers(self):
        return [
            self.vf_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            vf=self.vf,
            target_vf=self.target_vf,
        )
