from collections import OrderedDict, namedtuple
from typing import Tuple
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import gtimer as gt

from drl_algos.utils import utils
from drl_algos.algos.algorithm import Algorithm
import torch.nn.functional as F

PPOLosses = namedtuple(
    'PPOLosses',
    'policy_loss critic_loss',
)


class PPO(Algorithm):
    """Class defining how to train a Soft Actor-Critic."""

    def __init__(
            self,
            env,
            policy,
            critic,

            policy_lr=1e-3,
            critic_lr=1e-3,
            optimizer_class=optim.Adam,
            n_iters=10,
            gamma=0.99,
            gae_lambda=0.95,

            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        """Initialises SAC algorithm."""
        super().__init__()

        # Networks
        self.policy = policy
        self.old_policy = deepcopy(self.policy)
        self.critic = critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_iters = n_iters
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
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.policy.device)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        # Set up optimisers
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.critic_optimizer = optimizer_class(
            self.critic.parameters(),
            lr=critic_lr,
        )

        # Stats
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        self._num_train_steps = 0

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

    def train(self, batches):
        self.old_policy.load_state_dict(self.policy.state_dict())
        batches = self._compute_td_target_and_advantages(batches)
        self.train_on_batch(batches)

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num updates', self._n_train_steps_total),
        ])
        stats.update(self.eval_statistics)
        return stats

    def train_on_batch(self, batches):
        gt.blank_stamp()
        for i in range(self.n_iters):
            self._n_train_steps_total += 1
            for batch in batches:
                batch = utils.to_tensor_batch(batch, self.policy.device)
                losses, eval_stats = self.compute_loss(batch,
                    skip_statistics=not self._need_to_update_eval_statistics
                )

                self.policy_optimizer.zero_grad()
                losses.policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                losses.critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                if self._need_to_update_eval_statistics:
                    self.eval_statistics = eval_stats
                    # Compute statistics using only one batch per epoch
                    self._need_to_update_eval_statistics = False
                gt.stamp('ppo training', unique=False)

    def compute_loss(self, batch, skip_statistics=False):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        td_target = batch["td_target"]
        advantages = batch["advantages"]
        old_dist = self.old_policy(obs)
        dist = self.policy(obs)
        if not self.policy.is_continuous:
            actions = torch.argmax(actions, dim=1, keepdim=True)
        old_log_prob = old_dist.log_prob(actions)
        curr_log_prob = dist.log_prob(actions)
        ratios = torch.exp(curr_log_prob - old_log_prob)
        surr_one = ratios * advantages
        surr_two = torch.clamp(ratios, 0.8, 1.2) * advantages

        policy_loss = -torch.min(surr_one, surr_two).mean()
        state_values = self.critic(obs)
        critic_loss = F.mse_loss(state_values, td_target)
        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['Critic Loss'] = np.mean(
                critic_loss.to('cpu').detach().numpy()
            )
            eval_statistics['Policy Loss'] = np.mean(
                policy_loss.to('cpu').detach().numpy()
            )
            policy_statistics = utils.add_prefix(
                                    dist.get_diagnostics(),
                                    "policy/"
                                )
            eval_statistics.update(policy_statistics)

        loss = PPOLosses(
            policy_loss=policy_loss,
            critic_loss=critic_loss
        )

        return loss, eval_statistics

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
            self.critic,
            self.old_policy,
        ]

    def get_optimizers(self):
        return [
            self.alpha_optimizer,
            self.critic_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            old_policy=self.old_policy,
            critic=self.critic
        )

    def _compute_td_target_and_advantages(self, batches):
        batches_td_adv = []
        for batch in batches:
            rewards = batch["rewards"]
            obs = batch["observations"]
            next_obs = batch["next_observations"]
            terminals = batch["terminals"]
            state_values = self.critic(obs).squeeze()
            next_state_values = self.critic(next_obs).squeeze()
            td_target = rewards + self.gamma * utils.to_numpy(next_state_values) * terminals
            delta = td_target - utils.to_numpy(state_values)
            delta = utils.to_numpy(delta)

            advantages = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.gae_lambda * advantage + item[0]
                advantages.append(advantage)
            advantages.reverse()
            advantages = np.array(advantages)
            advantages = advantages - advantages.mean() / advantages.std() + 1e-8
            
            batch.update(advantages=advantages, td_target=td_target)
            batches_td_adv.append(batch)
        return batches_td_adv