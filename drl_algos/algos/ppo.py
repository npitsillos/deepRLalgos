from collections import OrderedDict, namedtuple
from typing import Tuple
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import gtimer as gt

from drl_algos import utils
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
            clip_range=0.2,
            optim_eps=1e-5,

            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=False,
            target_entropy=None,
    ):
        """Initialises SAC algorithm."""
        super().__init__()

        # Networks
        self.policy = policy
        self.critic = critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_iters = n_iters
        self.clip_range = clip_range
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
            eps=optim_eps,
        )
        self.critic_optimizer = optimizer_class(
            self.critic.parameters(),
            lr=critic_lr,
            eps=optim_eps,
        )

        # Stats
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        self._num_train_steps = 0
        self.policy_losses = []
        self.critic_losses = []
        self.approx_kl = []
        self.entropy_losses = []
        self.clip_fractions = []

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

    def train(self, batches):
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
                losses = self.compute_loss(batch)

                self.policy_optimizer.zero_grad()
                losses.policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                losses.critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()
        eval_statistics = OrderedDict()
        eval_statistics["critic_loss"] = np.mean(self.critic_losses)
        eval_statistics["policy_loss"] = np.mean(self.policy_losses)
        eval_statistics["approx_kl"] = np.mean(self.approx_kl)
        eval_statistics["clip_fraction"] = np.mean(self.clip_fractions)
        eval_statistics["entropy"] = np.mean(self.entropy_losses)
        self.eval_statistics = eval_statistics
        gt.stamp('ppo training', unique=False)

    def compute_loss(self, batch, skip_statistics=False):
        obs = batch['observations']
        actions = batch['actions']
        returns = batch["returns"]
        advantages = batch["advantages"]
        td_target = batch["td_target"]
        old_log_prob = batch["log_probs"]
        curr_dist = self.policy(obs)
        
        if actions[0].sum() == 1:
            # hack to know whether discrete
            actions = torch.argmax(actions, dim=1, keepdim=True).squeeze()
        # old_log_prob = old_dist.log_prob(actions)
        curr_log_prob = curr_dist.log_prob(actions)
        ratios = torch.exp(curr_log_prob - old_log_prob.detach())
        surr_one = ratios * advantages
        surr_two = torch.clamp(ratios, 1-self.clip_range, 1+self.clip_range) * advantages

        entropy = -torch.mean(curr_dist.entropy())
        policy_loss = -torch.min(surr_one, surr_two).mean()

        critic_loss = F.mse_loss(returns.detach(), self.critic(obs).squeeze())
        """
        Save some statistics for eval
        """
        self.critic_losses.append(np.mean(
            critic_loss.to('cpu').detach().numpy()
        ))
        self.policy_losses.append(np.mean(
            policy_loss.to('cpu').detach().numpy()
        ))

        self.approx_kl.append(np.mean((old_log_prob - curr_log_prob).to('cpu').detach().numpy()))
        
        self.clip_fractions.append(torch.mean((torch.abs(ratios - 1) > self.clip_range).float()).item())
        self.entropy_losses.append(entropy.detach().cpu().numpy())

        loss = PPOLosses(
            policy_loss=policy_loss,
            critic_loss=critic_loss
        )

        return loss

    def end_epoch(self, epoch):
        self.critic_losses = []
        self.policy_losses = []
        self.approx_kl = []
        self.clip_fractions = []
        self.entropy_losses = []
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
            critic=self.critic
        )

    def _compute_td_target_and_advantages(self, batches):
        batches_td_adv = []
        for batch in batches:
            batch = utils.to_tensor_batch(batch, self.policy.device)
            rewards = batch["rewards"]
            obs = batch["observations"]
            next_obs = batch["next_observations"]
            terminals = batch["terminals"]
            state_values = self.critic(obs).squeeze()
            next_state_values = self.critic(next_obs).squeeze()
            td_target = rewards.squeeze() + self.gamma * next_state_values * (1 - terminals.squeeze())
            delta = td_target - state_values
            delta = utils.to_numpy(delta)

            advantages = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.gae_lambda * advantage + item
                advantages.append(advantage)
            advantages.reverse()
            advantages = np.array(advantages)

            advantages = utils.to_tensor(advantages, self.policy.device)
            
            returns = advantages + state_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            

            dist = self.policy(obs)
            if batch["actions"][0].sum() == 1:
            # hack to know whether discrete
                actions = torch.argmax(batch["actions"], dim=1, keepdim=True).squeeze()
            log_probs = dist.log_prob(actions)
            batch.update(advantages=advantages, returns=returns, td_target=td_target, log_probs=log_probs)

            batches_td_adv.append(batch)
        return batches_td_adv