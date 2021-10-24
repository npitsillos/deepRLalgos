from collections import OrderedDict, namedtuple
import os

import gtimer as gt
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as td
import torch.optim as optim

from drl_algos.networks.myrssm import RSSM
from drl_algos.networks import Network, Base, Mlp, Gru, policies, critics
from drl_algos import utils


ActorCriticLosses = namedtuple(
    'ActorCriticLosses',
    'policy_loss critic_loss',
)

class Model(Network):
    """
    TODO
        - check distributions are correctly interpretting event dims etc
            - probably need to add independent
            - looking at the original, it seems like they only wrapped the
            decoder in an independent
        - would be nice to report some sort of normalised loss for the Normal
        distributions
            - with a scale of 1 the best log_prob a dist can achieve is -0.9189
            for each value
            - each with an observation space of 3, the best log_prob is
            -0.9189*3 = -2.7567
    """

    def __init__(
        self,
        latent_size,
        encoder_hidden_sizes,
        forward_hidden_sizes,
        obs_size,
        action_size,
        act_fn=F.elu,
        discount_scale=5.,
        reward_scale=1.,
        obs_scale=1.,
        forward_scale=0.1,
        lr=2e-4,
        adam_eps=1e-5,
        adam_decay=1e-6,
        grad_clip=100,
    ):
        super().__init__()

        # Parameters
        self.latent_size = latent_size
        self.obs_size = obs_size
        self.action_size = action_size
        self.discount_scale = discount_scale
        self.reward_scale = reward_scale
        self.obs_scale = obs_scale
        self.forward_scale = forward_scale
        self.grad_clip = grad_clip

        # Build networks
        # (obs_t + act_t) -> features_t
        self.encoder = Mlp(
            encoder_hidden_sizes + [latent_size],
            obs_size + action_size,
            act_fn
        )
        # (features_t + act_t+1) -> features_t+1
        self.forward_model = Mlp(
            forward_hidden_sizes  + [latent_size],
            latent_size + action_size,
            act_fn
        )
        # (features_t) -> reward_t
        self.reward_decoder = Mlp(
            [1], latent_size, act_fn, False
        )
        # (features_t) -> discount_t
        self.discount_decoder = Mlp(
            [1], latent_size, F.sigmoid
        )
        # (features_t) -> obs_t
        self.obs_decoder = Mlp(
            [obs_size], latent_size, act_fn, False
        )

        # Optimizer
        self.optim = optim.Adam(
            self.parameters(),
            lr=lr,
            eps=adam_eps,
            weight_decay=adam_decay
        )

        # Stats
        self._num_train_steps = 0
        self._reset_eval_stats()

    def observe(
        self,
        obs,
        prev_action=None,
        prev_model_state=None,
        sample=False
    ):
        with torch.no_grad():
            if prev_action is None:
                prev_action = torch.zeros(1, self.action_size).to(
                    self.device
                )
            else:
                prev_action = utils.to_tensor(prev_action, self.device)
                if len(prev_action.shape) == 0:
                    prev_action = F.one_hot(prev_action, self.action_size)
                    prev_action = prev_action.float().unsqueeze(0)

            obs = utils.to_tensor(obs, self.device).unsqueeze(0)
            latent_state = self.encoder(torch.cat((obs, prev_action), 1))
            return latent_state

    def imagine(
        self,
        policy,
        start_states,
        start_terminals,
        horizon,
        discount
    ):
        # Process start states
        state = start_states
        states = [state]
        actions = [torch.zeros_like(policy(state).sample())]
        rewards = [self.reward_decoder(state)]
        discounts = [self.discount_decoder(state)]

        # Imagine forward
        for i in range(horizon):
            action = policy(state.detach()).rsample()
            state = self.forward_model(torch.cat((state, action), 1))
            reward = self.reward_decoder(state)
            discount = self.discount_decoder(state)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            discounts.append(discount)

        # Return imagination sequences
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        discounts = torch.stack(discounts)
        weights = torch.cumprod(
            torch.cat((torch.ones_like(discounts[0:1]), discounts[1:])), 0
        )
        discounts *= discount
        return states, actions, rewards, discounts, weights

    def train(self, batch):
        self._num_train_steps += 1
        batch = utils.to_tensor_batch(batch, self.device)
        return self.train_on_batch(batch)

    def train_on_batch(self, batch):
        gt.blank_stamp()
        loss, states, discounts = self.compute_loss(batch)

        # Backpropogate loss
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optim.step()

        gt.stamp('model training', unique=False)
        return states, discounts

    def compute_loss(self, batch):
        seq_len = batch["observations"].shape[0] + 1
        batch_size = batch["observations"].shape[1]

        # Preprocess data into required sequences
        obs = torch.cat(
            (batch["observations"][0:1], batch["next_observations"])
        )
        dummy_action = torch.zeros(
            1, batch_size, self.action_size
        ).to(self.device)
        actions = torch.cat((dummy_action, batch["actions"]))
        dummy_reward = torch.zeros(1, batch_size, 1).to(self.device)
        rewards = torch.cat((dummy_reward, batch["rewards"]))
        dummy_terminal = torch.zeros(1, batch_size, 1).to(self.device)
        discounts = 1. - torch.cat((dummy_terminal, batch["terminals"]))

        # Encode sequence
        input = torch.flatten(
            torch.cat((obs, actions), dim=2),
            end_dim=1
        )
        latent_state = self.encoder(input)
        latent_state = torch.reshape(latent_state, [seq_len, batch_size, -1])

        # Predict next state
        input = torch.flatten(
            torch.cat((latent_state[:-1], actions[1:]), dim=2),
            end_dim=1
        )
        predicted_next_state = self.forward_model(input)
        groundtruth_next_state = torch.flatten(latent_state[1:], end_dim=1)

        # Predict heads
        latent_state = torch.flatten(latent_state, end_dim=1)
        predicted_reward = self.reward_decoder(latent_state)
        predicted_discount = self.discount_decoder(latent_state)
        predicted_obs = self.obs_decoder(latent_state)

        # Compute losses
        obs = torch.flatten(obs, end_dim=1)
        actions = torch.flatten(actions, end_dim=1)
        rewards = torch.flatten(rewards, end_dim=1)
        discounts = torch.flatten(discounts, end_dim=1)
        forward_loss = F.mse_loss(predicted_next_state, groundtruth_next_state)
        reward_loss = F.mse_loss(predicted_reward, rewards)
        discount_loss = F.mse_loss(predicted_discount, discounts)
        obs_loss = F.mse_loss(predicted_obs, obs)
        model_loss = (
            self.forward_scale * forward_loss
            + self.reward_scale * reward_loss
            + self.discount_scale * discount_loss
            + self.obs_scale * obs_loss
        )

        # Update eval stats
        self.eval_statistics['Forward Loss'].append(
            forward_loss.item()
        )
        self.eval_statistics['Observation Loss'].append(
            obs_loss.item()
        )
        self.eval_statistics['Reward Loss'].append(
            reward_loss.item()
        )
        self.eval_statistics['Discount Loss'].append(
            discount_loss.item()
        )

        return model_loss, latent_state, discounts

    def end_epoch(
        self,
        epoch
    ):
        self._reset_eval_stats()

    def get_latent_state(
        self,
        state
    ):
        return state

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num train calls', self._num_train_steps),
        ])
        for key, value in self.eval_statistics.items():
            stats[key] = np.mean(value)
        return stats

    def _reset_eval_stats(self):
        self.eval_statistics = {
            "Forward Loss": [],
            "Observation Loss": [],
            "Reward Loss": [],
            "Discount Loss": [],
        }

class ActorCritic(Network):

    def __init__(
        self,
        world_model,
        hidden_sizes,
        horizon=15,
        discount=0.99,
        act_fn=F.elu,
        policy_lr=4e-5,
        critic_lr=1e-4,
        grad_clip=100,
        adam_eps=1e-5,
        adam_decay=1e-6,
        slow_update_freq=100,
        lam=.95,
        entropy_scale=2e-3,
    ):
        super().__init__()

        # Parameters
        self.world_model = world_model
        self.horizon = horizon
        self.discount = discount
        self.grad_clip = grad_clip
        self.slow_update_freq = slow_update_freq
        self.lam = lam
        self.entropy_scale = entropy_scale

        # Construct Networks
        self.policy = policies.MlpDiscretePolicy2(
            world_model.latent_size,
            world_model.action_size,
            hidden_sizes,
            act_fn
        )
        self.critic = critics.SacMlpCritic(
            world_model.latent_size,
            1,
            hidden_sizes,
            act_fn
        )
        self.critic_target = critics.SacMlpCritic(
            world_model.latent_size,
            1,
            hidden_sizes,
            act_fn
        )

        # Optimizer
        self.policy_optim = optim.Adam(
            self.policy.parameters(),
            lr=policy_lr,
            eps=adam_eps,
            weight_decay=adam_decay
        )
        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            eps=adam_eps,
            weight_decay=adam_decay
        )

        # Stats
        self._num_train_steps = 0
        self._reset_eval_stats()

    def train(self, start_states, start_terminals):
        self._num_train_steps += 1
        gt.blank_stamp()

        # Get imagination sequences and losses
        sequences = self.world_model.imagine(
            self.policy, start_states, start_terminals, self.horizon,
            self.discount
        )
        losses = self.compute_loss(*sequences)

        # Backpropogate policy loss
        self.policy_optim.zero_grad()
        losses.policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optim.step()

        # Backpropogate critic loss
        self.critic_optim.zero_grad()
        losses.critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()

        # Update target critic
        self.try_update_target_networks()
        gt.stamp('actor critic training', unique=False)

    def compute_loss(self, states, actions, rewards, discounts, weights):
        # rewards = torch.ones_like(rewards).to(self.device)

        targets = self.target(states, rewards, discounts)
        actor_loss = self.actor_loss(
            states, actions, rewards, weights, targets
        )
        critic_loss = self.critic_loss(states, weights, targets)

        self.eval_statistics['Imagined Returns'].append(
            torch.sum(targets*weights[:-1], dim=0).mean().item()
        )

        return ActorCriticLosses(actor_loss, critic_loss)

    def actor_loss(self, states, actions, rewards, weights, targets):
        """Confirmed same as original
            - tested by manually constructing distributions with known values
        """
        policy_dist = self.policy(states[:-2].detach())
        baseline_loc = self.critic_target(states[:-2])
        # TODO - probably don't need to actually construct the Normal
        #      - .mean just returns the loc
        baseline_dist = td.Normal(baseline_loc, 1)
        baseline = baseline_dist.mean
        advantage = (targets[1:] - baseline).detach()
        log_prob = policy_dist.log_prob(actions[1:-1])
        objective = log_prob * advantage.squeeze(2)
        entropy = policy_dist.entropy()
        objective += entropy * self.entropy_scale
        actor_loss = -(weights[:-2].squeeze(2).detach() * objective).mean()

        self.eval_statistics['Actor Log Prob'].append(
            log_prob.mean().item()
        )
        self.eval_statistics['Actor Entropy'].append(
            entropy.mean().item()
        )
        self.eval_statistics['Actor Loss'].append(
            actor_loss.item()
        )
        self.eval_statistics['Imagined Advantage'].append(
            advantage.mean().item()
        )

        return actor_loss

    def critic_loss(self, states, weights, targets):
        """ NOTES/TODO
            - in the original implementation it doesn't look like they detach
            the gradients from the states
            - confirmed same as original except for detach
                - tested by manually constructing distributions with known values
        """
        critic_loc = self.critic(states[:-1].detach())
        critic_dist = td.Normal(critic_loc, 1)
        critic_loss = -(
            critic_dist.log_prob(targets.detach())
            * weights[:-1].detach()
        ).mean()

        self.eval_statistics['Critic Prediction'].append(
            critic_dist.mean.mean().item()
        )
        self.eval_statistics['Critic Loss'].append(
            critic_loss.item()
        )

        return critic_loss

    def target(self, states, rewards, discounts):
        """Confirmed same as original except its dimensions are 3 not 2"""
        value_loc = self.critic_target(states)
        # TODO - probably don't need to actually construct the Normal
        #      - .mean just returns the loc
        value_dist = td.Normal(value_loc, 1)
        values = value_dist.mean

        targets = []
        prev_target = None
        for i in range(len(values)-2, -1, -1):
            if prev_target == None:
                target = rewards[i] + discounts[i] * values[i+1]
            else:
                target = rewards[i] + discounts[i] * (
                    (1.-self.lam) * values[i+1] + self.lam * prev_target
                )
            targets.append(target)
            prev_target = target
        targets = torch.stack(targets).flip(0)

        self.eval_statistics['Critic Target Prediction'].append(
            value_dist.mean.mean().item()
        )
        self.eval_statistics['Critic Target'].append(
            targets.mean().item()
        )

        return targets

    def end_epoch(self, epoch):
        self._reset_eval_stats()

    def try_update_target_networks(self):
        if self._num_train_steps % self.slow_update_freq == 0:
            utils.soft_update(self.critic, self.critic_target, 1)

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num train calls', self._num_train_steps),
        ])
        for key, value in self.eval_statistics.items():
            stats[key] = np.mean(value)
        return stats

    def _reset_eval_stats(self):
        self.eval_statistics = {
            "Critic Target": [],
            "Critic Loss": [],
            "Critic Prediction": [],
            "Critic Target Prediction": [],
            "Actor Entropy": [],
            "Actor Loss": [],
            "Actor Log Prob": [],
            "Imagined Returns": [],
            "Imagined Advantage": [],
        }
