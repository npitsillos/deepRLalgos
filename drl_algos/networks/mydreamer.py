import time
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

class CnnEncoder(Network):

    def __init__(self, depth=32, stride=2, shape=(1,64,64), activation=F.elu):
        super().__init__()
        self.depth = depth
        self.stride = stride
        self.shape = shape
        self.activation = activation

        self.conv1 = torch.nn.Conv2d(shape[0], 1 * depth, 4, stride)
        self.conv2 = torch.nn.Conv2d(1 * depth, 2 * depth, 4, stride)
        self.conv3 = torch.nn.Conv2d(2 * depth, 4 * depth, 4, stride)
        self.conv4 = torch.nn.Conv2d(4 * depth, 8 * depth, 4, stride)

        self.output_size = 1024

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        x = torch.reshape(obs, (-1, *img_shape))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = torch.reshape(x, (*batch_shape, -1))
        return x

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size

class CnnDecoder(Network):

    def __init__(self, depth=32, stride=2, shape=(1,64,64), activation=F.elu,
                 embed_size=1624):
        super().__init__()
        self.depth = depth
        self.stride = stride
        self.shape = shape
        self.activation = activation
        self.embed_size = embed_size

        c, h, w = shape
        conv1_kernel_size = 6
        conv2_kernel_size = 6
        conv3_kernel_size = 5
        conv4_kernel_size = 5
        padding = 0
        conv1_shape = conv_out_shape((h, w), padding, conv1_kernel_size, stride)
        conv1_pad = output_padding_shape((h, w), conv1_shape, padding, conv1_kernel_size, stride)
        conv2_shape = conv_out_shape(conv1_shape, padding, conv2_kernel_size, stride)
        conv2_pad = output_padding_shape(conv1_shape, conv2_shape, padding, conv2_kernel_size, stride)
        conv3_shape = conv_out_shape(conv2_shape, padding, conv3_kernel_size, stride)
        conv3_pad = output_padding_shape(conv2_shape, conv3_shape, padding, conv3_kernel_size, stride)
        conv4_shape = conv_out_shape(conv3_shape, padding, conv4_kernel_size, stride)
        conv4_pad = output_padding_shape(conv3_shape, conv4_shape, padding, conv4_kernel_size, stride)
        self.conv_shape = (32 * depth, *conv4_shape)
        self.linear = torch.nn.Linear(embed_size, 32 * depth * np.prod(conv4_shape).item())
        self.dconv1 = torch.nn.ConvTranspose2d(32 * depth, 4 * depth, conv4_kernel_size, stride, output_padding=conv4_pad)
        self.dconv2 = torch.nn.ConvTranspose2d(4 * depth, 2 * depth, conv3_kernel_size, stride, output_padding=conv3_pad)
        self.dconv3 = torch.nn.ConvTranspose2d(2 * depth, 1 * depth, conv2_kernel_size, stride, output_padding=conv2_pad)
        self.dconv4 = torch.nn.ConvTranspose2d(1 * depth, shape[0], conv1_kernel_size, stride, output_padding=conv1_pad)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.activation(self.dconv1(x))
        x = self.activation(self.dconv2(x))
        x = self.activation(self.dconv3(x))
        x = self.dconv4(x)
        mean = torch.reshape(x, (*batch_shape, *self.shape))
        return mean

def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))

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
        stoch_size,
        deter_size,
        action_size,
        rssm_hidden_size,
        obs_size,
        hidden_sizes,
        act_fn=F.elu,
        discrete_size=None,
        min_std=0.1,
        discount_scale=5.,
        kl_scale=.1,
        kl_balance=0.8,
        kl_free=0.0,
        lr=2e-4,
        adam_eps=1e-5,
        adam_decay=1e-6,
        grad_clip=100,
    ):
        super().__init__()

        # Parameters
        self.action_size = action_size
        self.discount_scale = discount_scale
        self.kl_scale = kl_scale
        self.grad_clip = grad_clip
        self.obs_size = obs_size

        # Build networks
        # self.obs_encoder = Mlp(
        #     hidden_sizes, obs_size, act_fn
        # )
        self.obs_encoder = CnnEncoder()
        self.rssm = RSSM(
            stoch_size, deter_size, action_size, rssm_hidden_size,
            self.obs_encoder.output_size, discrete_size, min_std, act_fn,
            kl_balance, kl_free
        )
        self.latent_size = self.rssm.stoch_state_size + self.rssm.deter_size
        # self.obs_decoder = Mlp(
        #     hidden_sizes + [obs_size], self.latent_size, act_fn, False
        # )
        self.obs_decoder = CnnDecoder(embed_size=self.latent_size)
        self.reward_model = Mlp(
            hidden_sizes + [1], self.latent_size, act_fn, False
        )
        self.discount_model = Mlp(
            hidden_sizes + [1], self.latent_size, act_fn, False
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

        self.obs_dist = None

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
            if prev_model_state is None:
                prev_model_state = self.rssm.init_state(1)
            obs = utils.to_tensor(obs, self.device)
            obs_embed = self.obs_encoder(obs.unsqueeze(0))
            _, post = self.rssm.observe(
                obs_embed, prev_action, prev_model_state, sample
            )
            return post

    def imagine(
        self,
        policy,
        start_states,
        start_terminals,
        horizon,
        discount
    ):
        # Rollout imagination
        start_states = self.rssm.flatten_state(start_states)
        start_terminals = torch.flatten(start_terminals, 0 , 1)
        states_seq, action_seq = self.rssm.rollout_imagination(
            horizon, policy, start_states
        )
        latent_state_seq = self.rssm.get_latent_state(states_seq)

        # Get Rewards
        reward_loc = self.reward_model(latent_state_seq)
        reward_dist = td.Independent(td.Normal(reward_loc, 1), 1)
        reward_seq = reward_dist.mean

        # Get discounts
        discount_logits = self.discount_model(latent_state_seq)
        discount_dist = td.Independent(td.Bernoulli(logits=discount_logits), 1)
        discount_seq = discount_dist.mean
        discount_seq[0] = start_terminals # Override starting terminal
        discount_seq *= discount

        # Get downweight sequence
        # TODO - would it not be better if this was done BEFORE scaling the
        # discount sequence by the discount
        downweight_seq = torch.cumprod(
            torch.cat((torch.ones_like(discount_seq[0:1]), discount_seq[1:])), 0
        )

        return latent_state_seq, action_seq, reward_seq, discount_seq, downweight_seq

    def train(self, batch):
        self._num_train_steps += 1
        batch = utils.to_tensor_batch(batch, self.device)
        return self.train_on_batch(batch)

    def train_on_batch(self, batch):
        gt.blank_stamp()
        loss, posteriors, discounts = self.compute_loss(batch)

        # Backpropogate loss
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optim.step()

        gt.stamp('dreamer training', unique=False)
        return posteriors, discounts

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

        # Tanh transform rewards for atari
        rewards = torch.tanh(rewards)

        # Rollout the RSSM
        embed = self.obs_encoder(obs)
        prev_rssm_state = self.rssm.init_state(batch_size)
        prior, posterior = self.rssm.rollout_observation(
            seq_len, embed, actions, prev_rssm_state
        )

        # Predict with heads
        post_latent_state = self.rssm.get_latent_state(posterior)
        obs_loc = self.obs_decoder(post_latent_state)
        reward_loc = self.reward_model(post_latent_state)
        discount_logits = self.discount_model(post_latent_state)

        # Construct head distributions
        # Note - I don't think the Independent is really necessary
        # start = time.time()
        # if self.obs_dist is None:
        #     print("is none")
        #     self.obs_dist = td.Normal(obs_loc, 1)
        #     obs_dist = self.obs_dist
        # else:
        #     print("not none")
        #     start = time.time()
        #     self.obs_dist.loc = obs_loc
        #     obs_dist = self.obs_dist
        #     print("set loc", time.time()-start)
        start = time.time()
        start2 = time.time()
        # obs_dist = td.Independent(td.Normal(obs_loc, 1), len(self.obs_size))
        print("obs dist", time.time()-start2)
        start2 = time.time()
        # reward_dist = td.Independent(td.Normal(reward_loc, 1), 1)
        print("reward dist", time.time()-start2)
        start2 = time.time()
        reward_dist = td.Normal(post_latent_state, 1, validate_args=False)
        print("dummy dist", time.time()-start2)
        start2 = time.time()
        x = td.Bernoulli(logits=discount_logits)
        print("disc dist", time.time()-start2)
        start2 = time.time()
        x = td.Independent(x, 1)
        print("disc dist", time.time()-start2)
        start2 = time.time()
        x = td.Bernoulli(logits=discount_logits)
        print("disc dist", time.time()-start2)
        discount_dist = td.Independent(td.Bernoulli(logits=discount_logits), 1)
        print("discount dist", time.time()-start2)
        print("construct all dists", time.time()-start)

        # Calculate losses
        start = time.time()
        obs_loss = -obs_dist.log_prob(obs).mean()
        reward_loss = -reward_dist.log_prob(rewards).mean()
        discount_loss = -discount_dist.log_prob(discounts).mean()
        kl_loss, kl_value = self.rssm.kl_loss(posterior, prior)
        model_loss = (
            obs_loss
            + reward_loss
            + self.discount_scale * discount_loss
            + self.kl_scale * kl_loss
        )

        # Update eval stats
        self.eval_statistics['Kl Loss'].append(
            kl_loss.item()
        )
        self.eval_statistics['Kl Value'].append(
            kl_value.mean().item()
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
        self.eval_statistics['Posterior Entropy'].append(
            self.rssm.get_dist(posterior).entropy().mean().item()
        )
        self.eval_statistics['Prior Entropy'].append(
            self.rssm.get_dist(prior).entropy().mean().item()
        )

        return model_loss, posterior, discounts

    def end_epoch(
        self,
        epoch
    ):
        self._reset_eval_stats()

    def get_latent_state(
        self,
        state
    ):
        return self.rssm.get_latent_state(state)

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num train calls', self._num_train_steps),
        ])
        for key, value in self.eval_statistics.items():
            stats[key] = np.mean(value)
        return stats

    def get_snapshot(self):
        return dict(
            obs_encoder=self.obs_encoder,
            rssm=self.rssm,
            obs_decoder=self.obs_decoder,
            reward_model=self.reward_model,
            discount_model=self.discount_model,
        )

    def _reset_eval_stats(self):
        self.eval_statistics = {
            "Kl Loss": [],
            "Kl Value": [],
            "Observation Loss": [],
            "Reward Loss": [],
            "Discount Loss": [],
            "Posterior Entropy": [],
            "Prior Entropy": [],
        }

class ActorCritic(Network):

    def __init__(
        self,
        world_model,
        hidden_sizes,
        horizon=15,
        discount=0.995,
        act_fn=F.elu,
        policy_lr=4e-5,
        critic_lr=1e-4,
        grad_clip=100,
        adam_eps=1e-5,
        adam_decay=1e-6,
        slow_update_freq=100,
        lam=.95,
        entropy_scale=1e-3,
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

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            critic=self.critic,
            critic_target=self.critic_target,
        )

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
