from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.kl import kl_divergence
import numpy as np
import gtimer as gt

from drl_algos.networks import Network, Base, Mlp, Gru
from drl_algos import utils, eval_util


class Dreamer(Network):
    """TODO - Add option to use gaussians like Dreamer V1."""

    def __init__(
        self,
        num_obs,
        num_actions,
        stoch,
        discrete,
        obs_encoder,
        repr_model,
        recur_model,
        transition_predictor,
        obs_decoder,
        reward_predictor,
        gamma_predictor,
        lr=2e-4,
        adam_eps=1e-5,
        adam_decay=1e-6,
        grad_clip=100,
    ):
        super().__init__()

        # Dreamer networks
        self.obs_encoder = obs_encoder # (obs_t) -> features_t
        self.repr_model = repr_model # (features_t, hidden_t) -> posterior_t
        self.recur_model = recur_model # (stoch_t-1, action_t-1, hidden_t-1) -> hidden_t
        self.transition_predictor = transition_predictor # (hidden_t) -> prior_t
        self.obs_decoder = obs_decoder # (hidden_t, stoch_t) -> obs_t
        self.reward_predictor = reward_predictor # (hidden_t, stoch_t) -> reward_t
        self.gamma_predictor = gamma_predictor # (hidden_t, stoch_t) -> gamma_t

        self.state_action_encoder = Mlp(
            input_size=repr_model.output_size + num_actions,
            layer_sizes=[recur_model.output_size],
            layer_init="orthogonal",
            layer_activation=F.elu,
            act_last_layer=True,
        )

        # Dreamer parameters
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.stoch = stoch
        self.discrete = discrete
        self.lr = lr
        self.adam_eps = adam_eps
        self.adam_decay = adam_decay
        self.grad_clip = grad_clip
        self.stoch_size = stoch*discrete
        self.deter_size = self.recur_model.output_size
        self.latent_size = self.deter_size + self.stoch_size

        # Loss function and optimizer
        self.loss_model = LossModel(self.stoch, self.discrete)
        self.optim = Adam(
            self.parameters(),
            lr=lr,
            eps=adam_eps,
            weight_decay=adam_decay
        )

        # Stats
        self._n_train_steps_total = 0
        self.eval_statistics = OrderedDict()
        self._num_train_steps = 0

    def train(self, batch):
        """TODO - Return model_states for behaviour learning."""
        self._num_train_steps += 1
        batch = utils.to_tensor_batch(batch, self.device)
        return self.train_on_batch(batch)

    def encode(self, batch):
        """Encode batch of sequences into model_states."""
        with torch.no_grad():
            batch = utils.to_tensor_batch(batch, self.device)
            obs = batch['observations']
            actions = batch['actions']
            next_obs = batch['next_observations']

            model_state = self.observe(obs[:,0])
            reward_pred = self.reward_predictor(model_state[0]).detach()
            gamma_pred = torch.distributions.bernoulli.Bernoulli(
                logits=self.gamma_predictor(model_state[0]).detach()
            ).probs
            latent_states = [model_state[0]]
            stochs = [model_state[1]]
            deters = [model_state[2]]
            rewards = [reward_pred] # Placeholder, no action for first observation
            gammas = [gamma_pred] # Valid since first state is never terminal
            for t in range(len(obs[0])):
                model_state = self.observe(next_obs[:,t], actions[:,t], model_state)
                reward_pred = self.reward_predictor(model_state[0]).detach()
                gamma_pred = torch.distributions.bernoulli.Bernoulli(
                    logits=self.gamma_predictor(model_state[0]).detach()
                ).probs
                latent_states.append(model_state[0])
                stochs.append(model_state[1])
                deters.append(model_state[2])
                rewards.append(reward_pred)
                gammas.append(gamma_pred)

            latent_states = torch.stack(latent_states, dim=1)
            latent_states = latent_states.reshape(-1, self.latent_size).detach()
            stochs = torch.stack(stochs, dim=1)
            stochs = stochs.reshape(-1, self.stoch_size).detach()
            deters = torch.stack(deters, dim=1)
            deters = deters.reshape(-1, self.deter_size).detach()
            actions = torch.cat((torch.zeros(actions.shape[0], 1, 1), actions), dim=1)
            actions = actions.reshape(-1, actions.shape[-1])
            rewards = torch.cat(rewards, dim=1).reshape(-1,1).detach()
            gammas = torch.cat(gammas, dim=1).reshape(-1,1).detach()
            return (latent_states, stochs, deters), actions, rewards, gammas

    def observe(self, obs, act=None, model_state=None, sample=True):
        """Observes an environment transition."""
        obs = utils.to_tensor(obs, self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act is not None:
            act = utils.to_tensor(act, self.device)

        if model_state is None:
            stoch = deter = None
        else:
            _, stoch, deter = model_state

        stoch, deter = self._observe(obs, act, stoch, deter, sample)
        latent_state = torch.cat((deter, stoch.reshape(-1, self.stoch*self.discrete)),
                                 dim=1).detach()
        return (latent_state, stoch, deter)

    def dream(self, model_state, policy, horizon):
        """Dreams with a stochastic policy.

        Action samples and log probs have gradients attached.
        """
        latent_state, stoch, deter = model_state
        latent_state = latent_state.detach()
        with torch.no_grad():
            reward = self.reward_predictor(model_state[0]).detach()
            gamma = torch.distributions.bernoulli.Bernoulli(
                logits=self.gamma_predictor(model_state[0]).detach()
            ).probs
        states = [latent_state]
        actions = []
        log_pis = []
        rewards = [reward]
        gammas = [gamma]
        for i in range(horizon):
            policy_dist = policy(latent_state)
            # CHANGE LOG - with discrete actions rsample not guarenteed
            # action, log_pi = policy_dist.rsample_and_logprob() # keep attached
            # TODO - I've hacked this part to make it work for Discrete distributions
            #        The implementation for sample indexes the array but that doesn't work when your
            #        your sampling multiple actions
            #        I'll need to update get_action in policy to make it work without
            action = policy_dist.categorical.distribution.sample()
            # TODO - below is just a hack to make it work for environments with
            #        2 discrete actions, i.e., CartPole, this should be made more
            #        general
            log_pi = policy_dist.log_prob(action)
            action = F.one_hot(action, 2)
            with torch.no_grad():
                latent_state, deter, stoch, reward, gamma_logits = self._dream(
                    action,
                    stoch,
                    deter
                )
                gamma = torch.distributions.bernoulli.Bernoulli(
                    logits=gamma_logits.detach()
                ).probs
            states.append(latent_state)
            actions.append(action)
            log_pis.append(log_pi.unsqueeze(-1))
            rewards.append(reward)
            gammas.append(gamma)
        policy_dist = policy(latent_state)
        # CHANGE LOG - with discrete actions rsample not guarenteed
        # action, log_pi = policy_dist.rsample_and_logprob() # keep attached
        # TODO - I've hacked this part to make it work for Discrete distributions
        #        The implementation for sample indexes the array but that doesn't work when your
        #        your sampling multiple actions
        #        I'll need to update get_action in policy to make it work without
        action = policy_dist.categorical.distribution.sample()
        # TODO - below is just a hack to make it work for environments with
        #        2 discrete actions, i.e., CartPole, this should be made more
        #        general
        log_pi = policy_dist.log_prob(action)
        action = F.one_hot(action, 2)
        actions.append(action)
        log_pis.append(log_pi.unsqueeze(-1))

        states = torch.stack(states, dim=1)
        actions = torch.stack(actions, dim=1)
        log_pis = torch.stack(log_pis, dim=1)
        rewards = torch.stack(rewards, dim=1)
        gammas = torch.stack(gammas, dim=1)
        return states, actions, log_pis, rewards, gammas

    def train_on_batch(self, batch):
        """TODO - Return model states for behaviour learning."""
        gt.blank_stamp()
        self._n_train_steps_total += 1

        model_states, loss, self.eval_statistics = self.compute_loss(batch)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optim.step()

        gt.stamp('dreamer training', unique=False)
        return model_states

    def compute_loss(self, batch):
        """TODO - return model_states for behaviour learning."""
        rewards = batch['rewards']
        gammas = -batch['terminals'] + 1
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # Lists to hold model_states
        latent_states = []
        stochs = []
        deters = []

        # Process first observation on its own to initialise model
        post_logits, post_sample, prior_logits, obs_pred, reward_pred, gamma_pred, deter = self._train(
            obs[:,0],
        )
        loss = self.loss_model(
            obs[:,0],
            None,
            gammas[:,0], # first state is never terminal so this is okay
            obs_pred,
            None,
            gamma_pred,
            post_logits,
            prior_logits
        )

        latent_state = torch.cat((deter, post_sample.reshape(-1, self.stoch*self.discrete)),
                                 dim=1)
        latent_states.append(latent_state)
        stochs.append(post_sample)
        deters.append(deter)

        # Iterate through sequence
        sequence_len = len(rewards[0])
        for t in range(sequence_len):
            post_logits, post_sample, prior_logits, obs_pred, reward_pred, gamma_pred, deter = self._train(
                next_obs[:,t],
                actions[:,t],
                post_sample,
                deter,
            )
            loss += self.loss_model(
                next_obs[:,t],
                rewards[:,t], #r time array starts at 1; 0: t=1
                gammas[:,t], #g time array starts at 1; 0: t=1
                obs_pred,
                reward_pred,
                gamma_pred,
                post_logits,
                prior_logits
            )
            latent_state = torch.cat((deter, post_sample.reshape(-1, self.stoch*self.discrete)),
                                     dim=1)
            latent_states.append(latent_state)
            stochs.append(post_sample)
            deters.append(deter)
        loss /= sequence_len

        latent_states = torch.stack(latent_states, dim=1)
        latent_states = latent_states.reshape(-1, self.latent_size).detach()
        stochs = torch.stack(stochs, dim=1)
        stochs = stochs.reshape(-1, self.stoch_size).detach()
        deters = torch.stack(deters, dim=1)
        deters = deters.reshape(-1, self.deter_size).detach()

        return (latent_states, stochs, deters), loss, self.loss_model.get_stats()

    def compute_deter(self, batch_size, act=None, stoch=None, deter=None):
        if deter is None:
            deter = torch.zeros((batch_size, self.recur_model.output_size))
            deter = deter.to(self.device)
        else:
            embed = self.state_action_encoder(torch.cat((stoch.reshape(-1, self.stoch*self.discrete), act), dim=1))
            deter = self.recur_model(embed, deter)
        return deter

    def encode_repr(self, obs, deter):
        features = self.obs_encoder(obs)
        features = features.reshape(-1, self.obs_encoder.output_size)
        input = torch.cat((features, deter), dim=1)
        post_logits = self.repr_model(input)
        post_sample = self.compute_post_sample(post_logits)
        return post_logits, post_sample

    def compute_post_sample(self, post_logits):
        # CHANGELOG - use dist.probs instead of softmax on logits
        post_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=post_logits.reshape(-1, self.stoch, self.discrete)
        )
        post_sample = post_dist.sample() + post_dist.probs - post_dist.probs.detach()
        return post_sample
        # post_sample = torch.distributions.one_hot_categorical.OneHotCategorical(
        #     logits=post_logits.reshape(-1, self.stoch, self.discrete)
        # ).sample()
        # z_probs = torch.softmax(post_logits.reshape(-1, self.stoch, self.discrete), dim=-1)
        # return post_sample + z_probs - z_probs.detach()

    def _observe(self, obs, act=None, stoch=None, deter=None, sample=True):
        with torch.no_grad():
            features = self.obs_encoder(obs)
            features = features.reshape(-1, self.obs_encoder.output_size)
            deter = self.compute_deter(obs.shape[0], act, stoch, deter)
            input = torch.cat((features, deter), dim=1)
            post_logits = self.repr_model(input)
            post_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=post_logits.reshape(-1, self.stoch, self.discrete)
            )
            if sample:
                post_sample = post_dist.sample()
            else:
                post_sample = F.one_hot(post_dist.mean.argmax(dim=2),
                                        self.discrete)
            return post_sample.detach(), deter.detach()

    def _dream(self, act, stoch, deter):
        with torch.no_grad():
            deter = self.compute_deter(act.shape[0], act, stoch, deter).detach()
            prior_logits = self.transition_predictor(deter)
            prior_sample = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=prior_logits.reshape(-1, self.stoch, self.discrete)
            ).sample() # no straight-through gradient
            latent_state = torch.cat((deter, prior_sample.reshape(-1, self.stoch*self.discrete)),
                                     dim=1).detach()
            reward_pred = self.reward_predictor(latent_state).detach()
            gamma_logits = self.gamma_predictor(latent_state).detach()
            return latent_state, deter, prior_sample, reward_pred, gamma_logits

    def _train(self, obs, act=None, stoch=None, deter=None):
        deter = self.compute_deter(obs.shape[0], act, stoch, deter)
        post_logits, post_sample = self.encode_repr(obs, deter)
        prior_logits = self.transition_predictor(deter)
        latent_state = torch.cat((deter, post_sample.reshape(-1, self.stoch*self.discrete)),
                                 dim=1)
        reward_pred = self.reward_predictor(latent_state)
        gamma_pred = self.gamma_predictor(latent_state)
        obs_pred = self.obs_decoder(latent_state)
        return post_logits, post_sample, prior_logits, obs_pred, reward_pred, gamma_pred, deter

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num train calls', self._num_train_steps),
        ])
        stats.update(self.eval_statistics)
        return stats

    def get_snapshot(self):
        return dict(
            obs_encoder=self.obs_encoder,
            repr_model=self.repr_model,
            recur_model=self.recur_model,
            transition_predictor=self.transition_predictor,
            obs_decoder=self.obs_decoder,
            reward_predictor=self.reward_predictor,
            gamma_predictor=self.gamma_predictor,
        )

    def end_epoch(self, epoch):
        pass


class LossModel(nn.Module):

    def __init__(self, stoch, discrete, obs_scale=1, reward_scale=1,
                 gamma_scale=5, kl_scale=.1, kl_balance=.8):
        """
        TODO:
            - needs tested (loss and stats)
                - especially kl_balancing
            - include free nats
                - kl_balance is meant to replace this, but it looks like they
                still used this for a couple environments
            - include free_avg parameters
                - this implementation is equivalent to free_avg=True
                - I don't think any of their tests used free_avg=False
            - include forward parameter
                - this implementation is equivalent to forward=False
                - I don't think any implementations used forward=True and I'm
                not clear what its benefit it

        Tuning advice:
            kl_scale:
                - .1 for discrete, 1 for continuous control
                - recommended search range {.1, .3, 1, 3}
        """
        super(LossModel, self).__init__()
        self.stoch = stoch
        self.discrete = discrete
        self.obs_scale = obs_scale
        self.reward_scale = reward_scale
        self.gamma_scale = gamma_scale
        self.kl_scale = kl_scale
        self.kl_balance = kl_balance
        self.reset_stats()

    def forward(self, obs, reward, gamma, obs_pred, reward_pred, gamma_logits,
                post_logits, prior_logits):
        # CHANGELOG - no longer calculate reward loss if reward pred not passed

        # Create distributions
        obs_dist = torch.distributions.normal.Normal(
            loc=obs_pred,
            scale=1.0
        )
        if reward_pred is not None:
            reward_dist = torch.distributions.normal.Normal(
                loc=reward_pred,
                scale=1.0
            )
        gamma_dist = torch.distributions.bernoulli.Bernoulli(
            logits=gamma_logits
        )
        post_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=post_logits.reshape(-1, self.stoch, self.discrete)
        )
        prior_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=prior_logits.reshape(-1, self.stoch, self.discrete)
        )
        post_dist_detach = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=post_logits.reshape(-1, self.stoch, self.discrete).detach()
        )
        prior_dist_detach = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=prior_logits.reshape(-1, self.stoch, self.discrete).detach()
        )

        # Calculate loss
        obs_loss = -obs_dist.log_prob(obs).mean() # the best this can converge to is .9189 because of the scale
        if reward_pred is not None:
            reward_loss = -reward_dist.log_prob(reward).mean() # the best this can converge to is .9189 because of the scale
        # TODO - I don't think the .round() is necessary
        gamma_loss = -gamma_dist.log_prob(gamma.round()).mean() # seems to learn with this

        # CHANGELOG - copied directly from tensorflow implementation
        value_lhs = kl_value = kl_divergence(post_dist, prior_dist_detach)
        value_rhs = kl_divergence(post_dist_detach, prior_dist)
        loss_lhs = torch.maximum(value_lhs.mean(), torch.tensor(0.0))
        loss_rhs = torch.maximum(value_rhs.mean(), torch.tensor(0.0))
        kl_loss = ((1 - self.kl_balance) * loss_lhs) + (self.kl_balance * loss_rhs)
        # kl_loss = ((1 - self.kl_balance) * value_lhs) + (self.kl_balance * value_rhs)


        # kl_value = kl_divergence(post_dist_detach, prior_dist)
        # prior_loss = self.kl_balance * kl_value
        # post_loss = (1 - self.kl_balance) * kl_divergence(post_dist, prior_dist_detach)
        # # CHANGELOG - bugfixed kl_scale being applied twice
        # kl_loss = (prior_loss + post_loss).mean()
        if reward_pred is not None:
            loss = (self.obs_scale*obs_loss
                    + self.reward_scale*reward_loss
                    + self.gamma_scale*gamma_loss
                    + self.kl_scale*kl_loss)
        else:
            loss = (self.obs_scale*obs_loss
                    + self.gamma_scale*gamma_loss
                    + self.kl_scale*kl_loss)

        # Track metrics
        self.losses.append(loss.cpu().detach().numpy())
        self.obs_losses.append(obs_loss.cpu().detach().numpy())
        if reward_pred is not None:
            self.reward_losses.append(reward_loss.cpu().detach().numpy())
        self.gamma_losses.append(gamma_loss.cpu().detach().numpy())
        self.kl_losses.append(kl_loss.cpu().detach().numpy())
        self.kl_values.append(kl_value.cpu().detach().numpy())
        self.post_ents.append(post_dist.entropy().cpu().detach().numpy())
        self.prior_ents.append(prior_dist.entropy().cpu().detach().numpy())

        return loss

    def get_stats(self):
        stats = OrderedDict()
        stats['Loss'] = np.mean([self.losses])
        stats['Observation Loss'] = np.mean(self.obs_losses)
        stats['Reward Loss'] = np.mean(self.reward_losses)
        stats['Gamma Loss'] = np.mean(self.gamma_losses)
        stats['KL Loss'] = np.mean(self.kl_losses)
        stats['KL Value'] = np.mean(self.kl_values)
        stats['Post Entropy'] = np.mean(self.post_ents)
        stats['Prior Entropy'] = np.mean(self.prior_ents)
        self.reset_stats()
        return stats

    def reset_stats(self):
        self.losses = []
        self.obs_losses = []
        self.reward_losses = []
        self.gamma_losses = []
        self.kl_losses = []
        self.kl_values = []
        self.post_ents = []
        self.prior_ents = []


class MlpDreamer(Dreamer):
    def __init__(
        self,
        obs_dim,
        act_dim,
        stoch=32,
        discrete=32,
        encoder_layers=[256],
        repr_hidden_sizes=[1024, 1024],
        recur_layers=[512],
        tran_hidden_sizes=[1024, 1024],
        decoder_hidden_sizes=[256],
        reward_hidden_sizes=[256],
        gamma_hidden_sizes=[256],
        layer_init="orthogonal",
        layer_activation=F.relu,
        **kwargs
    ):
        encoder = Mlp(
            input_size=obs_dim,
            layer_sizes=encoder_layers,
            layer_init=layer_init,
            layer_activation=layer_activation,
        )
        repr_model = Mlp(
            input_size=encoder_layers[-1] + recur_layers[-1],
            layer_sizes=repr_hidden_sizes + [stoch*discrete],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        recur_model = Gru(
            input_size=stoch*discrete + act_dim,
            layer_sizes=recur_layers,
        )
        tran_model = Mlp(
            input_size=recur_layers[-1],
            layer_sizes=tran_hidden_sizes + [stoch*discrete],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        decoder = Mlp(
            input_size=stoch*discrete + recur_layers[-1],
            layer_sizes=decoder_hidden_sizes + [obs_dim],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        reward_model = Mlp(
            input_size=stoch*discrete + recur_layers[-1],
            layer_sizes=reward_hidden_sizes + [1],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        gamma_model = Mlp(
            input_size=stoch*discrete + recur_layers[-1],
            layer_sizes=gamma_hidden_sizes + [1],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        super().__init__(
            num_obs=obs_dim,
            num_actions=act_dim,
            stoch=stoch,
            discrete=discrete,
            obs_encoder=encoder,
            repr_model=repr_model,
            recur_model=recur_model,
            transition_predictor=tran_model,
            obs_decoder=decoder,
            reward_predictor=reward_model,
            gamma_predictor=gamma_model,
            **kwargs
        )


class MlpDreamer2(Dreamer):
    """I think this is closer to the original

    CHANGELOG - tran_hidden_sizes [600, 600] -> [600]"""

    def __init__(
        self,
        obs_dim,
        act_dim,
        stoch=32,
        discrete=32,
        encoder_layers=[400, 400, 400, 400],
        repr_hidden_sizes=[600],
        recur_layers=[600],
        tran_hidden_sizes=[600],
        decoder_hidden_sizes=[400, 400, 400, 400],
        reward_hidden_sizes=[400, 400, 400, 400],
        gamma_hidden_sizes=[400, 400, 400, 400],
        layer_init="orthogonal",
        layer_activation=F.elu,
        **kwargs
    ):
        encoder = Mlp(
            input_size=obs_dim,
            layer_sizes=encoder_layers,
            layer_init=layer_init,
            layer_activation=layer_activation,
        )
        repr_model = Mlp(
            input_size=encoder_layers[-1] + recur_layers[-1],
            layer_sizes=repr_hidden_sizes + [stoch*discrete],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        recur_model = Gru(
            input_size=recur_layers[-1],
            layer_sizes=recur_layers,
        )
        tran_model = Mlp(
            input_size=recur_layers[-1],
            layer_sizes=tran_hidden_sizes + [stoch*discrete],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        decoder = Mlp(
            input_size=stoch*discrete + recur_layers[-1],
            layer_sizes=decoder_hidden_sizes + [obs_dim],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        reward_model = Mlp(
            input_size=stoch*discrete + recur_layers[-1],
            layer_sizes=reward_hidden_sizes + [1],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        gamma_model = Mlp(
            input_size=stoch*discrete + recur_layers[-1],
            layer_sizes=gamma_hidden_sizes + [1],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        super().__init__(
            num_obs=obs_dim,
            num_actions=act_dim,
            stoch=stoch,
            discrete=discrete,
            obs_encoder=encoder,
            repr_model=repr_model,
            recur_model=recur_model,
            transition_predictor=tran_model,
            obs_decoder=decoder,
            reward_predictor=reward_model,
            gamma_predictor=gamma_model,
            **kwargs
        )


class MlpDreamer2Small(Dreamer):
    """I think this is closer to the original

    CHANGELOG - tran_hidden_sizes [600, 600] -> [600]"""

    def __init__(
        self,
        obs_dim,
        act_dim,
        stoch=20,
        discrete=20,
        encoder_layers=[100, 100, 100],
        repr_hidden_sizes=[200],
        recur_layers=[200],
        tran_hidden_sizes=[200],
        decoder_hidden_sizes=[100, 100, 100],
        reward_hidden_sizes=[100, 100, 100],
        gamma_hidden_sizes=[100, 100, 100],
        layer_init="orthogonal",
        layer_activation=F.elu,
        **kwargs
    ):
        encoder = Mlp(
            input_size=obs_dim,
            layer_sizes=encoder_layers,
            layer_init=layer_init,
            layer_activation=layer_activation,
        )
        repr_model = Mlp(
            input_size=encoder_layers[-1] + recur_layers[-1],
            layer_sizes=repr_hidden_sizes + [stoch*discrete],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        recur_model = Gru(
            input_size=recur_layers[-1],
            layer_sizes=recur_layers,
        )
        tran_model = Mlp(
            input_size=recur_layers[-1],
            layer_sizes=tran_hidden_sizes + [stoch*discrete],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        decoder = Mlp(
            input_size=stoch*discrete + recur_layers[-1],
            layer_sizes=decoder_hidden_sizes + [obs_dim],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        reward_model = Mlp(
            input_size=stoch*discrete + recur_layers[-1],
            layer_sizes=reward_hidden_sizes + [1],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        gamma_model = Mlp(
            input_size=stoch*discrete + recur_layers[-1],
            layer_sizes=gamma_hidden_sizes + [1],
            layer_init=layer_init,
            layer_activation=layer_activation,
            act_last_layer=False,
        )
        super().__init__(
            num_obs=obs_dim,
            num_actions=act_dim,
            stoch=stoch,
            discrete=discrete,
            obs_encoder=encoder,
            repr_model=repr_model,
            recur_model=recur_model,
            transition_predictor=tran_model,
            obs_decoder=decoder,
            reward_predictor=reward_model,
            gamma_predictor=gamma_model,
            **kwargs
        )

"""TODO - Add CnnDreamer"""
