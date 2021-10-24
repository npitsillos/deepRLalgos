from collections import namedtuple

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F

from drl_algos.networks import Network, Base, Mlp, Gru


RSSMDiscState = namedtuple('RSSMDiscState', ['logit', 'stoch', 'deter'])
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter'])

class RSSM(Network):
    """
    todo:
        - Implement determinisitic sampling for continuous distribution

        - My RSSM determinstic state is not all zeros on the first step like
        in the original. I don't get how they got all zeros since they do have
        bias terms

    """

    def __init__(
        self,
        stoch_size,
        deter_size,
        action_size,
        hidden_size,
        obs_size,
        discrete_size=None,
        min_std=0.1,
        act_fn=F.elu,
        kl_balance=0.8,
        kl_free=0.0,
    ):
        super().__init__()

        # Parameters
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.discrete_size = discrete_size
        self.min_std = min_std
        self.act_fn = act_fn
        self.kl_balance = kl_balance
        self.kl_free = kl_free

        # Calculate stochastic state and distribution sizes
        if self.discrete_size is None:
            self.stoch_state_size = self.stoch_size
            self.stoch_dist_size = 2 * self.stoch_state_size
        else:
            self.stoch_state_size = self.stoch_size * self.discrete_size
            self.stoch_dist_size = self.stoch_state_size

        # Build Networks
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self.fc_embed_state_action = Mlp(
            [self.deter_size],
            self.stoch_state_size + self.action_size,
            self.act_fn
        )
        self.fc_prior = Mlp(
            [self.hidden_size, self.stoch_dist_size],
            self.deter_size,
            self.act_fn,
            act_last_layer=False
        )
        self.fc_posterior = Mlp(
            [self.hidden_size, self.stoch_dist_size],
            self.deter_size + self.obs_size,
            self.act_fn,
            act_last_layer=False
        )

    def imagine(
        self,
        prev_action,
        prev_rssm_state,
        sample=True
    ):
        """
        prev_action - (batch_size, prev_action)
        prev_rssm_state - (batch_size, rssm_size) for each rssm element

        output - (prev_action, rssm_size) for each rssm element
        """
        state_action_embed = self.fc_embed_state_action(
            torch.cat([prev_rssm_state.stoch, prev_action], dim=-1)
        )
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter)

        if self.discrete_size is None:
            prior_mean, prior_std = torch.chunk(
                self.fc_prior(deter_state), 2, dim=-1
            )
            stats = {'mean':prior_mean, 'std':prior_std}
            prior_stoch_state, std = self._get_stoch_state(stats, sample)
            prior_rssm_state = RSSMContState(
                prior_mean, std, prior_stoch_state, deter_state
            )
        else:
            prior_logit = self.fc_prior(deter_state)
            stats = {'logit':prior_logit}
            prior_stoch_state = self._get_stoch_state(stats, sample)
            prior_rssm_state = RSSMDiscState(
                prior_logit, prior_stoch_state, deter_state
            )

        return prior_rssm_state

    def rollout_imagination(
        self,
        horizon,
        actor,
        prev_rssm_state
    ):
        """
        output should be of length horizon+1
        action_i is the action calculated at state_i-1 that led to state_i
        """
        latent_state = self.get_latent_state(prev_rssm_state)
        action = torch.zeros_like(actor(latent_state).sample())
        rssm_state = prev_rssm_state
        rssm_states = [rssm_state]
        actions = [action]

        for t in range(horizon):
            latent_state = self.get_latent_state(rssm_state)
            action = actor(latent_state.detach()).rsample()
            rssm_state = self.imagine(action, rssm_state)
            rssm_states.append(rssm_state)
            actions.append(action)

        rssm_states = self._rssm_stack_states(rssm_states, dim=0)
        actions = torch.stack(actions, dim=0)
        return rssm_states, actions

    def observe(
        self,
        obs_embed,
        prev_action,
        prev_rssm_state,
        sample=True
    ):
        """
        obs_embed - (batch_size, embedding_size)
        prev_action - (batch_size, action_size)
        prev_rssm_state - (batch_size, rssm_size) for each rssm element

        output - (batch_size, rssm_size) for each rssm element
        """
        prior_rssm_state = self.imagine(
            prev_action, prev_rssm_state, sample
        )

        # CHANGELOG - zeroing out deterministic state on first step
        if (prev_rssm_state.stoch == 0).all():
            prior_rssm_state = RSSMDiscState(prior_rssm_state.logit, prior_rssm_state.stoch, torch.zeros_like(prior_rssm_state.deter))

        deter_state = prior_rssm_state.deter

        x = torch.cat([deter_state, obs_embed], dim=-1)

        if self.discrete_size is None:
            posterior_mean, posterior_std = torch.chunk(
                self.fc_posterior(x), 2, dim=-1
            )
            stats = {'mean':posterior_mean, 'std':posterior_std}
            posterior_stoch_state, std = self._get_stoch_state(stats, sample)
            posterior_rssm_state = RSSMContState(
                posterior_mean, std, posterior_stoch_state, deter_state
            )
        else:
            posterior_logit = self.fc_posterior(x)
            stats = {'logit':posterior_logit}
            posterior_stoch_state = self._get_stoch_state(stats, sample)
            posterior_rssm_state = RSSMDiscState(
                posterior_logit, posterior_stoch_state, deter_state
            )

        return prior_rssm_state, posterior_rssm_state

    def rollout_observation(
        self,
        seq_len,
        obs_embed,
        action,
        prev_rssm_state):
        """
        NOTES/TODO
            - in original implementation action at index i is the action taken
            in state i-1 that led to state i
            - the very first state in an episode has a zerod action
            - for the sequences, the very first action should be zerod
            - maybe add an if prev_rssm_state is None then initialise
                - original reuses rssm state from previous call but then zeros
                it out
                    - not sure if that provides an efficiency benefit over
                    initialising a new state or if it is something to do with
                    gradients (but that seems unlikely)

            - should be able to infer seq_len instead of taking it as a
            parameter
        """
        priors = []
        posteriors = []
        for t in range(seq_len):
            prev_action = action[t]
            prior_rssm_state, posterior_rssm_state = self.observe(
                obs_embed[t], prev_action, prev_rssm_state
            )
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        prior = self._rssm_stack_states(priors, dim=0)
        post = self._rssm_stack_states(posteriors, dim=0)
        return prior, post

    def kl_loss(
        self,
        post,
        prior
    ):
        """Confirmed equivalent to original"""
        # CHANGELOG - added this to test ignoring first kl_loss
        post = RSSMDiscState(post.logit[1:], post.stoch[1:], post.deter[1:])
        prior = RSSMDiscState(prior.logit[1:], prior.stoch[1:], prior.deter[1:])

        prior_dist = self.get_dist(prior)
        post_dist = self.get_dist(post)
        prior_dist_detach = self.get_dist(self.detach_state(prior))
        post_dist_detach = self.get_dist(self.detach_state(post))

        value_lhs = td.kl.kl_divergence(post_dist, prior_dist_detach)
        value_rhs = td.kl.kl_divergence(post_dist_detach, prior_dist)
        free = torch.tensor(self.kl_free).float().to(self.device)
        loss_lhs = torch.maximum(value_lhs.mean(), free)
        loss_rhs = torch.maximum(value_rhs.mean(), free)
        loss = self.kl_balance * loss_lhs + (1 - self.kl_balance) * loss_rhs
        return loss, value_lhs

    def init_state(
        self,
        batch_size,
    ):
        if self.discrete_size is None:
            return RSSMContState(
                torch.zeros(
                    batch_size, self.stoch_state_size
                ).to(self.device),
                torch.zeros(
                    batch_size, self.stoch_state_size
                ).to(self.device),
                torch.zeros(
                    batch_size, self.stoch_state_size
                ).to(self.device),
                torch.zeros(
                    batch_size, self.deter_size
                ).to(self.device),
            )
        else:
            return RSSMDiscState(
                torch.zeros(
                    batch_size, self.stoch_state_size
                ).to(self.device),
                torch.zeros(
                    batch_size, self.stoch_state_size
                ).to(self.device),
                torch.zeros(
                    batch_size, self.deter_size
                ).to(self.device),
            )

    def detach_state(
        self,
        rssm_state
    ):
        if self.discrete_size is None:
            return RSSMContState(
                rssm_state.mean.detach(),
                rssm_state.std.detach(),
                rssm_state.stoch.detach(),
                rssm_state.deter.detach()
            )
        else:
            return RSSMDiscState(
                rssm_state.logit.detach(),
                rssm_state.stoch.detach(),
                rssm_state.deter.detach(),
            )

    def flatten_state(
        self,
        state
    ):
        if self.discrete_size is None:
            return RSSMContState(
                torch.flatten(state.mean, 0, len(state.mean.shape)-2),
                torch.flatten(state.std, 0, len(state.std.shape)-2),
                torch.flatten(state.stoch, 0, len(state.stoch.shape)-2),
                torch.flatten(state.deter, 0, len(state.deter.shape)-2),
            )
        else:
            return RSSMDiscState(
                torch.flatten(state.logit, 0, len(state.logit.shape)-2),
                torch.flatten(state.stoch, 0, len(state.stoch.shape)-2),
                torch.flatten(state.deter, 0, len(state.deter.shape)-2),
            )


    def get_latent_state(
        self,
        rssm_state
    ):
        return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)

    def get_dist(
        self,
        rssm_state
    ):
        if self.discrete_size is None:
            return td.independent.Independent(
                td.Normal(rssm_state.mean, rssm_state.std), 1
            )
        else:
            shape = rssm_state.logit.shape
            logit = torch.reshape(
                rssm_state.logit,
                shape=(*shape[:-1], self.stoch_size, self.discrete_size)
            )
            return td.Independent(
                td.OneHotCategoricalStraightThrough(logits=logit), 1
            )

    def _get_stoch_state(
        self,
        stats,
        sample=True
    ):
        """
        TODO - implement deterministic sampling for continuous state
             - double check deterministic sampling (pretty certain it is
             correct)
        """
        if self.discrete_size is None:
            mean = stats['mean']
            std = stats['std']
            std = F.softplus(std) + self.min_std
            return mean + std*torch.randn_like(mean), std
        else:
            logit = stats['logit']
            shape = logit.shape
            logit = torch.reshape(
                logit,
                shape=(*shape[:-1], self.stoch_size, self.discrete_size)
            )
            if sample:
                dist = td.OneHotCategorical(logits=logit)
                stoch = dist.sample()
                stoch += dist.probs - dist.probs.detach()
            else:
                stoch = F.one_hot(
                    torch.argmax(logit, len(logit.shape)-1),
                    self.discrete_size
                )
            return torch.flatten(stoch, start_dim=-2, end_dim=-1)

    def _rssm_stack_states(
        self,
        rssm_states,
        dim
    ):
        if self.discrete_size is None:
            return RSSMContState(
                torch.stack([state.mean for state in rssm_states], dim=dim),
                torch.stack([state.std for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter for state in rssm_states], dim=dim),
            )
        else:
            return RSSMDiscState(
                torch.stack([state.logit for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter for state in rssm_states], dim=dim),
            )
