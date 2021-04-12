import ray
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions import Categorical, Normal
from drl_algos.policies.actor import FeedForwardActor, RecurrentActor, CustomActor
from drl_algos.policies.critic import FeedForwardQ, FeedForwardValue, RecurrentQ, RecurrentValue, CustomModelQ, CustomModelValue

from drl_algos.utils.buffers_new import RolloutBuffer

class BaseAgent(nn.Module):
    
    def __init__(self, env_fn):
        super(BaseAgent, self).__init__()
        self.env = env_fn()
        self.state_dim = self.env.observation_space.shape
        # Assume space is continuous
        self.action_dim = self.env.action_space.shape
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = (self.env.action_space.n, )
    
    def sync_params(self, actor_params, critic_params=None):
        for param, new_param in zip(self.actor.parameters(), actor_params):
            param.data.copy_(new_param)
        if critic_params is not None:
            for param, new_param in zip(self.critic.parameters(), critic_params):
                param.data.copy_(new_param)

    def collect_rollout(self, min_steps):
        if self.is_on_policy:
            buffer = RolloutBuffer(self.state_dim, self.env.action_space.shape, 0.98, False, 0.95)
        else:
            buffer = ReplayBuffer()

        num_steps = 0

        while num_steps < min_steps:
            state = self.env.reset()
            done = False
            traj_steps = 0
            while not done:
                dist = self.get_action_dist(torch.tensor(state, dtype=torch.float))
                action = dist.sample()

                next_state, reward, done, info = self.env.step(action.numpy())
                if self.is_on_policy:
                    state_value = self.get_state_value(torch.tensor(state, dtype=torch.float))
                    act_log_prob = dist.log_prob(action)
                    sample = (state, action.detach().cpu().numpy(), state_value.detach().cpu().numpy(), reward, act_log_prob.detach().cpu().numpy(), done)
                else:
                    sample = (state, action.numpy(), next_state, reward, done)
                
                buffer.add_sample(sample)
                state = next_state
                traj_steps += 1
            if self.is_on_policy:
                terminal_state_value = 0 if done else self.get_state_value(torch.tensor(state, dtype=torch.float)).detach().cpu().numpy()
                buffer.end_trajectory(terminal_state_value)
            num_steps += traj_steps
        return buffer
    
    def evaluate(self):
        rewards = []
        for i in range(10):
            done = False
            state = self.env.reset()
            eps_reward = 0
            while not done:
                act = self.get_action_dist(torch.tensor(state, dtype=torch.float), deterministic=True)
                state, reward, done, info = self.env.step(act.item())
                eps_reward += reward

            rewards.append(eps_reward)

        return rewards            
        
class OnPolicyDiscreteActorCritic(BaseAgent):
    
    def __init__(self, env_fn):
        
        BaseAgent.__init__(self, env_fn)
        self.actor = FeedForwardActor(self.state_dim, self.action_dim)
        self.critic = FeedForwardValue(self.state_dim)
        self.dist = Categorical
        self.max_traj_len = 20
        self.is_on_policy = True
    
    def get_action_dist(self, state, deterministic=False):

        logits = self.actor(state)
        if deterministic:
            return torch.argmax(torch.softmax(logits, dim=0))
        else:
            dist = self.dist(logits=logits)
            return dist
    
    def get_state_value(self, state):
        return self.critic(state)


class OffPolicyContinuousActorCritic(BaseAgent):

    def __init__(self, state_dim, act_dim, env_fn, layers=(256, 256), activation_fn=F.relu, log_std=None):

        BaseAgent.__init__(self, env_fn)
        self.actor = FeedForwardActor(state_dim, act_dim, is_continuous=True, layers=layers, activation_fn=activation_fn)
        self.q1 = FeedForwardQ(state_dim, act_dim, layers=layers, activation_fn=activation_fn)
        self.q2 = deepcopy(self.q1)
        self.dist = Normal
        self.is_on_policy = False
        self.max_traj_len = 100
        if log_std is None:
            self.actor.log_std = nn.Linear(self.actor.layers[-1].out_features, act_dim)
    
    def get_action_dist(self, state, deterministic=False):

        means, std = self.actor(state)

        if deterministic:
            return means, std
        else:
            dist = self.dist(means, std)
            return dist
        