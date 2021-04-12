from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import torch

class RolloutBuffer:

    def __init__(self, obs_shape, act_shape, gamma, is_recurrent, gae_lamba, max_size=10000):
        self.gamma = gamma
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.is_recurrent = is_recurrent
        self.gae_lambda = gae_lamba
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.states = np.zeros((self.max_size, *self.obs_shape))
        self.actions = np.zeros((self.max_size, *self.act_shape))
        self.state_values = np.zeros(self.max_size)
        self.rewards = np.zeros(self.max_size)
        self.log_probs = np.zeros(self.max_size)
        self.dones = np.zeros(self.max_size)

        self.returns = np.zeros(self.max_size)
        self.advantages = np.zeros(self.max_size)

        self.size = 0
        self.traj_start = 0
        self.buffer_ready = False
        if self.is_recurrent:
            self.traj_idx = [0]

    def add_sample(self, sample):
        state, action, state_value, reward, log_prob, done = sample
        self.states[self.size] = state
        self.actions[self.size] = action
        self.state_values[self.size] = state_value
        self.rewards[self.size] = reward
        self.log_probs[self.size] = log_prob
        self.dones[self.size] = done

        self.size += 1

    def end_trajectory(self, final_value):
    
        if self.is_recurrent:
            self.traj_idx += [self.size]
        
        rewards = self.rewards[self.traj_start:self.size]
        state_values = self.state_values[self.traj_start:self.size]
        dones = self.dones[self.traj_start:self.size]

        advantages = np.zeros(len(rewards))
        last_gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = final_value
            else:
                next_value = state_values[i+1]
            delta = rewards[i] + self.gamma * next_value * dones[i] - state_values[i]
            last_gae = delta + self.gamma * self.gae_lambda * dones[i] * last_gae
            advantages[i] = last_gae
        
        self.returns[self.traj_start:self.size] = advantages + state_values

        self.advantages[self.traj_start:self.size] = advantages
        self.traj_start = self.size

    def sample_batch(self, batch_size=512):
        if not self.buffer_ready:
            self._finish_buffer()
        
        if self.is_recurrent:
            raise NotImplementedError("This is not supported yet")
        else:
            random_indices = SubsetRandomSampler(range(self.size))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)

            for i, indices in enumerate(sampler):
                states = self.states[indices]
                actions = self.actions[indices]
                returns = self.returns[indices]
                log_probs = self.log_probs[indices]
                advantages = self.advantages[indices]
                yield states, actions, returns, log_probs, advantages

    def _finish_buffer(self):
        with torch.no_grad():
            self.states = torch.tensor(self.states, dtype=torch.float)
            self.actions = torch.tensor(self.actions, dtype=torch.float)
            self.state_values = torch.tensor(self.state_values, dtype=torch.float)
            self.returns = torch.tensor(self.returns, dtype=torch.float)
            self.advantages = torch.tensor(self.advantages, dtype=torch.float)
            self.log_probs = torch.tensor(self.log_probs, dtype=torch.float)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-4)
        self.buffer_ready = True