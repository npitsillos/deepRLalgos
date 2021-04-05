import torch
import numpy as np

class RolloutBuffer():

    def __init__(self, batch_size, use_lstm=False):
        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_state_values = []
        self.batch_rewards = []
        self.batch_dones = []

        if use_lstm:
            self.batch_h_ins = []
            self.batch_h_outs = []
        self.use_lstm = use_lstm
        self.batch_size = batch_size

    def store(self, sample):
        if self.use_lstm:
            obs, action, log_prob, value, reward, done, h_in, h_out = sample
        else:
            obs, action, log_prob, value, reward, done = sample
        self.batch_obs.append(obs)
        self.batch_acts.append(action)
        self.batch_log_probs.append(log_prob)
        self.batch_state_values.append(value)
        self.batch_rewards.append(reward)
        self.batch_dones.append(done)

        if self.use_lstm:
            self.batch_h_ins.append(h_in)
            self.batch_h_outs.append(h_out)

    def clear(self):
        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_state_values = []
        self.batch_rewards = []
        self.batch_dones = []

        if self.use_lstm:
            self.batch_h_ins = []
            self.batch_h_outs = []

    def get_batches(self):
        num_states = len(self.batch_obs)
        batch_start = np.arange(0, num_states, self.batch_size)
        indices = np.arange(num_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        obs = np.array(self.batch_obs)
        acts = np.array(self.batch_acts)
        act_log_probs = np.array(self.batch_log_probs)
        state_values = np.array(self.batch_state_values)
        rewards = np.array(self.batch_rewards)
        dones = np.array(self.batch_dones)

        if self.use_lstm:
            h_ins = np.array(self.batch_h_ins)
            h_outs = np.array(self.batch_h_outs)
            
            return obs, acts, act_log_probs, state_values, rewards, dones, h_ins, h_outs, batches

        return obs, acts, act_log_probs, state_values, rewards, dones, batches

class ReplayBuffer():

    def __init(self, size, obs_shape, act_shape):
        self.max_size = size
        self.position = 0
        self.size = 0
        self.batch_obs = np.zeros((size, obs_shape), dtype=np.float32)
        self.batch_acts = np.zeros((size, act_shape), dtype=np.float32)
        self.batch_next_obs = np.zeros((size, obs_shape), dtype=np.float32)
        self.batch_rewards = np.zeros(size, dtype=np.float32)
        self.batch_dones = np.zeros(size, dtype=np.float32)

    def store(self, sample):
        obs, action, next_obs, reward, done = sample
        self.batch_obs[self.position] = obs
        self.batch_acts[self.position] = action
        self.batch_next_obs[self.position] = next_obs
        self.batch_rewards[self.position] = reward
        self.batch_dones[self.position] = done

        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=128):
        idx = np.random.randint(0, self.size, batch_size)

        batch_obs = torch.as_tensor(self.batch_obs[idx], dtype=torch.float32)
        batch_next_obs = torch.as_tensor(self.batch_next_obs[idx], dtype=torch.float32)
        batch_acts = torch.as_tensor(self.batch_acts[idx], dtype=torch.float32)
        batch_rewards = torch.as_tensor(self.batch_rewards[idx], dtype=torch.float32)
        batch_done = torch.as_tensor(self.batch_dones[idx], dtype=torch.float32)

        return batch_obs, batch_next_obs, batch_acts, batch_rewards, batch_done