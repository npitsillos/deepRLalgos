import numpy as np

def compute_gae(rewards, state_values, dones, last_value, gae_lambda, gamma):
    gae_advantages = []
    last_gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = state_values[i+1]
        delta = rewards[i] + gamma * next_value * dones[i] - state_values[i]
        last_gae = delta + gamma * gae_lambda * dones[i] * last_gae
        gae_advantages.insert(0, last_gae)
    return np.array(gae_advantages, dtype=np.float32)

def compute_discounted_returns(rewards, dones, gamma):
    # Obtained from https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py
    returns = []
    discounted_reward = 0
    for i in reversed(range(len(rewards))):
        discounted_reward = rewards[i] + gamma * discounted_reward * (1 - int(dones[i]))
        returns.insert(0, discounted_reward)
    
    return np.array(returns, dtype=np.float32)