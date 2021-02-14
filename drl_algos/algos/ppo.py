import gym
import torch
import numpy as np
import torch.nn.functional as F

from torch.distributions import Categorical, Normal
from torch.optim import Adam

from drl_algos.utils import RolloutBuffer
from drl_algos.utils.utils import compute_gae, compute_discounted_returns

class PPO():

    def __init__(self, policy, env, logger, config, callback=None):
        self.policy = policy
        self.env = env
        self.logger = logger
        self.config = config
        self.callback = callback

        self.optimizer = Adam(self.policy.parameters(), lr=self.config.LEARNING_RATE)
        self.buffer = RolloutBuffer(self.config.BATCH_SIZE)

    def learn(self):

        timesteps = 0
        best_score = self.env.reward_range[0]
        episodes = 0
        while timesteps < self.config.TOTAL_TIMESTEPS:

            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                
                if self.config.NORM_OBS:
                    obs = (obs - obs.mean()) / (obs.std() + 1e-10)
                
                # Feedforward observation to get action_probs and state_value
                policy_obs = torch.tensor(obs, dtype=torch.float).to(self.config.DEVICE)
                action_probs, state_value = self.policy(policy_obs)
                dist = self.policy.get_action_dist(action_probs)

                # Sample action from distribution
                action = dist.sample()
                act_log_prob = torch.squeeze(dist.log_prob(action)).item()
                action = action.cpu().detach().numpy()
                state_value = torch.squeeze(state_value).item()
                # Execute action in environment
                next_obs, reward, done, _ = self.env.step(action)
                timesteps += 1
                total_reward += reward
                # Store transition
                self.buffer.store((obs, action, act_log_prob, state_value, reward, done))
                
                if timesteps % self.config.UPDATE_FREQ == 0:
                    if self.config.USE_GAE:
                        if done:
                            self.last_value = 0
                        else:
                            _, self.last_value = self.policy(torch.tensor(next_obs, dtype=torch.float).to(self.config.DEVICE))
                    self.update()
                    self.buffer.clear()
                    if self.callback:
                        with torch.no_grad():
                            avg_reward = self.callback.eval_policy(self.policy)
                            if avg_reward > best_score:
                                print(f"Previous best score: {best_score}, new best {avg_reward} ... saving model")
                                torch.save(self.policy.state_dict(), self.config.WEIGHTS_PATH + "policy.pth")
                                best_score = avg_reward
                
                obs = next_obs
            episodes += 1
            self.logger.add_scalar("train_env_reward", total_reward, episodes)
        return episodes, timesteps

    def update(self):
        for _ in range(self.config.EPOCHS_PER_UPDATE):
            obs, acts, act_log_probs, state_values, rewards, dones, batches = self.buffer.get_batches()
            if self.config.USE_GAE:
                advantages = compute_gae(rewards, state_values, dones, self.last_value, self.config.GAE_LAMBDA, self.config.GAMMA)
                returns = advantages + state_values
            else:
                returns = compute_discounted_returns(rewards, dones, self.config.GAMMA)
                advantages = returns - state_values

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            advantages = torch.tensor(advantages, dtype=torch.float).to(self.config.DEVICE)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.config.DEVICE)
    
            for batch in batches:
                batch_obs = torch.tensor(obs[batch], dtype=torch.float).to(self.config.DEVICE)
                batch_acts = torch.tensor(acts[batch], dtype=torch.float).to(self.config.DEVICE)
                batch_old_log_probs = torch.tensor(act_log_probs[batch], dtype=torch.float).to(self.config.DEVICE)

                action_probs, state_values = self.policy(batch_obs)
                dist = self.policy.get_action_dist(action_probs)
                curr_log_probs = dist.log_prob(batch_acts)

                ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                surr_one = advantages[batch] * ratios
                surr_two = torch.clamp(ratios, 1 - self.config.CLIP_RANGE, 1 + self.config.CLIP_RANGE) * advantages[batch]

                actor_loss = -torch.min(surr_one, surr_two).mean()
                critic_loss = F.mse_loss(torch.squeeze(state_values), returns[batch])

                entropies = dist.entropy()
                entropy_loss = -torch.mean(entropies)

                total_loss = actor_loss + self.config.VALUE_COEF * critic_loss + self.config.ENTROPY_COEF * entropy_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()