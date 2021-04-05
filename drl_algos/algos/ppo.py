import gym
import torch
import numpy as np
import torch.nn.functional as F

from torch.optim import Adam

from drl_algos.utils import RolloutBuffer
from drl_algos.utils.utils import compute_gae, compute_discounted_returns

class PPO():

    def __init__(self, agent, env, logger, config, callback=None):
        self.agent = agent
        self.env = env
        self.logger = logger
        self.config = config
        self.callback = callback

        self.actor_optimizer = Adam(self.agent.actor.parameters(), lr=self.config.LEARNING_RATE)
        self.critic_optimizer = Adam(self.agent.critic.parameters(), lr=self.config.LEARNING_RATE)
        self.buffer = RolloutBuffer(self.config.BATCH_SIZE, use_lstm=self.config.USE_LSTM)

        self.epochs = 0
        self.timesteps = 0

    def learn(self):

        best_score = self.env.reward_range[0]
        episodes = 0
        while self.timesteps < self.config.TOTAL_TIMESTEPS:
            if self.config.USE_LSTM:
                h_out = (torch.zeros([1, 1, 32], dtype=torch.float).to(self.config.DEVICE),
                            torch.zeros([1, 1, 32], dtype=torch.float).to(self.config.DEVICE))
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                
                if self.config.NORM_OBS:
                    obs = (obs - obs.mean()) / (obs.std() + 1e-10)
                
                # Feedforward observation to get action_probs and state_value
                policy_obs = torch.tensor(obs, dtype=torch.float).to(self.config.DEVICE)
                if self.config.USE_LSTM:
                    h_in = h_out
                    action, act_log_prob, state_value, h_out = self.agent.act(policy_obs, h_in=h_in)
                else:
                    action, act_log_prob, state_value = self.agent.act(policy_obs)
                
                # Execute action in environment
                next_obs, reward, done, _ = self.env.step(action)
                self.timesteps += 1
                total_reward += reward

                # Store transition
                if self.config.USE_LSTM:
                    self.buffer.store((obs, action, act_log_prob, state_value, reward, done, h_in, h_out))    
                else:
                    self.buffer.store((obs, action, act_log_prob, state_value, reward, done))
                
                if self.timesteps % self.config.UPDATE_FREQ == 0:
                    if self.config.USE_GAE:
                        if done:
                            self.last_value = 0
                        else:
                            if self.config.USE_LSTM:
                                self.last_value = self.agent.get_state_value(torch.tensor(next_obs, dtype=torch.float).to(self.config.DEVICE), h_in=h_out)
                            else:
                                self.last_value = self.agent.get_state_value(torch.tensor(next_obs, dtype=torch.float).to(self.config.DEVICE))
                    self.update()
                    self.buffer.clear()
                    if self.callback:
                        with torch.no_grad():
                            performance_metric = self.callback.eval_policy(self.agent)
                            if performance_metric > best_score or performance_metric == best_score:
                                print(f"Previous best score: {best_score}, new best {performance_metric} ... saving model")
                                self.agent.save_checkpoint(self.config.WEIGHTS_PATH)
                                best_score = performance_metric
                
                obs = next_obs
            episodes += 1
            self.logger.add_scalar("train_env_reward", total_reward, episodes)
        return episodes, self.timesteps

    def update(self):
        update_avg_loss = 0
        for _ in range(self.config.EPOCHS_PER_UPDATE):
            if self.config.USE_LSTM:
                obs, acts, act_log_probs, state_values, rewards, dones, h_ins, h_outs, batches = self.buffer.get_batches()
            else:
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
            total_loss = 0
            policy_epoch_loss = 0
            entropy_epoch_loss = 0
            actor_epoch_loss = 0
            critic_epoch_loss = 0
            for batch in batches:
                batch_obs = torch.tensor(obs[batch], dtype=torch.float).to(self.config.DEVICE)
                batch_acts = torch.tensor(acts[batch], dtype=torch.float).to(self.config.DEVICE)
                batch_old_log_probs = torch.tensor(act_log_probs[batch], dtype=torch.float).to(self.config.DEVICE)
                
                if self.config.USE_LSTM:
                    batch_h_ins = (h_ins[batch][0][0].detach(), h_ins[batch][0][1].detach())
                    curr_log_probs, entropies  = self.agent.act(batch_obs, actions=batch_acts, h_in=batch_h_ins)
                else:
                    curr_log_probs, entropies = self.agent.act(batch_obs, actions=batch_acts)
                
                ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                surr_one = advantages[batch] * ratios
                surr_two = torch.clamp(ratios, 1 - self.config.CLIP_RANGE, 1 + self.config.CLIP_RANGE) * advantages[batch]

                policy_loss = -torch.min(surr_one, surr_two).mean()

                entropy_loss = -torch.mean(entropies)
                actor_loss = policy_loss + self.config.ENTROPY_COEF * entropy_loss
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.config.USE_LSTM:
                    state_values = self.agent.get_state_value(batch_obs, h_in=batch_h_ins)
                else:
                    state_values = self.agent.get_state_value(batch_obs)
                critic_loss = F.mse_loss(torch.squeeze(state_values), returns[batch])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                total_loss = actor_loss + self.config.VALUE_COEF * critic_loss + self.config.ENTROPY_COEF * entropy_loss

                total_loss += total_loss.item()
                policy_epoch_loss += policy_loss.item()
                entropy_epoch_loss += entropy_loss.item()
                actor_epoch_loss += actor_loss.item()
                critic_epoch_loss += critic_loss.item()

            update_avg_loss += total_loss
            self.epochs += 1
            self.logger.add_scalar("total_loss", total_loss, self.epochs)
            self.logger.add_scalar("policy_loss", policy_epoch_loss, self.epochs)
            self.logger.add_scalar("entropy_loss", entropy_epoch_loss, self.epochs)
            self.logger.add_scalar("actor_loss", actor_epoch_loss, self.epochs)
            self.logger.add_scalar("critic_loss", critic_epoch_loss, self.epochs)

        self.logger.add_scalar("avg_epoch_loss", update_avg_loss / self.config.EPOCHS_PER_UPDATE, self.timesteps)