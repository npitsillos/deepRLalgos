import torch
import itertools
import numpy as np
import torch.nn.functional as F

from copy import deepcopy
from torch.optim import Adam

from drl_algos.utils import ReplayBuffer
from drl_algos.utils.utils import compute_gae, compute_discounted_returns

class SAC():
    
    def __init__(self, agent, env, logger, config, callback=None):
        self.agent = agent
        self.target_agent = deepcopy(self.agent)
        self.env = env
        self.logger = logger
        self.config = config
        self.callback = callback

        self.q_params = itertools.chain(self.agent.q1.parameters(), self.agent.q2.parameters())
        self.q_optim = Adam(self.q_params, lr=self.config.LEARNING_RATE)
        self.pi_optim =  Adam(self.agent.pi.parameters(), lr=self.config.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(self.config.SIZE, self.env.observation_space.shape, self.env.action_space.n)
    
    def learn(self):
        
        # Collect s, a, r, s', d and add to replay buffer
        # collect until enough for update
        timesteps = 0
        while timesteps < self.config.TOTAL_TIMESTEPS:
            
            obs = self.env.reset()
            done = False
            while not done:

                if self.config.NORM_OBS:
                    obs = (obs - obs.mean()) / (obs.std() + 1e-10)
                
                if timesteps > self.config.START_STEPS:
                    action, _ = self.agent.act(torch.tensor(obs, dtype=torch.float).to(self.config.DEVICE))
                else:
                    action = self.env.action_space.sample()
                
                next_obs, reward, done, _ = self.env.step(action)
                timesteps += 1
                self.replay_buffer.store((obs, action, next_obs, reward, done))

                obs = next_obs

                if timesteps >= self.config.UPDATE_AFTER and timesteps % self.config.UPDATE_FREQ == 0:
                    self.update()
    
    def update(self):

        for i in range(self.config.EPOCHS_PER_UPDATE):
            batch_obs, batch_next_obs, batch_acts, batch_rewards, batch_dones = self.replay_buffer.sample_batch()

            # compute targets for q functions
            # current q values
            q1 = self.agent.q1(batch_obs, batch_acts)
            q2 = self.agent.q2(batch_obs, batch_acts)

            with torch.no_grad():
                # target actions from current policy
                actions_curr, log_probs_curr = self.agent.pi(batch_next_obs)

                q1_target = self.target_agent.q1(batch_next_obs, actions_curr)
                q2_target = self.target_agent.q2(batch_next_obs, actions_curr)
                q_target = torch.min(q1_target, q2_target)

                targets = batch_rewards + self.config.GAMMA * (1 - batch_dones) * (q_target - self.config.ALPHA * log_probs_curr)

            loss_q1 = F.mse_loss(q1, targets)
            loss_q2 = F.mse_loss(q2, targets)

            total_loss = loss_q1 + loss_q2

            self.q_optim.zero_grad()
            total_loss.backward()
            self.q_optim.step()
            
            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False
            
            actions, log_probs = self.agent.pi(batch_obs)
            q1 = self.agent.q1(batch_obs, actions)
            q2 = self.agent.q2(batch_obs, actions)
            q = torch.min(q1, q2)

            # Entropy regularised policy loss
            loss = (q - self.config.ALPHA * log_probs).mean()

            self.pi_optim.zero_grad()
            loss.backward()
            self.pi_optim.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Perform polyak averaging
            with torch.no_grad():
                for p, p_target in zip(self.agent.parameters(), self.target_agent.parameters()):
                    p_target.data.mul_(self.config.POLYAK)
                    p_target.data.add_((1 - self.config.POLYAK) * p_target.data)