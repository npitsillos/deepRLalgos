import torch
import numpy as np

from copy import deepcopy

class SAC():
    
    def __init__(self, policy, env, logger, config, callback=None):
        self.policy = policy
        self.target_policy = deepcopy(self.policy)
        self.env = env
        self.logger = logger
        self.config = config
        self.callback = callback
    
    def learn(self):
        
        # Collect s, a, r, s', d and add to replay buffer
        # collect until enough for update

        # for however many updates
            # sample bacth from replay buffer
            # for q1 & q2 compute targets -> run self.policy.q1(s, a) same for q2
            # get target actions from pi self.policy(s) *** need target net *** on next state
            # r + gamma * (1-d) * (q)
            # ^^^ los for SAC Q values
            # run a, log_prob = policy.pi(s) and policy.q1(s, a) policy.q2(s, a) and take min
            pass