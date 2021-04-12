import torch
import numpy as np
import ray
import torch.nn.functional as F
from torch.optim import Adam

class PPO():

    def __init__(self, env_fn, agent_class):

        self.agent = agent_class(env_fn)
        self.policy_optim = Adam(self.agent.actor.parameters(), lr=0.0005)
        self.critic_optim = Adam(self.agent.critic.parameters(), lr=0.0005)
        self.gamma = 0.99
        self.clip_range = 0.2
        self.num_workers = 1
        self.num_eps = 10000

        agents = [ray.remote(agent_class) for _ in range(self.num_workers)]
        self.workers = [agent.remote(env_fn) for agent in agents]

    def train(self):
        
        for i in range(self.num_eps):
            print(f"Epoch {i+1}")
            self.sync_policies()
            
            # Evaluate agents
            rewards = ray.get([worker.evaluate.remote() for worker in self.workers])
            print(f"Mean reward in 10 eval episodes {np.mean(rewards)}")
            # collect xp
            buffer = ray.get([w.collect_rollout.remote(100) for w in self.workers])
            for _ in range(3):
                for sample in buffer[0].sample_batch(batch_size=20):
                    states, actions, returns, log_probs, advantages = sample
                    curr_dist = self.agent.get_action_dist(states)

                    curr_log_probs = curr_dist.log_prob(actions)

                    ratios = torch.exp(curr_log_probs - log_probs)
                    surr1 = advantages * ratios
                    surr2 = torch.clamp(ratios, 1 - 0.1, 1 + 0.1) * advantages

                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -torch.mean(curr_dist.entropy()) * 0.01

                    self.policy_optim.zero_grad()
                    (actor_loss + entropy_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=0.05)
      
                    self.policy_optim.step()

                    state_values = self.agent.get_state_value(states).squeeze()
                    critic_loss = F.mse_loss(state_values, returns)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=0.05)
                    self.critic_optim.step()

    def sync_policies(self):
        actor_params = ray.put(list(self.agent.actor.parameters()))
        critic_params = ray.put(list(self.agent.critic.parameters()))

        for worker in self.workers:
            worker.sync_params.remote(actor_params, critic_params=critic_params)