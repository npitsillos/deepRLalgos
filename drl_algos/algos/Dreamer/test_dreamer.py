import gym
import torch
import numpy as np

from drl_algos.algos.Dreamer.dreamer import DreamerBehaviour
from drl_algos.networks import critics, policies, models
from drl_algos.data import ModelPathCollector2, EpisodicReplayBuffer, Logger2
from drl_algos.trainers import DreamerAlgorithm2 as Trainer
from drl_algos import utils

"""
Notes:
    - Definietly something going on with the Dreamer policy as it is always
    choosing actions around 0 for evaluation. I reckon its a problem with the
    policy loss which seems to stay consistently low.
    - Need to check if model is actually learning, say comparing real
    trajectories to imagined ones.
        - Lets train a SAC agent on pendulum then save the entire replay buffer
        - Then lets train the model but hold back some trajectories for
        evaluation
        - Lets periodically compare the real trajectories to the imagined ones
        - Gives us certainty that the model is actually learning and performing
        well. If the model isn't accurate then nothing else will work
        - Side note - could input normalisation/standardisation be a concern
        here since we've moved from pixel values to low state (which may not
        be normalized)?
    - After confirming the model is working we can look into the DRL algorithm
        - Might even be worth training it using real data rather than the model
        at least to confirm that it is working
        - Then it might be worth trying to train it with a pretrained model
        - Then finally moving onto the harder case of learning both at the same
        time
"""

torch.manual_seed(0)
np.random.seed(0)

# Device for the networks
DEVICE = "cuda:0"

# Environment info
ENV_NAME = "Pendulum-v0"

# Hyperparams
BUFFER_SIZE = 50000
MAX_PATH_LEN = 200
BATCH_SIZE = 2
SEQ_LEN = 5
ACTOR_HIDDEN = [64,64]

# Create and seed envs
env = gym.make(ENV_NAME).env
eval_env = gym.make(ENV_NAME).env
env.seed(0)
eval_env.seed(1)

# Env dimensions
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

# Create model
model = models.MlpDreamer(obs_dim, action_dim)

# Create actors
policy = policies.MlpGaussianPolicy(
             hidden_sizes=ACTOR_HIDDEN,
             input_size=model.latent_size,
             output_size=action_dim,
         )
eval_policy = policies.MakeDeterministic(policy)

# Create critics
critic = critics.MlpCritic2(
    hidden_sizes=ACTOR_HIDDEN,
    input_size=model.latent_size,
    output_size=1,
    base_kwargs={},
)

target_critic = critics.MlpCritic2(
    hidden_sizes=ACTOR_HIDDEN,
    input_size=model.latent_size,
    output_size=1,
    base_kwargs={},
)

# Create DRL algorithm
algorithm = DreamerBehaviour(policy, critic, target_critic)

# Create buffer
replay_buffer = EpisodicReplayBuffer(
        BUFFER_SIZE,
        env,
        MAX_PATH_LEN
    )

# Create exploration and evaluation path collectors
expl_path_collector = ModelPathCollector2(
                          env,
                          policy,
                          model,
                          MAX_PATH_LEN
                      )

eval_path_collector = ModelPathCollector2(
                          eval_env,
                          eval_policy,
                          model,
                          MAX_PATH_LEN
                      )

# Create training routine
logger = Logger2("dreamer-pendulum")
trainer = Trainer(
              model=model,
              algorithm=algorithm,
              exploration_env=env,
              evaluation_env=eval_env,
              exploration_path_collector=expl_path_collector,
              evaluation_path_collector=eval_path_collector,
              replay_buffer=replay_buffer,
              logger=logger,
              batch_size=50,
              sequence_len=50,
              imagination_horizon=15,
              num_epochs=100,
              num_eval_eps_per_epoch=10,
              num_expl_steps_per_train_loop=1000,
              num_trains_per_train_loop=250,
          )
trainer.to("cuda:0")
trainer.train()
