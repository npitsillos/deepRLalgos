import gym
import torch
import numpy as np

from drl_algos.algos import SAC
from drl_algos.networks import critics
from drl_algos.networks import policies
from drl_algos.data import ReplayBuffer, MdpPathCollector
from drl_algos.trainers import BatchRLAlgorithm
from drl_algos import utils

config = {
    ### Generic experimental parameters, shared by all
    "env": "Pendulum-v0",
    "env_train_kwargs": {},
    "env_eval_kwargs": {},
    "max_path_len": 200,

    "device": "cuda:0",
    "seed": 500,

    # This is what the if else uses
    "algorithm": "SAC"

    ### Everything below here is specific to the algorithm
    ### The script will fill the rest in
    ### There should be a default that can be merged with this to
    ### fill in missing parameters. The config can be minimal for
    ### the user, but when it saves it will show the entire config
    ### used to set up the experiments. If default values get
    ### changed the config will override it. If new keys are added
    ### without adding a default to the config, these will be
    ### missing.

    "algorithm_kwargs" : {
        "tau" : .001,
    }
    "critic": "MlpCritic"
    "critic_kwargs": {
        "hidden_sizes": [64,64],
    }
    "policy": "MlpGaussianPolicy"
    "policy_kwargs": {
        "activation": torch.relu,
    }

    # I think we should settle on a default implementation for each
    # algorithm. There shouldn't be any need for eretrainer if we
    # ensure replay buffer self contained
    "trainer": "BatchRLAlgorithm" # remove this
    "trainer_kwargs": { # keep this
        "num_epoch": 20,
    }

    # Same as above
    "path_collector": "MdpPathCollector", # remove this
    "path_collector_kwargs": {}, # probably remove this

    # Keep this
    "replay_buffer": "EpisodicReplayBuffer"
    "replay_buffer_kwargs": {},
}

def create_critic(config, obs_dim, action_inputs, model):
    # Todo - should decide output dim based on Box or Discrete action space
    fn = getattr(critics, config.get("critic"))
    if model is None:
        kwargs = config.get("critic_kwargs").update({
            "input_size": obs_dim+action_inputs,
            "output_size": 1,
        })
    else:
        kwargs = config.get("critic_kwargs").update({
            "input_size": model.latent_size+action_inputs,
            "output_size": 1,
        })
    return fn(**kwargs)

def create_policy(config, obs_dim, action_dim, model):
    fn = getattr(policies, config.get("policy"))
    if model is None:
        kwargs = config.get("policy_kwargs").update({
            "input_size": obs_dim,
            "output_size": action_dim,
        })
    else:
        kwargs = config.get("policy_kwargs").update({
            "input_size": model.latent_size,
            "output_size": action_dim,
        })
    return fn(**kwargs)

def create_replay_buffer(config, env):
    fn = getattr(replay_buffer, config.get("replay_buffer"))
    return fn(env=env, **config.get("replay_buffer_kwargs"))

def create_path_collector(config, env, policy, model):
    fn = getattr(replay_buffer, config.get("path_collector"))
    if model is None:
        return fn(env=env, policy=policy, **config.get("path_collector_kwargs"))
    return fn(env=env, policy=policy, model=model, **config.get("path_collector_kwargs"))


def get_model(config, obs_dim, action_dim):
    if config.get("model") is None:
        return None
    fn = getattr(models, config.get("model"))
    kwargs = config.get("model_kwargs").update({
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    })
    return fn(**kwargs)

def make_deterministic(policy):
    return policies.MakeDeterministic(policy)

def run(config):
    experiment = build_experiment(config)
    experiment.run()

    builder = ExperimentBuilder(config)

    if builder.algorithm_name.upper() == "SAC":
        experiment =

    if algorithm.upper() == "SAC":
        # Create networks
        qf1 = builder.create_q_critic()
        qf2 = builder.create_q_critic()
        target_qf1 = builder.create_q_critic()
        target_qf2 = builder.create_q_critic()
        policy = builder.create_policy()
        eval_policy = make_deterministic(policy)

        # Create algorithm
        algorithm = builder.create_algorithm(
            env=env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **config.get("algorithm_kwargs")
        )

        # Create replay buffer
        replay_buffer = builder.create_replay_buffer()

        # Create exploration and evaluation path collectors
        expl_path_collector, eval_path_collector = builder.create_path_collectors(policy, eval_policy)

        # Create trainer
        trainer = trainers.BatchRLAlgorithm(
                      algorithm=algorithm,
                      exploration_env=env,
                      evaluation_env=eval_env,
                      exploration_path_collector=expl_path_collector,
                      evaluation_path_collector=eval_path_collector,
                      replay_buffer=replay_buffer,
                      max_path_length=max_path_len,
                      **config.get("trainer_kwargs")
                  )

    elif algorithm.upper() == "SACDreamer":
        # Get network functions and kwargs
        critic_fn = getattr(critics, config.get("critic"))
        critic_kwargs = config.get("critic_kwargs").update({
            "input_size": obs_dim+action_dim,
            "output_size": 1,
        })
        policy_fn = getattr(critics, config.get("policy"))
        policy_kwargs = config.get("policy_kwargs").update({
            "input_size": obs_dim,
            "output_size": action_dim,
        })

        # Create networks
        qf1 = critic_fn(**critic_kwargs)
        qf2 = critic_fn(**critic_kwargs)
        target_qf1 = critic_fn(**critic_kwargs)
        target_qf2 = critic_fn(**critic_kwargs)
        policy = policy_fn(**policy_kwargs)
        eval_policy = policies.MakeDeterministic(policy)

        # Create replay buffer
        replay_buffer_fn = getattr(replay_buffer, config.get("replay_buffer"))
        replay_buffer = replay_buffer_fn(env, **config.get("critic_kwargs"))

        # Create exploration and evaluation path collectors
        expl_path_collector = path_collector.MdpPathCollector(
                                  env,
                                  policy,
                              )
        eval_path_collector = path_collector.MdpPathCollector(
                                  eval_env,
                                  eval_policy,
                              )

        # Create algorithm
        algorithm = algos.SAC(
            env=env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **config.get("algorithm_kwargs")
        )

        # Create trainer
        trainer = trainers.BatchRLAlgorithm(
                      algorithm=algorithm,
                      exploration_env=env,
                      evaluation_env=eval_env,
                      exploration_path_collector=expl_path_collector,
                      evaluation_path_collector=eval_path_collector,
                      replay_buffer=replay_buffer,
                      max_path_length=max_path_len,
                      **config.get("trainer_kwargs")
                  )

    # This should be set dynamically
    # Set up logging
    utils.setup_logger('walker/')
    print()

    # Move onto GPU and start training
    trainer.to(device)
    trainer.train()
