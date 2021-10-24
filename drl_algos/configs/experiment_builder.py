from drl_algos.algos.SAC import SAC, SACDreamer
from drl_algos import utils

"""
The only things the user really wants to be able to change is:
    the env and its kwargs
    the algorithm and its kwargs
    the network and its kwargs
    the replay buffer and its kwargs

    the path collectors should be decided based on the trainer
    the trainer should be decided based on the algorithm
        - Off policy should use batch RL
            - there may be equivalents for ere etc, but ignore this for now
        - On policy shold use its own thing (leave for anyone interested to implement)
        - There should be equivalents for training with a model

    so the whole process to creating an experiment is to
        - create the environments and get the dimensions
        - seed the experiment
        - create the model if one provided
            - if/else to call correct init
            - creates networks and returns the model
        - create the algorithm
            - if/else to call correct init
            - create networks and returns the algorithm
        - create the trainer
            - if/else to call correct init
                - determined by on vs off-policy and whether there is a model
            - creates replay buffer

    if model_name=="dreamer":
        model = builder.build_dreamer()
"""



def build_trainer(config):


    builder = ExperimentBuilder(config)

def build_trainer(config):
    builder = ExperimentBuilder(config)




class ExperimentBuilder(object):

    def __init__(self, config):
        self.config = config
        self.initialise_experiment()
        self.build_experiment()

    def run_experiment(self):
        # Todo add better logging
        utils.setup_logger('walker/')
        print()

        self.trainer.to(self.device)
        self.trainer.train()

    def initialise_experiment(self):
        """Generic initilisation shared by all experiments."""
        # Get algorithm and optional model parameters
        self.algorithm_fn = config.get("algorithm")
        self.model_fn = config.get("model")

        # Get environment parameters
        self.env_name = config.get("env")
        self.env_train_kwargs = config.get("env_train_kwargs")
        self.env_eval_kwargs = config.get("env_eval_kwargs")
        self.max_path_len = config.get("max_path_len")

        # Get seed and device parameters
        self.seed = config.get("seed")
        self.device = config.get("device")

        # build training and evaluation environments
        self.env = gym.make(self.env_name, **self.env_train_kwargs).env
        self.eval_env = gym.make(self.env_name, **self.env_eval_kwargs).env

        # Seed experiment
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.eval_env.seed(self.seed+1)

        # Get env dimensions
        self.obs_dim = self.input_dim = self.env.observation_space.low.size
        self.action_dim = self.env.action_space.low.size

        # If there is a model, build it
        self.model = None
        if self.model_name is not None:
            self.model_kwargs = config.get("model_kwargs")
            self.model_kwargs.update({
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            })
            self.model = self.model_fn(**self.model_kwargs)
            self.input_dim = self.model.latent_size

    def build_experiment(self):
        if self.algorithm_fn.upper() == "SAC":
            self.build_sac_experiment()
        else:
            raise ValueError("Algorithm name not recognised.")

    def build_sac_algorithm(self):
        # Build networks
        qf1 = self.build_q_critic()
        qf2 = self.build_q_critic()
        target_qf1 = self.build_q_critic()
        target_qf2 = self.build_q_critic()
        policy = self.build_policy()
        eval_policy = make_deterministic(policy)

        # Build algorithm
        algorithm = SAC(
            env=env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **self.config.get("algorithm_kwargs")
        )
        return algorithm

    def build_batch_trainer(self, algorithm, policy, eval_policy):
        # build replay buffer
        replay_buffer = self.build_replay_buffer()

        # build exploration and evaluation path collectors
        expl_path_collector, eval_path_collector = builder.build_path_collectors(policy, eval_policy)

        # build trainer
        trainer = trainers.BatchRLAlgorithm(
            algorithm=algorithm,
            exploration_env=self.env,
            evaluation_env=self.eval_env,
            exploration_path_collector=expl_path_collector,
            evaluation_path_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=self.max_path_len,
            **self.config.get("trainer_kwargs"),
        )
        return trainer

    def build_critic(self, input_size):
        fn = self.config.get("critic")
        kwargs = self.config.get("critic_kwargs")
        kwargs.update({
            "input_size": input_size,
            "output_size": 1,
        })
        return fn(**kwargs)

    def build_q_critic(self):
        return build_critic(self.input_dim + self.action_dim)

    def build_v_critic(self):
        return build_critic(self.input_dim)

    def build_policy(self):
        fn = self.config.get("policy")
        kwargs = self.config.get("policy_kwargs")
        kwargs.update({
            "input_size": self.input_dim,
            "output_size": self.action_dim,
        })
        return fn(**kwargs)

    def build_replay_buffer(self):
        fn = self.config.get("replay_buffer")
        kwargs = self.config.get("replay_buffer_kwargs")
        kwargs.update({
            "env": self.env,
            "max_path_len": self.max_path_len,
        })
        return fn(**kwargs)

    def build_path_collectors(self, policy, eval_policy):
        fn = self.config.get("path_collector")
        kwargs = self.config.get("path_collector_kwargs")
        kwargs.update({
            "env": self.env,
            "policy": policy,
            "model": self.model,
        })
        train_path_collector = fn(**kwargs)
        kwargs.update({
            "env": self.eval_env,
            "policy": eval_policy,
        })
        eval_path_collector = fn(**kwargs)
        return train_path_collector, eval_path_collector
