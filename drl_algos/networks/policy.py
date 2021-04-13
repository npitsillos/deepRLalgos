class PolicyNew:
    """
        Base class for all discrete Actor networks.

        :param features_in: Size of latent features.
        :param action_dim: Action output dimension.
        :param is_continuous: Whether action is continuous.
    """
    def __init(self, actor, dist):
        self.actor = actor
        self.dist = dist

    def get_action(self, state):
        raise NotImplementedError

    def reset(self):
        pass

class StochasticPolicyNew(PolicyNew):

    def get_action(self, obs):
        obs = utils.to_tensor(obs[None], self.device)
        if self.actor.is_continuous:
            means, std = self.actor(obs)
            dist = self.dist(means, std)
        else:
            logits = self.actor(obs)
            dist = self.dist(logits)
        actions = dist.sample()
        actions = utils.to_numpy(actions)
        return actions[0, :], {}


class DeterministicPolicyNew(StochasticPolicyNew):

    def __init__(self, policy):
        super().__init__()
        self.device = policy.device
        self._policy = policy

    def forward(self, *args, **kwargs):
        dist = self._policy(*args, **kwargs)
        return Delta(dist.mle_estimate())