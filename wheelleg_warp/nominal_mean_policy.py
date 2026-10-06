"""Conditional reference mean, using SB3's existing Gaussian distribution paths."""
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution


class PairedMlpExtractor(MlpExtractor):
    def __init__(self, net_arch, activation_fn, device, eta):
        super().__init__(39, net_arch, activation_fn, device)
        self.eta = eta

    def forward_actor(self, features):
        current = self.policy_net(features[:, :39])
        reference = self.policy_net(features[:, 39:])
        return current-self.eta*reference

    def forward_critic(self, features):
        return self.value_net(features[:, :39])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class NominalMeanPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, eta=0, **kwargs):
        if type(eta) is not int or eta not in (0, 1):
            raise ValueError('Eta must be exactly0 or1')
        if observation_space.shape != (78,) or action_space.shape != (6,):
            raise ValueError('Policy requires paired78 storage and6 virtual actions')
        if kwargs.get('use_sde', False) or not kwargs.get('share_features_extractor', True):
            raise ValueError('Qualification uses shared flat features and diagonal Gaussian')
        self.eta = eta
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        if self.features_dim != 78:
            raise ValueError('Paired policy requires unchanged flat78 features')
        self.mlp_extractor = PairedMlpExtractor(self.net_arch, self.activation_fn, self.device, self.eta)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError('Only the registered diagonal Gaussian is supported')
        # Biasfree linear output makes W(h_current-eta*h_ref) the declared mean difference.
        self.action_net.register_parameter('bias', None)
        with torch.no_grad():
            self.action_net.weight.zero_()
        # Drop the removed bias from optimizer ownership; no update has occurred here.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data['eta'] = self.eta
        return data
