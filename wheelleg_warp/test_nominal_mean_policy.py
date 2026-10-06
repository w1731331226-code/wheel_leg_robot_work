"""Likelihood/gradient/init/reload admission without optimizer or task updates."""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
import stable_baselines3.common.policies as sb3_policies
import stable_baselines3.common.distributions as sb3_distributions
import stable_baselines3.common.torch_layers as sb3_layers
from nominal_mean_policy import NominalMeanPolicy
from dashboard.live_env import atomic_json
from review_yaw_sector import ROOT, sha

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_mean_matched_learning_v1'


class NoStepEnv(gym.Env):
    observation_space = Box(-np.inf, np.inf, (78,), dtype=np.float32)
    action_space = Box(-1, 1, (6,), dtype=np.float32)
    def reset(self, *, seed=None, options=None):
        return np.zeros(78, np.float32), {}
    def step(self, action):
        raise AssertionError('Source qualification must not execute an environment step')


def make(eta, p):
    torch.manual_seed(202001)
    policy = NominalMeanPolicy(NoStepEnv.observation_space, NoStepEnv.action_space,
        lambda _: .0003, eta=eta, net_arch=[64, 64]).to('cuda')
    with torch.no_grad(): policy.log_std.copy_(torch.tensor(p['initial_log_std'], device='cuda'))
    policy.set_training_mode(False)
    return policy


def check():
    torch.set_num_threads(1)
    p = json.loads((OUT/'proposal.json').read_text()); plain, anchor = make(0, p), make(1, p)
    x = torch.randn(128, 78, device='cuda'); actions = torch.linspace(-1.4, 1.4, 128*6, device='cuda').reshape(128, 6)
    assert all(torch.equal(a, anchor.state_dict()[name]) for name, a in plain.state_dict().items())
    counts = [sum(v.numel() for v in policy.parameters()) for policy in (plain, anchor)]; assert counts[0] == counts[1]
    for policy in (plain, anchor):
        assert policy.action_net.bias is None
        owned = {id(v) for group in policy.optimizer.param_groups for v in group['params']}
        assert owned == {id(v) for v in policy.parameters()} and not policy.optimizer.state
        assert policy.mlp_extractor.policy_net[0].in_features == policy.mlp_extractor.value_net[0].in_features == 39
        assert not policy.get_distribution(x).distribution.mean.any()
    torch.manual_seed(202002); a0, v0, l0 = plain(x)
    torch.manual_seed(202002); a1, v1, l1 = anchor(x)
    assert torch.equal(a0, a1) and torch.equal(v0, v1) and torch.equal(l0, l1)
    # Nonzero synthetic weights exercise the constraint and gradient paths, not a training update.
    with torch.no_grad():
        weights = torch.randn_like(plain.action_net.weight)*.03
        plain.action_net.weight.copy_(weights); anchor.action_net.weight.copy_(weights)
    identical = torch.cat([x[:, :39], x[:, :39]], dim=1)
    assert not anchor.get_distribution(identical).distribution.mean.any()
    changed_ref = x.clone(); changed_ref[:, 39:] *= -2
    assert torch.equal(anchor.predict_values(x), anchor.predict_values(changed_ref))
    errors = []
    for policy in (plain, anchor):
        distribution = policy.get_distribution(x).distribution
        current = policy.mlp_extractor.policy_net(x[:, :39])
        reference = policy.mlp_extractor.policy_net(x[:, 39:])
        manual_mean = policy.action_net(current)-policy.eta*policy.action_net(reference)
        torch.testing.assert_close(distribution.mean, manual_mean, atol=2e-7, rtol=2e-6)
        std = policy.log_std.exp()
        manual_logprob = (-.5*((actions-distribution.mean)/std)**2-policy.log_std-.5*np.log(2*np.pi)).sum(1)
        manual_entropy = (policy.log_std+.5*np.log(2*np.pi*np.e)).sum().expand(128)
        value, logprob, entropy = policy.evaluate_actions(x, actions)
        torch.testing.assert_close(logprob, manual_logprob, atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(entropy, manual_entropy, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(value, policy.predict_values(x), atol=0, rtol=0)
        old = logprob.detach().clone(); new = policy.evaluate_actions(x, actions)[1]
        assert torch.equal(torch.exp(new-old), torch.ones_like(new))
        sampled, sampled_value, sampled_logprob = policy(x)
        evaluated_value, evaluated_logprob, _ = policy.evaluate_actions(x, sampled)
        assert torch.equal(sampled_value, evaluated_value) and torch.equal(sampled_logprob, evaluated_logprob)
        deterministic = policy.predict(x.detach().cpu().numpy(), deterministic=True)[0]
        np.testing.assert_allclose(deterministic, np.clip(distribution.mean.detach().cpu().numpy(), -1, 1), atol=2e-7, rtol=2e-6)
        errors.append(float(abs(logprob-manual_logprob).max().detach()))
    z = x.detach().clone().requires_grad_(True)
    gradient = torch.autograd.grad(anchor.get_distribution(z).distribution.mean.sum(), z)[0]
    assert gradient[:, :39].abs().max() > 0 and gradient[:, 39:].abs().max() > 0
    zp = x.detach().clone().requires_grad_(True)
    plain_gradient = torch.autograd.grad(plain.get_distribution(zp).distribution.mean.sum(), zp)[0]
    assert not plain_gradient[:, 39:].any()
    anchor.zero_grad(set_to_none=True)
    anchor.evaluate_actions(x, actions)[1].mean().backward()
    analytic = float(anchor.action_net.weight.grad[0, 0]); initial = float(anchor.action_net.weight[0, 0].detach()); eps = .001
    with torch.no_grad():
        anchor.action_net.weight[0, 0] = initial+eps; plus = float(anchor.evaluate_actions(x, actions)[1].mean())
        anchor.action_net.weight[0, 0] = initial-eps; minus = float(anchor.evaluate_actions(x, actions)[1].mean())
        anchor.action_net.weight[0, 0] = initial
    finite = (plus-minus)/(2*eps)
    np.testing.assert_allclose(analytic, finite, atol=.001, rtol=.01)
    assert not anchor.optimizer.state  # No optimizer.step/PPO.learn.
    with tempfile.TemporaryDirectory() as tmp:
        for policy in (plain, anchor):
            file = Path(tmp)/f'policy{policy.eta}.pt'; policy.save(file)
            other = NominalMeanPolicy.load(file, device='cuda')
            assert other.eta == policy.eta and other.action_net.bias is None
            for name, value in policy.state_dict().items(): assert torch.equal(value, other.state_dict()[name])
            assert torch.equal(policy.evaluate_actions(x, actions)[1], other.evaluate_actions(x, actions)[1])
        agent = PPO(NominalMeanPolicy, NoStepEnv(), device='cuda', seed=202003, n_steps=8, batch_size=8,
                    policy_kwargs=dict(eta=1, net_arch=[64, 64]))
        agent.save(Path(tmp)/'agent.zip'); loaded = PPO.load(Path(tmp)/'agent.zip', device='cuda')
        assert loaded.policy.eta == 1 and loaded.num_timesteps == agent.num_timesteps == 0
        assert loaded._n_updates == agent._n_updates == 0
        assert torch.equal(agent.policy.get_distribution(x).distribution.mean, loaded.policy.get_distribution(x).distribution.mean)
        assert not loaded.policy.optimizer.state
    atomic_json(OUT/'round202_policy_unit.json', dict(verified=True, matched_parameter_counts=counts,
        initial_state_functions_distribution_and_seeded_actions_exact=True, nonzero_weights_reference_mean_exactzero=True,
        critic_ignores_reference=True, shared_actor_reference_gradient_nonzero=True, plain_reference_gradient_zero=True,
        manual_normal_logprob_entropy_maxerrors=errors, original_PPO_ratio_identity_exact=True,
        independent_mean_form_close=True, finite_difference=dict(analytic=analytic, finite=finite, eps=eps),
        all_SB3_distribution_paths_consistent=True, optimizer_has_exact_live_parameters_no_orphan_bias=True,
        policy_and_PPO_reload_eta_weights_logprob_exact=True,
        installed_SB3_sha256=dict(policies=sha(Path(sb3_policies.__file__)), distributions=sha(Path(sb3_distributions.__file__)),
                                 torch_layers=sha(Path(sb3_layers.__file__))),
        synthetic_parameter_fixture_not_trained_policy=True, optimizer_updates=0, training_steps=0, task_evaluations=0,
        limits='Synthetic inputs/weights, autodiff and initialization/reload only. No PPOupdate, rollout, Adamstate mutation, '
               'or method-performance/safety validation. Actual24kengineering still required.'))
    print('PASS matched Gaussian/logprob/entropy/ratio/gradient/init/reload;0 optimizer/task updates', flush=True)
    print('parameters', counts, 'manual errors', errors, 'gradient', analytic, finite, flush=True)


if __name__ == '__main__':
    check()
