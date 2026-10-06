"""Frozen policy mean on prospective public-reference packets; no task rollouts."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
import train_height_comparison  # Sets the existing tools import path.
from nominal_packet_reference import reference, unit
from smoke_reward_training import weight_digest
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/equal_exposure_recovery_v1'


def run():
    unit(); p = json.loads((OUT/'proposal.json').read_text())
    result = json.loads((OUT/'round198_pair_review.json').read_text())
    assert result['verified'] and not result['diagnostic_recovery_gate_passed']
    x = np.zeros((15, 39), np.float32)
    x[:, 11] = np.repeat(np.array([.115, .16, .24, .3, .38])-.3, 3)
    x[:, 9] = np.tile([0., .4, .8], 5)
    r = reference(x); rows = []; inputs = {}
    for spec in p['controllers']['models']:
        prefix = Path(spec['prefix'])
        for suffix, key in [('.zip', 'checkpoint_sha256'), ('.pkl', 'normalization_sha256')]:
            file = prefix.with_suffix(suffix); assert sha(file) == spec['checkpoint'][key]
            inputs[str(file.relative_to(ROOT))] = sha(file)
        model = PPO.load(str(prefix.with_suffix('.zip')), device='cuda')
        with prefix.with_suffix('.pkl').open('rb') as f: norm = pickle.load(f)
        rms = (norm.obs_rms.mean.copy(), norm.obs_rms.var.copy(), norm.obs_rms.count)
        before = (model.num_timesteps, model._n_updates, weight_digest(model))
        normalized = norm.normalize_obs(r)
        model.policy.set_training_mode(False)
        with torch.no_grad():
            tensor = torch.as_tensor(normalized, device=model.device)
            mu = model.policy.get_distribution(tensor).distribution.mean
            reference_mu = model.policy.get_distribution(tensor.clone()).distribution.mean
            anchored_at_reference = mu-reference_mu
            assert torch.equal(anchored_at_reference, torch.zeros_like(mu))
            values = mu.cpu().numpy()
        for i, m in enumerate(values):
            rows.append(dict(model=spec['label'], commanded_height_m=float(r[i, 11]+.3), speed_command_m_s=float(r[i, 9]),
                raw_Gaussian_mean=m.tolist(), clipped_deterministic_action=np.clip(m, -1, 1).tolist(),
                max_abs_mean=float(abs(m).max()), wheel_common_mean=float((m[4]+m[5])/2),
                wheel_differential_mean=float((m[4]-m[5])/2), anchor_identity_at_same_input_exact=True))
        assert before == (model.num_timesteps, model._n_updates, weight_digest(model))
        np.testing.assert_array_equal(rms[0], norm.obs_rms.mean); np.testing.assert_array_equal(rms[1], norm.obs_rms.var)
        assert rms[2] == norm.obs_rms.count
    assert len(rows) == 45
    for file in (OUT/'proposal.json', OUT/'round198_pair_review.json', ROOT/'wheelleg_warp/nominal_packet_reference.py'):
        inputs[str(file.relative_to(ROOT))] = sha(file)
    atomic_json(OUT/'round199_nominal_mean_audit.json', dict(verified=True, round=199, static_reference_rows=45, rows=rows,
        source_sha256=sha(__file__), input_sha256=inputs, raw_reference_packet=r.tolist(),
        new_task_evaluations=0, training_updates=0,
        facts='Frozen networks have outputs on a symmetric public kinematic reference. mu(o)-mu(reference(o)) is exactlyzero '
              'when o=reference(o), using identical normalization/network. This algebra is not a new algorithm or physical certificate.',
        limits='Synthetic kinematic packets, not exact powered/contact dynamic equilibria, actual on-policy observations or a '
               'causal decomposition ofzero-task yaw bias. No pulse, action intervention, newpolicy or PPOupdate. '
               'Anchoring deterministic Gaussianmean does not remove stochastic exploration or filtered actuator memory.',
        next='200 deepreview should decide a source-first matched anchoredmean-vs-plain learning study; no oldpolicy correction replay '
             'or newtraining admitted by this static diagnostic. Fullpaper goal remains incomplete.'))
    print('PASS15 public-reference fixtures,45 frozen means/RMS/weights unchanged;0 task evaluations', flush=True)
    for label in [m['label'] for m in p['controllers']['models']]:
        selected = [r for r in rows if r['model'] == label]
        print(label, 'maxabsmean', max(r['max_abs_mean'] for r in selected),
              'maxabswheeldiff', max(abs(r['wheel_differential_mean']) for r in selected), flush=True)


if __name__ == '__main__':
    run()
