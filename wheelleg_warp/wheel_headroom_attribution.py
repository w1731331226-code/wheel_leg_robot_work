"""Read-only wheel-box attribution on the registered frozen-policy contexts."""
import gc
import json
import math
from pathlib import Path

import numpy as np
import warp as wp
from stable_baselines3 import PPO

import fixed_steering_leg as parent
from native.controller import D, V6, command_bounds, control_physical_nominal
from review_yaw_sector import ROOT, sha, success
from dashboard.live_env import atomic_json

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/wheel_headroom_attribution_v1'
FIELDS = ['attempted', 'valid', 'invalid', 'filtered_energy', 'accepted_energy',
          'wheel_loss_energy', 'hip_loss_energy', 'rejected_abs', 'wheel_loss_abs',
          'hip_loss_abs', 'loss_cross_energy', 'max_decomposition_error',
          'max_acceptance_error', 'max_lambda_order_excess', 'max_nominal_box_excess',
          'max_hip_loss', 'sum_global_lambda', 'sum_wheel_lambda',
          'max_counterfactual_box_excess']


@wp.kernel
def record(qvel: wp.array2d[float], ids: wp.array[int], memory: wp.array2d[D],
           active: wp.array[int], diagnostic: wp.array2d[D], gains: wp.array2d[D],
           tracking: wp.array[int], stats: wp.array2d[D]):
    w = wp.tid()
    if active[w] == 0 or tracking[w] == 0:
        return
    stats[w, 0] += D(1)
    if diagnostic[w, 14] != D(0):
        stats[w, 2] += D(1)
        return
    speeds = V6(); scales = V6()
    valid = True
    for j in range(6):
        speeds[j] = D(qvel[w, ids[4+j]])
        scales[j] = gains[w, j]
        if not wp.isfinite(speeds[j]) or not wp.isfinite(scales[j]) or scales[j] <= D(0):
            valid = False
    bounds = command_bounds(speeds, scales)
    r = D(.3) * memory[w, 18]; lam = diagnostic[w, 12]; lw = D(1)
    if not wp.isfinite(r) or not wp.isfinite(lam) or lam < D(0) or lam > D(1):
        valid = False
    box_excess = D(0)
    for j in range(4, 6):
        base = diagnostic[w, j]; residual = r
        if j == 5:
            residual = -r
        if not wp.isfinite(base) or not wp.isfinite(diagnostic[w, 6+j]):
            valid = False
        box_excess = wp.max(box_excess, wp.abs(base)-bounds[j])
        if residual > D(0):
            lw = wp.min(lw, (bounds[j]-base)/residual)
        elif residual < D(0):
            lw = wp.min(lw, (-bounds[j]-base)/residual)
    if not valid:
        stats[w, 2] += D(1)
        return
    lw = wp.clamp(lw, D(0), D(1))
    accepted = diagnostic[w, 10]; wheel_loss = (D(1)-lw)*r
    hip_loss = (lw-lam)*r; rejected = r-accepted
    stats[w, 1] += D(1)
    stats[w, 3] += r*r; stats[w, 4] += accepted*accepted
    stats[w, 5] += wheel_loss*wheel_loss; stats[w, 6] += hip_loss*hip_loss
    stats[w, 7] += wp.abs(rejected); stats[w, 8] += wp.abs(wheel_loss)
    stats[w, 9] += wp.abs(hip_loss); stats[w, 10] += D(2)*wheel_loss*hip_loss
    stats[w, 11] = wp.max(stats[w, 11], wp.abs(rejected-wheel_loss-hip_loss))
    stats[w, 12] = wp.max(stats[w, 12], wp.max(wp.abs(accepted-lam*r), wp.abs(diagnostic[w, 11]+lam*r)))
    stats[w, 13] = wp.max(stats[w, 13], lam-lw)
    stats[w, 14] = wp.max(stats[w, 14], box_excess)
    stats[w, 15] = wp.max(stats[w, 15], wp.abs(hip_loss))
    stats[w, 16] += lam; stats[w, 17] += lw
    for j in range(4, 6):
        residual = r
        if j == 5:
            residual = -r
        stats[w, 18] = wp.max(stats[w, 18], wp.abs(diagnostic[w, j]+lw*residual)-bounds[j])


def attach(raw, stats, tracking):
    raw._headroom_buffers = (stats, tracking)
    wait, reset = raw.step_wait, raw.reset
    def reset_all():
        result = reset(); stats.zero_(); tracking.fill_(1)
        return result
    def step_wait():
        result = wait()
        if result[2].any():
            values, mask = stats.numpy(), tracking.numpy()
            for w in np.flatnonzero(result[2] & (mask != 0)):
                result[3][w]['headroom_stats'] = values[w].tolist(); mask[w] = 0
            tracking.assign(mask)
        return result
    raw.reset, raw.step_wait = reset_all, step_wait
    gc.collect()
    assert raw._headroom_buffers[0] is stats
    return raw


def collect(cases, model, norm, condition, owner_only=False):
    n = len(cases); stats = wp.zeros((n, len(FIELDS)), dtype=D)
    tracking = wp.array(np.ones(n, np.int32))
    launch, factory = wp.launch, parent.experiment.raw_env
    def extended(kernel, dim, inputs=None, **kwargs):
        result = launch(kernel, dim, **kwargs) if inputs is None else launch(kernel, dim, inputs, **kwargs)
        if kernel is control_physical_nominal:
            assert len(inputs) == 20 and inputs[18].shape == (n, 6)
            launch(record, n, [inputs[1], inputs[7], inputs[6], inputs[5], inputs[15], inputs[18], tracking, stats])
        return result
    def instrumented(cases, mode):
        return attach(factory(cases, mode), stats, tracking)
    wp.launch, parent.experiment.raw_env = extended, instrumented
    try:
        return parent.collect(cases, model, norm, condition, owner_only)
    finally:
        wp.launch, parent.experiment.raw_env = launch, factory


def unit():
    from test_route_state import ScriptedEnv
    n = 128; rng = np.random.default_rng(156)
    ids = np.arange(10, dtype=np.int32); qv = rng.uniform(-80, 80, (n, 10)).astype(np.float32)
    gains = rng.uniform(.8, 1.2, (n, 6)); mem = np.zeros((n, 24)); mem[:, 18] = rng.uniform(-1, 1, n); mem[0, 18] = 0
    rpm = np.abs(qv[:, 8:10].astype(float))*60/(2*np.pi)
    bounds = 4.5*np.where(rpm > 490, np.maximum(0, (710-rpm)/(710-490)), 1)/np.maximum(1, gains[:, 4:6])
    diag = np.zeros((n, 38)); diag[:, 4:6] = rng.uniform(-1, 1, (n, 2))*bounds
    r = .3*mem[:, 18]; rr = np.c_[r, -r]; ratio = np.ones_like(rr)
    np.divide(bounds-diag[:, 4:6], rr, out=ratio, where=rr > 0)
    np.divide(-bounds-diag[:, 4:6], rr, out=ratio, where=rr < 0)
    lw = np.clip(np.minimum(1, ratio.min(axis=1)), 0, 1)
    lam = np.minimum(lw, rng.uniform(0, 1, n)); diag[:, 12] = lam; diag[:, 10:12] = lam[:, None]*rr
    active = np.ones(n, np.int32); tracking = active.copy(); active[1] = 0; tracking[2] = 0; diag[3, 14] = 1
    arrays = [wp.array(qv), wp.array(ids), wp.array(mem, dtype=D), wp.array(active), wp.array(diag, dtype=D), wp.array(gains, dtype=D), wp.array(tracking)]
    before = [a.numpy().copy() for a in arrays]; stats = wp.zeros((n, len(FIELDS)), dtype=D)
    wp.launch(record, n, [*arrays, stats]); got = stats.numpy(); expected = np.zeros_like(got)
    for w in range(n):
        if not active[w] or not tracking[w]: continue
        expected[w, 0] = 1
        if diag[w, 14]: expected[w, 2] = 1; continue
        a = lam[w]*r[w]; wl = (1-lw[w])*r[w]; hl = (lw[w]-lam[w])*r[w]
        expected[w, 1] = 1; expected[w, 3:11] = [r[w]**2, a*a, wl*wl, hl*hl, abs(r[w]-a), abs(wl), abs(hl), 2*wl*hl]
        expected[w, 15:18] = [abs(hl), lam[w], lw[w]]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    for a, b in zip(arrays, before): np.testing.assert_array_equal(a.numpy(), b)
    # Real wrapper logic: first terminal freezes; an explicit reset starts anew.
    s = wp.zeros((2, len(FIELDS)), dtype=D); mask = wp.array(np.ones(2, np.int32)); raw = attach(ScriptedEnv(), s, mask)
    raw.reset(); s.fill_(2); raw.step(np.zeros((2, 3), np.float32)); result = raw.step(np.zeros((2, 3), np.float32))
    assert result[3][0]['headroom_stats'] == [2.]*len(FIELDS) and mask.numpy().tolist() == [0, 1]
    raw.tick = 1; s.fill_(3); result = raw.step(np.zeros((2, 3), np.float32)); assert 'headroom_stats' not in result[3][0]
    raw.reset(); assert not s.numpy().any() and mask.numpy().tolist() == [1, 1]; raw.close()
    result = dict(verified=True,synthetic_GPU_rows=n,numpy_equivalence=True,all_inputs_unchanged=True,first_terminal_freeze_and_explicit_reset=True,physical_steps=0,training_updates=0)
    atomic_json(OUT/'unit.json', result); print('PASS128 GPU/NumPy/read-only and terminal/reset checks', flush=True)


def verify():
    p = json.loads((OUT/'proposal.json').read_text()); parent.verify()
    assert sha(ROOT/p['parent_proposal']) == p['parent_proposal_sha256']
    assert sha(ROOT/p['original_source_contract']) == p['original_source_contract_sha256']
    c = json.loads((OUT/'source_contract.json').read_text())
    assert c['proposal_sha256'] == sha(OUT/'proposal.json')
    assert all(sha(ROOT/n) == v for n, v in c['source_sha256'].items())
    return p


def freeze():
    assert not (OUT/'source_contract.json').exists(); parent.verify(); unit()
    p = json.loads((OUT/'proposal.json').read_text()); old = json.loads((ROOT/p['parent_proposal']).read_text())
    owner = collect(old['cases']['regular']+old['cases']['controlled'], None, None, p['conditions'][1], True)
    assert owner['verified'] and owner['physical_steps'] == 0
    atomic_json(OUT/'owner_check.json', owner)
    sources = dict(json.loads((ROOT/p['original_source_contract']).read_text())['source_sha256'])
    sources['wheelleg_warp/wheel_headroom_attribution.py'] = sha(__file__)
    atomic_json(OUT/'source_contract.json', dict(proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,unit_sha256=sha(OUT/'unit.json'),owner_sha256=sha(OUT/'owner_check.json'),fields=FIELDS,budget_episodes=816,training_updates=0))
    verify(); print('ADMITTED read-only816; no episodes run', flush=True)


def run():
    p = verify(); old = json.loads((ROOT/p['parent_proposal']).read_text()); destination = OUT/'runs'; destination.mkdir(exist_ok=False)
    ledger = []; completed = 0
    try:
        for m in p['models']:
            prefix = Path(m['prefix'])
            assert sha(prefix.with_suffix('.zip')) == m['checkpoint']['checkpoint_sha256'] and sha(prefix.with_suffix('.pkl')) == m['checkpoint']['normalization_sha256']
            model = PPO.load(str(prefix)+'.zip', device='cuda')
            for condition in p['conditions']:
                for panel in p['panels']:
                    label = f'{m["seed"]}_{condition}_{panel}'
                    atomic_json(OUT/'progress.json', dict(status='running',completed_episodes=completed,pending_job=label))
                    result = collect(old['cases'][panel], model, str(prefix)+'.pkl', condition)
                    path = destination/(label+'.json'); atomic_json(path, result)
                    ledger.append(dict(seed=m['seed'],condition=condition,panel=panel,path='runs/'+path.name,sha256=sha(path),episodes=len(result['runs'])))
                    completed += len(result['runs']); atomic_json(OUT/'completed_jobs.json', dict(completed_episodes=completed,records=ledger)); verify()
                    print('COMPLETED',completed,label,flush=True)
        assert completed == p['budget_gpu_episodes']; atomic_json(OUT/'completion.json', dict(completed_episodes=completed,records=ledger,source_contract_sha256=sha(OUT/'source_contract.json'),training_updates=0))
        atomic_json(OUT/'progress.json', dict(status='complete',completed_episodes=completed))
    except BaseException as e:
        atomic_json(OUT/'interruption.json', dict(completed_episodes=completed,error=str(e),pending_job_consumption_unknown=True,silently_resumable=False)); raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['unit','freeze','run'])
    globals()[parser.parse_args().command]()
