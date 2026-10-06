"""Optional fixed-policy wheel projection; original controller sources stay frozen."""
import json
from pathlib import Path

import numpy as np
import warp as wp
from stable_baselines3 import PPO

import fixed_steering_leg as parent
from wheel_headroom_attribution import attach
from wheel_component_projection import allocate
from native.controller import D, V6, command_bounds, control_physical_nominal
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/wheel_component_projection_v1'
MODES = {'uniform_original': 0, 'component_wheels': 1}
FIELDS = ['attempted', 'valid', 'invalid', 'filtered_energy', 'original_left_energy',
          'original_right_energy', 'selected_left_energy', 'selected_right_energy',
          'selected_common_energy', 'selected_differential_energy', 'original_differential_energy',
          'sum_signed_differential_gain', 'executed_left_energy', 'executed_right_energy',
          'sum_original_global_lambda', 'max_original_acceptance_error',
          'max_original_float32_box_excess', 'max_selected_float32_box_excess',
          'max_common_bound_excess', 'max_signed_differential_degradation',
          'max_selected_diag_rounding_error', 'max_zero_nonbinding_command_change']


@wp.kernel
def select(mode: int, qvel: wp.array2d[float], ids: wp.array[int], memory: wp.array2d[D],
           active: wp.array[int], diagnostic: wp.array2d[D], gains: wp.array2d[D],
           ctrl: wp.array2d[float], tracking: wp.array[int], stats: wp.array2d[D]):
    w = wp.tid()
    if active[w] == 0:
        return
    log = tracking[w] != 0
    if log:
        stats[w, 0] += D(1)
    if diagnostic[w, 14] != D(0):
        if log:
            stats[w, 2] += D(1)
        return
    speeds = V6(); scales = V6(); valid = True
    for j in range(6):
        speeds[j] = D(qvel[w, ids[4+j]]); scales[j] = gains[w, j]
        if not wp.isfinite(speeds[j]) or not wp.isfinite(scales[j]) or scales[j] <= D(0):
            valid = False
    bounds = command_bounds(speeds, scales)
    r = D(.3)*memory[w, 18]; lam = diagnostic[w, 12]
    if not wp.isfinite(r) or wp.abs(r) > D(.3)+D(1.e-12) or not wp.isfinite(lam) or lam < D(0) or lam > D(1):
        valid = False
    for j in range(4, 6):
        if not wp.isfinite(diagnostic[w, j]) or wp.abs(diagnostic[w, j]) > bounds[j]+D(1.e-9) or not wp.isfinite(diagnostic[w, 6+j]) or not wp.isfinite(ctrl[w, j]):
            valid = False
    if not valid:
        # Refuse the optional projection; preserve the original guarded output.
        if log:
            stats[w, 2] += D(1)
        return
    bL = diagnostic[w, 4]; bR = diagnostic[w, 5]
    oldL = diagnostic[w, 10]; oldR = diagnostic[w, 11]
    oldctrlL = ctrl[w, 4]; oldctrlR = ctrl[w, 5]
    left = oldL; right = oldR
    if mode == 1:
        uL = wp.clamp(bL+r, -bounds[4], bounds[4])
        uR = wp.clamp(bR-r, -bounds[5], bounds[5])
        ctrl[w, 4] = float(uL); ctrl[w, 5] = float(uR)
        left = uL-bL; right = uR-bR
        diagnostic[w, 10] = left; diagnostic[w, 11] = right
    if not log:
        return
    common = (left+right)/D(2); diff = (left-right)/D(2); olddiff = (oldL-oldR)/D(2)
    exL = D(ctrl[w, 4])-D(float(bL)); exR = D(ctrl[w, 5])-D(float(bR))
    stats[w, 1] += D(1); stats[w, 3] += r*r
    stats[w, 4] += oldL*oldL; stats[w, 5] += oldR*oldR
    stats[w, 6] += left*left; stats[w, 7] += right*right
    stats[w, 8] += common*common; stats[w, 9] += diff*diff; stats[w, 10] += olddiff*olddiff
    stats[w, 11] += wp.sign(r)*(diff-olddiff)
    stats[w, 12] += exL*exL; stats[w, 13] += exR*exR; stats[w, 14] += lam
    stats[w, 15] = wp.max(stats[w, 15], wp.max(wp.abs(oldL-lam*r), wp.abs(oldR+lam*r)))
    stats[w, 16] = wp.max(stats[w, 16], wp.max(wp.abs(D(oldctrlL))-bounds[4], wp.abs(D(oldctrlR))-bounds[5]))
    stats[w, 17] = wp.max(stats[w, 17], wp.max(wp.abs(D(ctrl[w, 4]))-bounds[4], wp.abs(D(ctrl[w, 5]))-bounds[5]))
    stats[w, 18] = wp.max(stats[w, 18], wp.abs(common)-wp.abs(r)/D(2))
    stats[w, 19] = wp.max(stats[w, 19], -wp.sign(r)*(diff-olddiff))
    stats[w, 20] = wp.max(stats[w, 20], wp.max(wp.abs(D(ctrl[w, 4])-(bL+left)), wp.abs(D(ctrl[w, 5])-(bR+right))))
    if r == D(0) or lam == D(1):
        stats[w, 21] = wp.max(stats[w, 21], wp.max(wp.abs(D(ctrl[w, 4])-D(oldctrlL)), wp.abs(D(ctrl[w, 5])-D(oldctrlR))))


def attach_stats(raw, stats, tracking):
    raw = attach(raw, stats, tracking); raw._wheel_projection_buffers = (stats, tracking)
    wait = raw.step_wait
    def step_wait():
        result = wait()
        for info in result[3]:
            if 'headroom_stats' in info:
                info['wheel_projection_stats'] = info.pop('headroom_stats')
        return result
    raw.step_wait = step_wait
    assert raw._wheel_projection_buffers[0] is stats
    return raw


def collect(cases, model, norm, condition, allocator, owner_only=False):
    n = len(cases); stats = wp.zeros((n, len(FIELDS)), dtype=D); tracking = wp.array(np.ones(n, np.int32))
    launch, factory = wp.launch, parent.experiment.raw_env; context = {}
    def extended(kernel, dim, inputs=None, **kwargs):
        result = launch(kernel, dim, **kwargs) if inputs is None else launch(kernel, dim, inputs, **kwargs)
        if kernel is control_physical_nominal:
            assert len(inputs) == 20 and inputs[18].shape == (n, 6)
            context['control'] = inputs
        elif kernel is parent.apply_budget:
            a = context['control']
            launch(select, n, [MODES[allocator],a[1],a[7],a[6],a[5],a[15],a[18],a[14],tracking,stats])
        return result
    def instrumented(cases, mode):
        return attach_stats(factory(cases, mode), stats, tracking)
    wp.launch, parent.experiment.raw_env = extended, instrumented
    try:
        result = parent.collect(cases, model, norm, condition, owner_only)
        if owner_only:
            assert context and stats.device.is_cuda; result['projection_owner_and_graph_order_checked'] = True
        return result
    finally:
        wp.launch, parent.experiment.raw_env = launch, factory


def unit():
    from test_route_state import ScriptedEnv
    n = 128; rng = np.random.default_rng(158); ids = np.arange(10, dtype=np.int32)
    qv = rng.uniform(-80, 80, (n, 10)).astype(np.float32); gains = rng.uniform(.8, 1.2, (n, 6))
    rpm = abs(qv[:, 8:10].astype(float))*60/(2*np.pi)
    limits = 4.5*np.where(rpm > 490,np.maximum(0,(710-rpm)/220),1)/np.maximum(1,gains[:,4:6])
    b = rng.uniform(-1,1,(n,2))*limits; s = rng.uniform(-.3,.3,n); s[0] = 0
    b[6] = 0; s[6] = .01; limits[6] = 4.5; qv[6] = 0; gains[6] = 1
    projected, _, fractions = allocate(b, limits, s); lam = fractions.min(axis=1)*rng.uniform(.3,1,n); lam[6] = 1
    mem = np.zeros((n,24)); mem[:,18] = s/.3
    diag = np.zeros((n,38)); diag[:,4:6] = b; diag[:,10:12] = lam[:,None]*np.c_[s,-s]; diag[:,12] = lam
    ctrl = rng.uniform(-2,2,(n,6)).astype(np.float32); ctrl[:,4:6] = b+diag[:,10:12]
    active = np.ones(n,np.int32); tracking = active.copy(); active[1] = 0; tracking[2] = 0; diag[3,14] = 2
    mem[4,18] = np.nan; gains[5,4] = 0
    for mode in (0,1):
        arrays = [wp.array(qv),wp.array(ids),wp.array(mem,dtype=D),wp.array(active),wp.array(diag,dtype=D),wp.array(gains,dtype=D),wp.array(ctrl),wp.array(tracking)]
        stats = wp.zeros((n,len(FIELDS)),dtype=D); wp.launch(select,n,[mode,*arrays,stats])
        out = arrays[6].numpy(); dd = arrays[4].numpy(); log = stats.numpy()
        expected = ctrl.copy(); expected_diag = diag.copy(); valid = np.ones(n,bool); valid[[1,3,4,5]] = False
        if mode == 1:
            expected[valid,4:6] = projected[valid]; expected_diag[valid,10:12] = projected[valid]-b[valid]
        np.testing.assert_array_equal(out,expected); np.testing.assert_allclose(dd,expected_diag,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(out[:,:4],ctrl[:,:4]); np.testing.assert_array_equal(dd[:,:10],diag[:,:10]); np.testing.assert_array_equal(dd[:,12:],diag[:,12:])
        for i, original in [(0,qv),(1,ids),(2,mem),(3,active),(5,gains),(7,tracking)]: np.testing.assert_array_equal(arrays[i].numpy(),original)
        assert not log[[1,2]].any() and np.all(log[[3,4,5],2] == 1) and np.all(log[valid & (tracking != 0),1] == 1)
        assert np.max(log[:,15]) < 1e-12 and np.max(log[:,16:18]) < 1e-6 and np.max(log[:,18:20]) < 1e-12 and np.max(log[:,21]) == 0
        for w in np.flatnonzero(valid & (tracking != 0)):
            left,right = expected_diag[w,10:12]; oldL,oldR = diag[w,10:12]; common = (left+right)/2; diff = (left-right)/2; olddiff = (oldL-oldR)/2
            ex = expected[w,4:6].astype(float)-b[w].astype(np.float32).astype(float)
            exp = [s[w]**2,oldL**2,oldR**2,left**2,right**2,common**2,diff**2,olddiff**2,np.sign(s[w])*(diff-olddiff),ex[0]**2,ex[1]**2,lam[w]]
            np.testing.assert_allclose(log[w,3:15],exp,rtol=1e-12,atol=1e-12)
    st = wp.zeros((2,len(FIELDS)),dtype=D); mask = wp.array(np.ones(2,np.int32)); raw = attach_stats(ScriptedEnv(),st,mask)
    raw.reset(); st.fill_(2); raw.step(np.zeros((2,3),np.float32)); result = raw.step(np.zeros((2,3),np.float32))
    assert result[3][0]['wheel_projection_stats'] == [2.]*len(FIELDS) and mask.numpy().tolist() == [0,1]
    raw.tick = 1; st.fill_(3); result = raw.step(np.zeros((2,3),np.float32)); assert 'wheel_projection_stats' not in result[3][0]
    raw.reset(); assert not st.numpy().any() and mask.numpy().tolist() == [1,1]; raw.close()
    atomic_json(OUT/'gpu_unit.json',dict(verified=True,rows_per_mode=n,modes=[0,1],numpy_command_diag_and_energy_equivalence=True,hips_nominal_filter_state_and_inputs_unchanged=True,zero_nonbinding_float32_identity=True,masked_world_still_projected_without_recording=True,invalid_projection_refused_original_output_retained=True,first_terminal_reset=True,physical_steps=0,training_updates=0))
    print('PASS128x2 GPU projection/identity/semantics/invalid/terminal checks',flush=True)


def verify():
    parent.verify(); p = json.loads((OUT/'proposal.json').read_text()); c = json.loads((OUT/'source_contract.json').read_text())
    assert sha(ROOT/p['parent_proposal']) == p['parent_proposal_sha256'] and sha(ROOT/p['parent_source_contract']) == p['parent_source_contract_sha256']
    assert c['proposal_sha256'] == sha(OUT/'proposal.json') and all(sha(ROOT/n) == v for n,v in c['source_sha256'].items())
    assert c['unit_sha256'] == sha(OUT/'gpu_unit.json') and c['owner_sha256'] == sha(OUT/'owner_check.json')
    return p


def freeze():
    assert not (OUT/'source_contract.json').exists(); parent.verify(); unit()
    p = json.loads((OUT/'proposal.json').read_text()); old = json.loads((ROOT/p['parent_proposal']).read_text())
    owner = collect(old['cases']['regular']+old['cases']['controlled'],None,None,p['conditions'][1],p['allocators'][1],True)
    assert owner['verified'] and owner['physical_steps'] == 0; atomic_json(OUT/'owner_check.json',owner)
    sources = dict(json.loads((ROOT/p['parent_source_contract']).read_text())['source_sha256'])
    for n in ['wheel_headroom_attribution.py','wheel_component_projection.py','gpu_wheel_component.py']:
        sources['wheelleg_warp/'+n] = sha(ROOT/'wheelleg_warp'/n)
    atomic_json(OUT/'source_contract.json',dict(proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,unit_sha256=sha(OUT/'gpu_unit.json'),owner_sha256=sha(OUT/'owner_check.json'),fields=FIELDS,budget_first_episode_evaluations=1632,training_updates=0,
        semantics='Selector after original leg room, before physics; wheel diag updates feed existing reward/observation. Original lambda is retained as original global statistic; parent steering max acceptance error is a historical scalar identity, not a component-mode validity check. Invalid inputs refuse optional projection, preserving original guarded output.'))
    verify(); print('ADMITTED optional1632 fixed-policy evaluations; no episodes run',flush=True)


def run():
    p = verify(); old = json.loads((ROOT/p['parent_proposal']).read_text()); dest = OUT/'runs'; dest.mkdir(exist_ok=False)
    ledger = []; completed = 0
    try:
        for m in p['models']:
            prefix = Path(m['prefix']); assert sha(prefix.with_suffix('.zip')) == m['checkpoint']['checkpoint_sha256'] and sha(prefix.with_suffix('.pkl')) == m['checkpoint']['normalization_sha256']
            model = PPO.load(str(prefix)+'.zip',device='cuda')
            for condition in p['conditions']:
                for allocator in p['allocators']:
                    for panel in p['panels']:
                        label = f'{m["seed"]}_{condition}_{allocator}_{panel}'
                        atomic_json(OUT/'progress.json',dict(status='running',completed_evaluations=completed,pending_job=label))
                        result = collect(old['cases'][panel],model,str(prefix)+'.pkl',condition,allocator)
                        path = dest/(label+'.json'); atomic_json(path,result)
                        ledger.append(dict(seed=m['seed'],condition=condition,allocator=allocator,panel=panel,path='runs/'+path.name,sha256=sha(path),evaluations=len(result['runs'])))
                        completed += len(result['runs']); atomic_json(OUT/'completed_jobs.json',dict(completed_evaluations=completed,records=ledger)); verify()
                        print('COMPLETED',completed,label,flush=True)
        assert completed == p['budget_first_episode_evaluations']
        atomic_json(OUT/'completion.json',dict(completed_evaluations=completed,records=ledger,source_contract_sha256=sha(OUT/'source_contract.json'),training_updates=0))
        atomic_json(OUT/'progress.json',dict(status='complete',completed_evaluations=completed))
    except BaseException as e:
        atomic_json(OUT/'interruption.json',dict(completed_evaluations=completed,error=str(e),pending_job_consumption_unknown=True,silently_resumable=False)); raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('command',choices=['unit','freeze','run'])
    globals()[parser.parse_args().command]()
