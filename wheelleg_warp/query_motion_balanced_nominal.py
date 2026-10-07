"""Registered 70 active static queries; one bank, no physics graph launch."""
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
import light_phase_reference as light
import motion_balanced_control as candidate
import motion_balanced_nominal as prototype
from native.controller import D
from native.terrain import HeightTerrainScenario
from train_height_comparison import raw_env
from query_nominal_compatibility import initialize
from test_nom_yaw_filter_probe import control_args
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = prototype.OUT


def run():
    assert not any((OUT / name).exists() for name in
                   ('query_contract.json', 'completion.json', 'failure.json'))
    start = time.monotonic()
    p = json.loads((OUT / 'proposal.json').read_text())
    a = json.loads((OUT / 'source_admission.json').read_text())
    assert p['static_query_budget'] == 70 and a['verified']
    assert a['proposal_sha256'] == sha(OUT / 'proposal.json')
    assert all(sha(ROOT / n) == h for n, h in a['source_sha256'].items())
    assert prototype.COPY.read_text() == prototype.expected_source()
    references = list(p['moving_references'])
    move = OUT.parent / 'moving_equilibrium_contract_v1/proposal.json'
    for r in json.loads(move.read_text())['static_reference_inputs']:
        references.append(dict(height=r['height_m'], speed=0., path=r['path'],
                               sha256=r['sha256']))
    q, v, required, cases = [], [], [], []
    for i, r in enumerate(references):
        path = ROOT / r['path']
        assert sha(path) == r['sha256']
        with np.load(path, allow_pickle=False) as z:
            prefix = 'reference_' if r['speed'] == 0. else ''
            q.append(z[prefix + 'q'].copy())
            v.append(z[prefix + 'v'].copy())
            required.append(z[prefix + 'ctrl'].copy())
        # Constructor's traversal horizon needs nonzero speed; query command is set below.
        cases.append(dict(seed=2690000+i, scenario=HeightTerrainScenario(
            speed=r['speed'] or .7, mass=7., height_l=0., height_r=0.,
            center=1.8, offset=0., mu_l=.8, mu_r=.8, drive_difference=0.,
            delay_ms=0., solver_iterations=100, stand_height_m=r['height']).__dict__))
    q, v, required = map(np.array, (q, v, required))
    sources = dict(a['source_sha256'])
    for name in ('query_motion_balanced_nominal.py', 'query_nominal_compatibility.py',
                 'light_phase_reference.py', 'test_nom_yaw_filter_probe.py'):
        sources['wheelleg_warp/' + name] = sha(ROOT / 'wheelleg_warp' / name)
    fd_calls = []
    fd = mujoco.mjd_transitionFD
    def counted_fd(*args, **kwargs):
        fd_calls.append(dict(eps=float(args[2]), centered=bool(args[3])))
        return fd(*args, **kwargs)
    mujoco.mjd_transitionFD = counted_fd
    try:
        raw = light.instrument(raw_env, cases, 'virtual6')
    finally:
        mujoco.mjd_transitionFD = fd
    queries, records, arrays = 0, [], {}
    device = raw.data.qpos.device
    launch_graph = wp.capture_launch
    def reject_graph(*args, **kwargs):
        raise AssertionError('Physics graph launch forbidden in static query')
    wp.capture_launch = reject_graph
    try:
        assert len(fd_calls) == 10 and raw._light_phase_topology['verified']
        assert device.is_cuda and raw.cpu.opt.timestep == .0005
        assert raw.cpu.opt.iterations == 100 and int(raw.cpu.opt.integrator) == 3
        values = [prototype.reference_values(raw.cpu, qi, ci)[:2]
                  for qi, ci in zip(q, required)]
        atomic_json(OUT / 'query_contract.json', dict(
            verified=True, proposal_sha256=sha(OUT / 'proposal.json'),
            admission_sha256=sha(OUT / 'source_admission.json'),
            source_sha256=sources, references=references,
            static_query_budget=70, allocated_worlds=15,
            inactive_control_threads=20, baseline_transitionFD_calls=fd_calls,
            topology=raw._light_phase_topology,
            scope='moving10 x3integrals x2arms +static5 x2arms; inactive last5 in final two batches. One frozen bank; reset identical q/v/memory/phase each arm.'))
        for j, integral in enumerate((-.3, 0., .3)):
            active = np.ones(15, np.int32)
            if j:
                active[10:] = 0
            for arm, kernel in (
                    ('old', light.phase.roles.experimental.control_physical_nominal),
                    ('candidate', candidate.control_physical_nominal)):
                raw.reset()
                raw.active.assign(active)
                raw.data.qpos.assign(q.astype(np.float32))
                raw.data.qvel.assign(v.astype(np.float32))
                raw.data.sensordata.zero_()
                raw.data.ctrl.zero_()
                raw.targets.zero_()
                raw.nominal_correction.zero_()
                raw.command.assign(np.array([r['speed'] for r in references], float))
                state = raw.state.numpy()
                state[:, 0], state[:, 1] = 4000, -1
                raw.state.assign(state)
                wp.launch(initialize, 15, [raw.data.qpos, raw.data.qvel, raw.ids,
                                          raw.k['state'], D(integral)])
                memory = raw.k['state'].numpy()
                memory[10:, 11] = 0.
                raw.k['state'].assign(memory)
                light.prepare_step(raw, 0)
                args = control_args(raw)
                base_reference = raw._light_phase_buffers[0].numpy().copy()
                ref = base_reference
                if arm == 'candidate':
                    ref = np.array([prototype.private_reference(
                        base_reference[i], r['height'], r['speed'], *values[i])
                        for i, r in enumerate(references)])
                args[12] = wp.array(ref, dtype=D, device=device)
                unchanged = [('q', raw.data.qpos), ('v', raw.data.qvel),
                             ('sensor', raw.data.sensordata), ('command', raw.command),
                             ('environment_state', raw.state), ('clock', raw.data.time),
                             ('gains', raw.k['gains']), ('feed', raw.k['feed']),
                             ('angles', raw.k['angles']), ('heights', raw.k['heights']),
                             ('active', raw.active), ('targets', raw.targets),
                             ('nominal_correction', raw.nominal_correction),
                             ('reference', args[12])]
                before = {name: value.numpy().copy() for name, value in unchanged}
                before['memory_before'] = raw.k['state'].numpy().copy()
                queries += int(active.sum())
                wp.launch(kernel, 15, args, block_dim=32)
                wp.launch(light.phase.roles.finish, 15,
                          [0, raw.diag, raw._light_phase_buffers[1]])
                result = {**before, 'memory_after': raw.k['state'].numpy(),
                          'ctrl': raw.data.ctrl.numpy(), 'diag': raw.diag.numpy(),
                          'base_reference': base_reference, 'required_ctrl': required,
                          'original_q': q, 'original_v': v,
                          'role': raw._light_phase_buffers[1].numpy(),
                          'phase': raw._light_phase_buffers[2].numpy()}
                for name, value in unchanged:
                    np.testing.assert_array_equal(value.numpy(), before[name])
                for name in ('ctrl', 'diag', 'memory_after'):
                    assert np.isfinite(result[name]).all()
                assert not result['clock'].any()
                result['ctrl_minus_required'] = result['ctrl'].astype(float) - required
                path = OUT / f'paired_integral_{j}_{arm}.npz'
                np.savez_compressed(path, **result)
                arrays[j, arm] = result
                for i in np.flatnonzero(active):
                    records.append(dict(arm=arm, world=int(i), height=references[i]['height'],
                        speed=references[i]['speed'], state11=float(before['memory_before'][i, 11]),
                        path=path.name, sha256=sha(path),
                        maximum_command_gap_Nm=float(abs(result['ctrl_minus_required'][i]).max()),
                        wheel_gap_Nm=float(abs(result['ctrl_minus_required'][i, -2:]).max()),
                        raw_nominal_gap_Nm=float(abs(result['diag'][i, 15:21] - required[i]).max()),
                        projection_error=float(result['diag'][i, 14])))
                print('completed', queries, '/70 activequeries', arm, integral, flush=True)
            old, new = arrays[j, 'old'], arrays[j, 'candidate']
            for name in ('q', 'v', 'sensor', 'command', 'environment_state', 'clock',
                         'gains', 'feed', 'angles', 'heights', 'active', 'targets',
                         'nominal_correction', 'memory_before', 'base_reference'):
                np.testing.assert_array_equal(old[name], new[name])
            np.testing.assert_array_equal(old['reference'], new['reference'][:, :16])
        zero_exact = all(np.array_equal(arrays[0, 'old'][name][10:],
                                       arrays[0, 'candidate'][name][10:])
                         for name in ('ctrl', 'diag', 'memory_after', 'role', 'phase'))
        assert queries == len(records) == 70
        assert all(sha(ROOT / n) == h for n, h in sources.items())
        atomic_json(OUT / 'completion.json', dict(
            verified=True, individual_static_queries=queries, records=records,
            zero_speed_ctrl_diag_memory_role_phase_exact=zero_exact,
            source_contract_sha256=sha(OUT / 'query_contract.json'),
            device=str(device), end_to_end_query_seconds=time.monotonic()-start,
            baseline_gain_construction_transitionFD_calls=len(fd_calls),
            new_experimental_transitionFD_calls=0, integration_steps=0,
            optimization_calls=0, new_training_samples=0,
            controller_admitted=False, baseline_replacement_admitted=False,
            limits='Static registered nominal points only; F32 uploads and protected output recorded. No runtime interpolation or dynamic stability qualification; finite direction review next.'))
        print('DONE26970static;zero-speedexact=', zero_exact, flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(error=repr(error),
                   attempted_individual_queries=queries, records=records,
                   implicit_retry=False))
        raise
    finally:
        wp.capture_launch = launch_graph
        raw.close()


if __name__ == '__main__':
    run()
