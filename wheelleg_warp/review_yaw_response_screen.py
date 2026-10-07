"""Independent reconstruction of the registered six-motor CPU diagnostic."""
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from review_yaw_sector import ROOT, sha

sys.path.insert(0, str(ROOT/'wheelleg_ppo/tools'))
import hardware_profile as hw
from native.terrain import HeightTerrainScenario, model

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/yaw_response_screen_v1'
JOINTS = ('alphaL', 'betaL', 'alphaR', 'betaR', 'wheel1', 'wheel2')


def rpy(q):
    w, x, y, z = q[3:7]
    return np.array([math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     math.asin(np.clip(2*(w*y-z*x), -1, 1)),
                     math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


def wrapped(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


def write(name, value):
    path = OUT/name; temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run():
    assert not any((OUT/n).exists() for n in ('review_started.json', 'review.json', 'review_failure.json'))
    np.testing.assert_allclose(rpy([0, 0, 0, 2**-.5, 0, 0, 2**-.5]), [0, 0, math.pi/2], atol=1e-15, rtol=0)
    reg = json.loads((OUT/'registration.json').read_text())
    primary = json.loads((OUT/'completion.json').read_text())
    source = json.loads((OUT/'source_contract.json').read_text())
    assert primary['verified'] and primary['cpu_steps'] == 336 and primary['states'] == 24
    assert primary['registration_sha256'] == sha(OUT/'registration.json')
    assert primary['source_contract_sha256'] == sha(OUT/'source_contract.json')
    assert all(sha(ROOT/n) == v for n, v in {**reg['input_sha256'], **source['source_sha256']}.items())
    assert sha(OUT/'states.npz') == reg['states_sha256']
    assert len(primary['records']) == len(reg['records']) == 24
    with np.load(OUT/'states.npz', allow_pickle=False) as z:
        states, saved_trace = z['pre'].copy(), z['trace'].copy()
    assert mujoco.get_mjcb_control() is None and mujoco.get_mjcb_passive() is None
    write('review_started.json', dict(registration_sha256=sha(OUT/'registration.json'),
          primary_sha256=sha(OUT/'completion.json'), reviewer_sha256=sha(__file__), cpu_budget=312))
    steps = 0; records = []; cache = {}; models = {}
    try:
        for expected, actual in zip(reg['records'], primary['records']):
            i, case, index = expected['index'], expected['case'], expected['snapshot_step']-1
            assert actual['index'] == i and actual['case'] == case
            assert actual['success_control'] == expected['success'] and actual['baseline_repeat_passed']
            path = ROOT/expected['trace']
            if case not in cache:
                assert sha(path) == expected['trace_sha256']
                with np.load(path, allow_pickle=False) as z:
                    cache[case] = {k: z[k].copy() for k in ('pre', 'post', 'trace', 'columns')}
            raw = cache[case]; columns = list(map(str, raw['columns']))
            hit = np.flatnonzero(raw['trace'][:, columns.index('left_target_normal_N')]+
                                 raw['trace'][:, columns.index('right_target_normal_N')]>1e-6)
            assert expected['first_target_load_step'] == int(hit[0])+1
            assert index == int(hit[0])+expected['offset_steps']
            state = raw['pre'][index]
            np.testing.assert_array_equal(states[i], state)
            np.testing.assert_array_equal(saved_trace[i], raw['trace'][index])
            rp = ROOT/expected['result']; assert sha(rp) == expected['result_sha256']
            row = next(x for x in json.loads(rp.read_text())['runs'] if x['seed'] == case)
            assert row['scenario'] == expected['scenario'] and row['success'] == expected['success']
            if case not in models: models[case] = model(HeightTerrainScenario(**expected['scenario']))
            m = models[case]
            assert (m.nq, m.nv, m.nu, m.na, m.nmocap, m.nplugin) == (17, 16, 6, 0, 0, 0)
            assert m.opt.timestep == .0005 and int(m.opt.integrator) == 3 and m.opt.iterations == 100
            assert list(m.actuator_trnid[:, 0]) == [m.joint(n).id for n in JOINTS]
            qa = [m.jnt_qposadr[m.joint(n).id] for n in JOINTS[:4]]
            bounds = []
            for j, name in enumerate(JOINTS):
                rpm = abs(state[17+m.jnt_dofadr[m.joint(name).id]])*60/(2*math.pi)
                rated, unloaded, peak = ((hw.HIP_RATED_RPM, hw.HIP_NO_LOAD_RPM, hw.HIP_PEAK_TORQUE)
                                         if j < 4 else (hw.MOTOR_RATED_RPM, hw.MOTOR_NO_LOAD_RPM, hw.MOTOR_PEAK_TORQUE))
                value = peak if rpm <= rated else peak*max(0., (unloaded-rpm)/(unloaded-rated))
                bounds.append(value/max(1., row['actuator_gain_upper'][j]))
            np.testing.assert_allclose(bounds, actual['bounds_Nm'], atol=1e-12, rtol=0)
            assert list(actual['branches']) == reg['arms']
            rebuilt = {}
            for name in reg['arms']:
                b = actual['branches'][name]; ctrl = state[33:39].copy()
                if name != 'recorded_ctrl':
                    _, motor, sign = name.split('_'); motor = int(motor)
                    ctrl[motor] = min(bounds[motor], max(-bounds[motor], ctrl[motor]+int(sign)*reg['perturbation_Nm']))
                np.testing.assert_array_equal(ctrl, b['ctrl'])
                np.testing.assert_array_equal(ctrl-state[33:39], b['applied_delta_Nm'])
                d = mujoco.MjData(m); d.qpos[:] = state[:17]; d.qvel[:] = state[17:33]
                d.ctrl[:] = ctrl; d.qacc_warmstart[:] = 0.; d.time = expected['pre_time_s']
                assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
                steps += 1; mujoco.mj_step(m, d)
                assert d.time == expected['pre_time_s']+.0005
                for field, value in [('q', d.qpos), ('v', d.qvel), ('rpy', rpy(d.qpos)),
                                     ('active_joint_margin_rad', 1.4-abs(d.qpos[qa]))]:
                    assert np.isfinite(value).all()
                    np.testing.assert_allclose(value, b[field], atol=1e-12, rtol=0)
                assert len(b['contacts']) == d.ncon
                for k, contact in enumerate(b['contacts']):
                    c = d.contact[k]; force = np.zeros(6); mujoco.mj_contactForce(m, d, k, force)
                    for field, value in [('geom', c.geom), ('frame', c.frame), ('point', c.pos),
                                         ('distance', c.dist), ('local_force_torque', force)]:
                        assert np.isfinite(value).all()
                        np.testing.assert_allclose(value, contact[field], atol=1e-12, rtol=0)
                rebuilt[name] = rpy(d.qpos)[2]
            baseline = actual['branches']['recorded_ctrl']
            for name, yaw in rebuilt.items():
                np.testing.assert_allclose(wrapped(yaw-rebuilt['recorded_ctrl']),
                                           actual['branches'][name]['yaw_minus_baseline_rad'], atol=1e-15, rtol=0)
            for j in range(6):
                midpoint = .5*(actual['branches'][f'motor_{j}_1']['yaw_minus_baseline_rad']+
                               actual['branches'][f'motor_{j}_-1']['yaw_minus_baseline_rad'])
                assert midpoint == actual['midpoint_yaw_rad'][str(j)]
            gpu = raw['post'][index]
            np.testing.assert_array_equal(np.array(baseline['q'])-gpu[:17], actual['cpu_minus_original_gpu_q'])
            np.testing.assert_array_equal(np.array(baseline['v'])-gpu[17:33], actual['cpu_minus_original_gpu_v'])
            effect = max(abs(b['yaw_minus_baseline_rad']) for name, b in actual['branches'].items() if name != 'recorded_ctrl')
            yaw_error = wrapped(baseline['rpy'][2]-rpy(gpu)[2])
            records.append(dict(index=i, case=case, offset_steps=expected['offset_steps'], success_control=expected['success'],
                cpu_minus_gpu_yaw_rad=yaw_error, maximum_perturbation_yaw_rad=effect,
                backend_yaw_error_to_perturbation_ratio=abs(yaw_error)/effect if effect else None,
                yaw_per_actual_motor_Nm={name: b['yaw_minus_baseline_rad']/b['applied_delta_Nm'][int(name.split('_')[1])]
                    if b['applied_delta_Nm'][int(name.split('_')[1])] != 0 else None
                    for name, b in actual['branches'].items() if name != 'recorded_ctrl'}))
        assert steps == 312 and primary['cpu_steps']+steps == reg['cpu_step_budget']['total']
        assert all(sha(ROOT/n) == v for n, v in source['source_sha256'].items())
        write('review.json', dict(verified=True, states=24, arms_reconstructed=312, cpu_review_steps=steps,
            total_cpu_steps=648, records=records, registration_sha256=sha(OUT/'registration.json'),
            completion_sha256=sha(OUT/'completion.json'), reviewer_sha256=sha(__file__),
            maximum_abs_cpu_gpu_yaw_rad=max(abs(r['cpu_minus_gpu_yaw_rad']) for r in records),
            backend_yaw_error_exceeds_local_perturbation_states=sum(abs(r['cpu_minus_gpu_yaw_rad'])>r['maximum_perturbation_yaw_rad'] for r in records),
            screen_status='closed_after_registered_budget', training_updates=0, new_gpu_rollouts=0,
            limits=reg['limitations'], next='228 finite method/information/learning-necessity decision;no extension of this screen or revival of failed fixed heuristics.'))
        print('PASS227 independently reconstructed312arms;total648;screenclosed', flush=True)
    except BaseException as e:
        write('review_failure.json', dict(error=repr(e), attempted_cpu_steps=steps, records=records)); raise


if __name__ == '__main__': run()
