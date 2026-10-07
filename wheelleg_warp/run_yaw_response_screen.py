"""Registered CPU plant sensitivity; no control or learning admission."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from review_yaw_sector import ROOT, sha

sys.path.insert(0, str(ROOT/'wheelleg_ppo/tools'))
import wheelleg_sim as sim
from ppo_env import XML
from native.terrain import HeightTerrainScenario, model

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/yaw_response_screen_v1'
JOINTS = ('alphaL', 'betaL', 'alphaR', 'betaR', 'wheel1', 'wheel2')


def write(name, value):
    p = OUT/name
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def commands(base, bounds):
    result = {'recorded_ctrl': base.copy()}
    for j in range(6):
        for sign in (-1, 1):
            ctrl = base.copy()
            ctrl[j] = np.clip(base[j]+sign*.01, -bounds[j], bounds[j])
            result[f'motor_{j}_{sign}'] = ctrl
    return result


def prepare():
    reg = json.loads((OUT/'registration.json').read_text())
    assert reg['states'] == 24 and reg['cpu_step_budget']['total'] == 648
    assert sha(OUT/'states.npz') == reg['states_sha256']
    for name, digest in {**reg['input_sha256'], **reg['runtime_source_sha256']}.items():
        assert sha(ROOT/name) == digest
    with np.load(OUT/'states.npz', allow_pickle=False) as z:
        pre, trace, columns = z['pre'].copy(), z['trace'].copy(), list(map(str, z['columns']))
        np.testing.assert_array_equal(z['state_sizes'], [17, 16, 6])
    assert pre.shape == (24, 39) and np.isfinite(pre).all()
    originals = {}; cases = {}; prepared = []
    for r in reg['records']:
        if r['trace'] not in originals:
            with np.load(ROOT/r['trace'], allow_pickle=False) as z:
                originals[r['trace']] = {k: z[k].copy() for k in ('pre', 'post', 'trace')}
        i, step = r['index'], r['snapshot_step']-1
        original = originals[r['trace']]
        np.testing.assert_array_equal(pre[i], original['pre'][step])
        np.testing.assert_array_equal(trace[i], original['trace'][step])
        rows = json.loads((ROOT/r['result']).read_text())['runs']
        row = next(x for x in rows if x['seed'] == r['case'])
        assert sha(ROOT/r['result']) == r['result_sha256'] and row['scenario'] == r['scenario']
        assert row['success'] == r['success']
        if r['case'] not in cases:
            m = model(HeightTerrainScenario(**r['scenario']))
            assert (m.nq, m.nv, m.nu, m.na, m.nmocap, m.nplugin) == (17, 16, 6, 0, 0, 0)
            assert m.opt.timestep == .0005 and m.opt.iterations == 100
            assert int(m.opt.integrator) == 3
            assert list(m.actuator_trnid[:, 0]) == [m.joint(n).id for n in JOINTS]
            cases[r['case']] = m
        m = cases[r['case']]
        qa = [m.jnt_qposadr[m.joint(n).id] for n in JOINTS[:4]]
        va = [m.jnt_dofadr[m.joint(n).id] for n in JOINTS]
        gains = np.array(row['actuator_gain_upper'])
        bounds = np.array([sim.hw.torque_limit(float('inf'), float(pre[i, 17+v]), j < 4, 0., .0005)[0]
                           /max(1., gains[j]) for j, v in enumerate(va)])
        np.testing.assert_allclose(bounds[-2:], trace[i, 58:60], atol=1e-12, rtol=0)
        ctrl = commands(pre[i, 33:39], bounds)
        assert list(ctrl) == reg['arms']
        for name, u in ctrl.items():
            assert np.isfinite(u).all()
            if name == 'recorded_ctrl': continue
            motor = int(name.split('_')[1])
            np.testing.assert_array_equal(np.delete(u, motor), np.delete(pre[i, 33:39], motor))
            assert abs(u[motor]) <= bounds[motor]
        prepared.append((r, m, qa, pre[i], bounds, ctrl, original['post'][step, :33].copy()))
    assert len(prepared) == 24
    assert mujoco.get_mjcb_control() is None and mujoco.get_mjcb_passive() is None
    return reg, prepared


def advance(m, state, ctrl, time, qa):
    d = mujoco.MjData(m)
    d.qpos[:] = state[:17]; d.qvel[:] = state[17:33]
    d.ctrl[:] = ctrl; d.time = time; d.qacc_warmstart[:] = 0.
    assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
    mujoco.mj_step(m, d)
    assert d.time == time+.0005
    forces = []
    for i in range(d.ncon):
        force = np.zeros(6); mujoco.mj_contactForce(m, d, i, force)
        c = d.contact[i]
        forces.append(dict(geom=list(map(int, c.geom)), frame=c.frame.tolist(),
                           point=c.pos.tolist(), distance=float(c.dist), local_force_torque=force.tolist()))
    result = dict(q=d.qpos.tolist(), v=d.qvel.tolist(), rpy=list(sim.euler(d)),
                  active_joint_margin_rad=(1.4-abs(d.qpos[qa])).tolist(), contacts=forces)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    return result


def run():
    assert not any((OUT/name).exists() for name in ('progress.json', 'completion.json', 'failure.json'))
    np.testing.assert_array_equal(commands(np.zeros(6), np.zeros(6))['motor_0_1'], np.zeros(6))
    np.testing.assert_allclose(sim.euler(SimpleNamespace(qpos=np.array([0, 0, 0, 1, 0, 0, 0]))), [0, 0, 0], atol=0)
    reg, prepared = prepare()  # All source/data/command checks precede integration.
    sources = {str(Path(mod.__file__).resolve().relative_to(ROOT)): sha(mod.__file__)
               for mod in list(sys.modules.values()) if getattr(mod, '__file__', None)
               and Path(mod.__file__).resolve().is_relative_to(ROOT) and str(mod.__file__).endswith('.py')}
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    sources[str(Path(XML).relative_to(ROOT))] = sha(XML)
    write('source_contract.json', dict(verified=True, registration_sha256=sha(OUT/'registration.json'),
          source_sha256=sources, mujoco_version=mujoco.__version__, numpy_version=np.__version__,
          static_checks='All24 original states/rows, geometry/order/envelope, quaternion-zero and zero-clipped command fixture;0integration',
          contact_timestamp='Forces/contact descriptors are the solver pre-integration configuration; q/v/rpy are post-integration. No post-step forward recomputation.',
          physics_steps_before_contract=0))
    records = []; steps = 0
    try:
        for r, m, qa, state, bounds, ctrl, gpu in prepared:
            branches = {}
            for name, u in ctrl.items():
                steps += 1
                branches[name] = dict(ctrl=u.tolist(), applied_delta_Nm=(u-state[33:39]).tolist(),
                                      **advance(m, state, u, r['pre_time_s'], qa))
            steps += 1
            repeated = advance(m, state, ctrl['recorded_ctrl'], r['pre_time_s'], qa)
            baseline = branches['recorded_ctrl']
            for field in ('q', 'v', 'rpy', 'active_joint_margin_rad'):
                np.testing.assert_allclose(repeated[field], baseline[field], atol=1e-12, rtol=0)
            assert repeated['contacts'] == baseline['contacts']
            yaw = baseline['rpy'][2]
            for name, branch in branches.items():
                branch['yaw_minus_baseline_rad'] = float(np.arctan2(np.sin(branch['rpy'][2]-yaw), np.cos(branch['rpy'][2]-yaw)))
            records.append(dict(index=r['index'], case=r['case'], success_control=r['success'],
                bounds_Nm=bounds.tolist(), branches=branches, baseline_repeat_passed=True,
                cpu_minus_original_gpu_q=(np.array(baseline['q'])-gpu[:17]).tolist(),
                cpu_minus_original_gpu_v=(np.array(baseline['v'])-gpu[17:]).tolist(),
                midpoint_yaw_rad={str(j): .5*(branches[f'motor_{j}_1']['yaw_minus_baseline_rad']+
                                            branches[f'motor_{j}_-1']['yaw_minus_baseline_rad']) for j in range(6)}))
            write('progress.json', dict(status='running', states=len(records), cpu_steps=steps))
        assert steps == 336 and len(records) == 24
        assert all(sha(ROOT/n) == value for n, value in sources.items())
        write('completion.json', dict(verified=True, states=24, cpu_steps=steps, records=records,
              registration_sha256=sha(OUT/'registration.json'), source_contract_sha256=sha(OUT/'source_contract.json'),
              training_updates=0, new_gpu_rollouts=0, independent_review_pending=True,
              limits=reg['limitations']))
        write('progress.json', dict(status='complete', states=24, cpu_steps=steps))
        print('PASS226 24states,312arms+24repeat=336CPUsteps;review312pending', flush=True)
    except BaseException as e:
        write('failure.json', dict(error=repr(e), attempted_cpu_steps=steps, completed_states=len(records)))
        write('partial.json', dict(records=records)); raise


if __name__ == '__main__': run()
