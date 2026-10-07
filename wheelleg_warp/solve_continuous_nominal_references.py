"""Registered20-point solve; reuse original problem, bounds and residual counter."""
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
import solve_moving_equilibrium as original
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = original.OUT.parent / 'continuous_nominal_reference_v1'
sim = original.sim


def constraints(m, d):
    names = ('alphaL','betaL','alphaR','betaR','passA_L','passC_L','passA_R','passC_R')
    ids = [m.joint(n).id for n in names]
    angles = d.qpos[m.jnt_qposadr[ids]]
    dofs = [m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
    limits = np.array([sim.hw.torque_limit(float('inf'), float(d.qvel[n]), j<4, 0., .0005)[0]
                       for j, n in enumerate(dofs)])
    lengths, loops = [], []
    for side in ('L','R'):
        hip = (d.xanchor[m.joint('alpha'+side).id]+d.xanchor[m.joint('beta'+side).id])/2
        center = d.xpos[m.body('wheel'+side).id]
        end = d.site_xpos[m.site('couplerB_'+side+'_end').id]
        lengths.extend([float(np.linalg.norm(center-hip)), float(np.linalg.norm(end-hip))])
        loops.append(float(np.linalg.norm(center-end)))
    floor = m.geom('floor').id
    wheels = {m.geom('wheel_collide_L').id, m.geom('wheel_collide_R').id}
    return dict(active_margin_rad=float((1.4-abs(angles[:4])).min()),
        eight_margin_rad=float(np.minimum(angles-m.jnt_range[ids,0],m.jnt_range[ids,1]-angles).min()),
        command_excess_Nm=float(np.maximum(abs(d.ctrl)-limits,0).max()),
        actual_torque_excess_Nm=float(np.maximum(abs(d.actuator_force)-limits,0).max()),
        minimum_actual_leg_m=min(lengths), maximum_loop_error_m=max(loops),
        bilateral_only_floor_support=bool(d.ncon==2 and all(floor in set(c.geom) and
            (set(c.geom)-{floor}).issubset(wheels) for c in d.contact)))


def run():
    assert not any((OUT / n).exists() for n in ('source_admission.json','batch_started.json','completion.json','failure.json'))
    started = time.monotonic()
    p = json.loads((OUT / 'proposal.json').read_text())
    assert len(p['new_reference_points'])==20 and p['solver_residual_calls_maximum']==24000
    assert p['max_nfev_per_point']==100 and not p['integration_admitted'] and not p['training_admitted']
    assert all(sha(ROOT / n)==h for n,h in p['source_sha256'].items())
    _, m, _ = original.setup()
    seeds = []
    for r in p['existing_moving_references']:
        path = ROOT / r['path']
        assert sha(path)==r['sha256']
        with np.load(path, allow_pickle=False) as z:
            seeds.append(dict(**r, q=z['q'].copy(), ctrl=z['ctrl'].copy(), solution=z['solution'].copy()))
    counter = dict(used=0, limit=24000)
    records, unit = [], []
    try:
        for r in seeds:
            _, _, _, f, data = original.problem(m, r, r['height'], r['speed'], counter)
            error = f(r['solution'])
            mujoco.mj_forward(m, data)
            assert abs(error).max()<=1e-6 and abs(data.qacc).max()<=1e-4
            unit.append(dict(height=r['height'], speed=r['speed'], maximum_residual=float(abs(error).max())))
        prepared = []
        for point in p['new_reference_points']:
            seed = min(seeds, key=lambda r:(abs(r['height']-point['height']),abs(r['speed']-point['speed'])))
            x, lo, hi, f, data = original.problem(m, seed, point['height'], point['speed'], counter)
            assert np.all((x>lo)&(x<hi)) and np.isfinite(f(x)).all()
            prepared.append((point, seed, x, lo, hi, f, data))
        assert counter['used']==30
        x, _, _, f, _ = original.problem(m, seeds[0], .115, -1., dict(used=0,limit=0))
        try: f(x)
        except RuntimeError: pass
        else: raise AssertionError('Counter exhaustion accepted')
        try: prepared[0][5](np.full(10,np.nan))
        except ValueError: pass
        else: raise AssertionError('NaN accepted')
        atomic_json(OUT / 'source_admission.json', dict(
            verified=True, proposal_sha256=sha(OUT / 'proposal.json'), runner_sha256=sha(__file__),
            reused_solver_sha256=sha(original.__file__), source_sha256=p['source_sha256'],
            old10_reference_replay_checks=unit, unit_residual_calls=counter['used'],
            total_residual_limit_including_unit=24000, new_initializations_checked=20,
            initial_reference_paths=[seed['path'] for _,seed,*_ in prepared],
            optimization_calls=0, integration_steps=0, transitionFD_calls=0,
            solve_batch_admitted=True, map_or_controller_admitted=False))
        atomic_json(OUT / 'batch_started.json', dict(
            admission_sha256=sha(OUT / 'source_admission.json'), initial_residual_calls=counter['used']))
        for i, (point, seed, x, lo, hi, f, data) in enumerate(prepared):
            before = counter['used']
            result = least_squares(f, x, bounds=(lo,hi), xtol=1e-12, ftol=1e-12,
                                  gtol=1e-12, max_nfev=100, x_scale='jac')
            residual = f(result.x)
            mujoco.mj_forward(m, data)
            qacc = data.qacc.copy()
            values = constraints(m, data)
            path = OUT / f'reference_{i:02d}.npz'
            np.savez_compressed(path, initial=x, solution=result.x, lower=lo, upper=hi,
                residual=residual, q=data.qpos.copy(), v=data.qvel.copy(), ctrl=data.ctrl.copy(),
                qacc=qacc, qfrc_inverse=data.qfrc_inverse.copy(), qfrc_actuator=data.qfrc_actuator.copy(),
                qfrc_passive=data.qfrc_passive.copy(), actuator_force=data.actuator_force.copy(),
                qacc_warmstart=data.qacc_warmstart.copy())
            passed = bool(result.success and abs(residual).max()<=1e-6 and abs(qacc).max()<=1e-4 and
                values['active_margin_rad']>=0 and values['eight_margin_rad']>=0 and
                max(values['command_excess_Nm'],values['actual_torque_excess_Nm'])<=1e-6 and
                values['minimum_actual_leg_m']>=.1147044660616607 and values['bilateral_only_floor_support'])
            record = dict(**point, path=path.name, sha256=sha(path), initializer_path=seed['path'],
                initializer_sha256=seed['sha256'], solver_success=bool(result.success),
                solver_status=int(result.status), nfev=int(result.nfev),
                actual_residual_calls=counter['used']-before,
                force_residual_max_abs=float(abs(residual).max()),
                forward_qacc_max_abs=float(abs(qacc).max()), **values, passed=passed)
            records.append(record)
            atomic_json(OUT / 'progress.json', dict(completed_states=len(records),residual_calls=counter['used']))
            print('REFERENCE', i, point['height'], point['speed'], point['role'], passed,
                  'calls',counter['used'],'force',record['force_residual_max_abs'],flush=True)
            if not passed:
                raise RuntimeError('Original reference gate failed; registered stopping rule')
        assert len(records)==20 and counter['used']<=24000
        assert all(sha(ROOT / n)==h for n,h in p['source_sha256'].items())
        atomic_json(OUT / 'completion.json', dict(
            verified=True, records=records, residual_calls=counter['used'], unit_residual_calls=30,
            optimization_states=20, boundary_anchors=10, validation_not_fit=10,
            admission_sha256=sha(OUT / 'source_admission.json'), runner_sha256=sha(__file__),
            total_seconds=time.monotonic()-started, integration_steps=0, transitionFD_calls=0,
            new_training_samples=0, controller_admitted=False, table_fit_executed=False,
            limits='Cached solver-data forward and original gates only; independent freshcold review required. Validation points never fit. No interpolation/GPU/staticquery/task/PPO admission.'))
        print('DONE27420/20 original gates; calls',counter['used'],'/24000; no integration/FD/PPO',flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(error=repr(error), residual_calls=counter['used'],
                   records=records, implicit_retry=False))
        raise


if __name__=='__main__':
    run()
