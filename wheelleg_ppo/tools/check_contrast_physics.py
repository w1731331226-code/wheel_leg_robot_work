"""Check the frozen contrast bases through the real final actuator chain; no training."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import mujoco
import numpy as np
from ppo_env import MODES, XML, Residual, Scenario, WheelLegEnv, TORQUE_SCALE
from prepare_ppo import snapshot
import wheelleg_sim as sim

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT/'tools/results/contrast_physics_2026-09-21/ppo_env_before.py'
BASIS = ROOT/'tools/results/contrast_design_2026-09-21/basis.json'
METHODS = {'M1':'diff1','M3':'diff3','N3+':'mixed3_plus','N3-':'mixed3_minus',
           'B2-V':'virtual6','B2':'torque6'}


def clone(model, data, state):
    d = mujoco.MjData(model)
    mujoco.mj_copyData(d, model, data)
    return d, copy.deepcopy(state)


def stats(values):
    x = np.asarray(values)
    return dict(per_motor_rms_Nm=np.sqrt(np.mean(x*x, axis=0)).tolist(),
                normalized_rms=float(np.sqrt(np.mean((x/TORQUE_SCALE)**2))),
                covariance_Nm2=np.cov(x, rowvar=False).tolist())


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('previous_env', BEFORE)
    old = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    old.XML = XML
    historical = json.loads((ROOT/'tools/results/yaw_precision_v1_2026-09-20/protocol_manifest.json').read_text())
    assert hashlib.sha256(BEFORE.read_bytes()).hexdigest() == historical['source_sha256']['tools/ppo_env.py']
    for name, digest in historical['source_sha256'].items():
        if name != 'tools/ppo_env.py':
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    paths = [ROOT/'tools/ppo_env.py', ROOT/'tools/prepare_ppo.py', Path(__file__), BEFORE, BASIS]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    paired = np.clip(np.random.default_rng(610).normal(size=(256,3)), -1, 1)
    six = np.clip(np.random.default_rng(610).normal(size=(256,6)), -1, 1)
    bases = json.loads(BASIS.read_text())
    rows, contexts = [], {}
    for condition, scenario in [('flat',Scenario()),('unequal_height',Scenario(height_l=.02,center=1.5)),
                                ('single_contact',Scenario(height_l=.02,center=1.5))]:
        with contextlib.redirect_stdout(io.StringIO()):
            model, start, state, context = snapshot(scenario, condition)
        assert model.opt.timestep == .0005 and state.jp == 'DRIVE'
        contexts[condition] = context
        # The new bases use the existing virtual6 map as an independent reference.
        for method, mode in METHODS.items():
            residual = Residual(mode)
            for a in paired[:16]:
                if method in ('N3+','N3-'):
                    matrix = np.array(bases['N3_plus' if method=='N3+' else 'N3_minus'])
                    expected = Residual('virtual6').map(model,start,matrix @ a)
                    np.testing.assert_allclose(residual.map(model,start,a),expected,atol=1e-12)
                elif method == 'M1':
                    np.testing.assert_array_equal(residual.map(model,start,a[2:]),
                                                  Residual('diff3').map(model,start,np.array([0.,0.,a[2]])))
        for mode in old.MODES:
            actions = six if old.MODES[mode]==6 else paired[:,:old.MODES[mode]]
            for a in actions:
                np.testing.assert_array_equal(Residual(mode).map(model,start,a),old.Residual(mode).map(model,start,a))
        # Every mode with zero action reproduces the old diff3 physical trajectory.
        reference = []
        d, st = clone(model,start,state)
        for _ in range(80):
            sim.control(model,d,st,old.Residual('diff3')); mujoco.mj_step(model,d)
            reference.append((d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),st.motor_peak_t.copy()))
        for mode in MODES:
            d, st = clone(model,start,state); residual = Residual(mode)
            for q,dq,u,timers in reference:
                sim.control(model,d,st,residual); mujoco.mj_step(model,d)
                np.testing.assert_array_equal(d.qpos,q); np.testing.assert_array_equal(d.qvel,dq)
                np.testing.assert_array_equal(d.ctrl,u); assert st.motor_peak_t == timers
        for method, mode in METHODS.items():
            requests, actuals, lambdas, blocked, clipped = [], [], [], [], []
            actions = paired[:,2:] if method=='M1' else six if MODES[mode]==6 else paired
            for action in actions:
                d, st = clone(model,start,state); residual = Residual(mode)
                residual.set_action(action); residual.action[:] = action
                # Candidate map is recorded even when the base output blocks execution.
                requests.append(residual.map(model,d,action))
                sim.control(model,d,st,residual)
                assert np.all(d.ctrl >= residual.low-1e-9) and np.all(d.ctrl <= residual.high+1e-9)
                actual = d.ctrl-residual.base
                if not residual.final_clipped:
                    np.testing.assert_allclose(actual,residual.executed,atol=1e-12)
                actuals.append(actual); lambdas.append(residual.lam)
                blocked.append(residual.event == 'base_infeasible'); clipped.append(residual.final_clipped)
            rows.append(dict(condition=condition,method=method,request=stats(requests),actual=stats(actuals),
                        lambda_min=float(min(lambdas)),lambda_mean=float(np.mean(lambdas)),
                        saturation_fraction=float(np.mean(np.array(lambdas)<1-1e-10)),
                        base_blocked_fraction=float(np.mean(blocked)),final_clip_fraction=float(np.mean(clipped))))
        # Nonzero N3 inputs pass through slew, mapping, final limits and physical stepping.
        for mode in ('mixed3_plus','mixed3_minus'):
            d, st = clone(model,start,state); residual = Residual(mode)
            residual.set_action(np.array([.1,-.1,.1]))
            for _ in range(600):
                sim.control(model,d,st,residual)
                assert np.all(d.ctrl>=residual.low-1e-9) and np.all(d.ctrl<=residual.high+1e-9)
                mujoco.mj_step(model,d)
                assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        print(condition,'PASS',flush=True)
    for mode in ('mixed3_plus','mixed3_minus','diff1'):
        with contextlib.redirect_stdout(io.StringIO()):
            env=WheelLegEnv(mode=mode,scenario=Scenario()); obs,_=env.reset(seed=1)
            for action in (np.zeros(MODES[mode]),np.full(MODES[mode],.1)):
                obs,reward,_,_,_=env.step(action)
                assert env.observation_space.contains(obs) and np.isfinite(reward)
            try: env.step(np.full(MODES[mode],float('nan')))
            except ValueError: pass
            else: raise AssertionError('Invalid action accepted')
            env.close()
    contrasts=[]
    for condition in contexts:
        group={r['method']:r for r in rows if r['condition']==condition}
        for method in ('N3+','N3-'):
            base=group['M3']; value=group[method]
            contrasts.append(dict(condition=condition,method=method,
              requested_rms_ratio_to_M3=value['request']['normalized_rms']/base['request']['normalized_rms'],
              actual_rms_ratio_to_M3=(value['actual']['normalized_rms']/base['actual']['normalized_rms']
                                    if base['actual']['normalized_rms']>0 else None),
              saturation_delta=value['saturation_fraction']-base['saturation_fraction']))
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    result=dict(integration_pass=True,source_sha256=hashes,samples=256,seed=610,contexts=contexts,runs=rows,
                contrasts=contrasts,formal_training_started=False,gate_evaluated=False,test_set_evaluated=False,
                note='Steady targets at three known snapshots, not a learned-policy rollout or proof of matched covariance.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(contrasts,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)
