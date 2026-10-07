"""One frozen20-point validation batch; validation data never fit a map."""
import json
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
import diagnose_geometry_seed as diagnostic
from solve_continuous_nominal_references import constraints
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

original = diagnostic.original
sim = diagnostic.sim
OUT = diagnostic.OUT.parent / 'geometry_seed_validation_v1'


def run():
    assert not any((OUT/n).exists() for n in ('source_admission.json','started.json','completion.json','failure.json'))
    started=time.monotonic()
    p=json.loads((OUT/'proposal.json').read_text())
    assert p['reference_count']==len(p['points'])==20 and p['max_nfev_per_point']==100
    assert p['per_point_residual_limit']==1200 and p['total_residual_limit']==24000
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    m,_=sim.load_model(original.ml.XML,True)
    assert (m.nq,m.nv,m.nu)==(17,16,6) and abs(m.body_mass.sum()-7.)<1e-12
    assert m.opt.timestep==.0005 and m.opt.iterations==100 and int(m.opt.integrator)==3
    assert mujoco.get_mjcb_control() is None and mujoco.get_mjcb_passive() is None
    donors=[]
    for r in p['existing_moving_reference_inputs']:
        assert sha(ROOT/r['path'])==r['sha256']
        with np.load(ROOT/r['path']) as z:donors.append(dict(**r,q=z['q'].copy(),ctrl=z['ctrl'].copy()))
    prepared,checks=[],[]
    for point in p['points']:
        donor=min(donors,key=lambda r:(abs(r['height']-point['height']),abs(r['speed']-point['speed'])))
        counter=dict(used=0,limit=1200)
        old,lo,hi,f,data=original.problem(m,donor,point['height'],point['speed'],counter)
        new=old.copy()
        new[0]+=point['height']-sim.fk_joints(old[2],old[4])['leg_len']
        new[2:6]=diagnostic.geometry(m,point['height'])
        np.testing.assert_array_equal(new[[1,6,7,8,9]],old[[1,6,7,8,9]])
        assert np.all((new>lo)&(new<hi))
        assert abs(sim.fk_joints(new[2],new[4])['leg_len']-point['height'])<=1e-12
        checks.append(dict(**point,donor_path=donor['path'],old_initial=old.tolist(),new_initial=new.tolist()))
        prepared.append((point,donor,counter,old,new,lo,hi,f,data))
    atomic_json(OUT/'source_admission.json',dict(
        verified=True,proposal_sha256=sha(OUT/'proposal.json'),runner_sha256=sha(__file__),
        source_sha256=p['source_sha256'],initial_geometry_checks=checks,unit_residual_calls=0,
        no_fit=True,solver_max_nfev=100,per_point_limit=1200,total_limit=24000))
    atomic_json(OUT/'started.json',dict(admission_sha256=sha(OUT/'source_admission.json')))
    records=[];calls=0
    try:
        for i,(point,donor,counter,old,new,lo,hi,f,data) in enumerate(prepared):
            initial_residual=f(new)
            result=least_squares(f,new,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,
                                 max_nfev=100,x_scale='jac')
            error=f(result.x);mujoco.mj_forward(m,data)
            values=constraints(m,data)
            passed=bool(result.success and abs(error).max()<=1e-6 and abs(data.qacc).max()<=1e-4 and
                values['active_margin_rad']>=0 and values['eight_margin_rad']>=0 and
                max(values['command_excess_Nm'],values['actual_torque_excess_Nm'])<=1e-6 and
                values['minimum_actual_leg_m']>=.1147044660616607 and values['bilateral_only_floor_support'])
            path=OUT/f'reference_{i:02d}.npz'
            np.savez_compressed(path,old_initial=old,initial=new,initial_residual=initial_residual,
                solution=result.x,lower=lo,upper=hi,residual=error,q=data.qpos.copy(),v=data.qvel.copy(),
                ctrl=data.ctrl.copy(),qacc=data.qacc.copy(),actuator_force=data.actuator_force.copy(),
                qfrc_inverse=data.qfrc_inverse.copy(),qfrc_actuator=data.qfrc_actuator.copy(),
                qfrc_passive=data.qfrc_passive.copy(),qacc_warmstart=data.qacc_warmstart.copy())
            calls+=counter['used'];assert counter['used']<=1200 and calls<=24000
            record=dict(**point,path=path.name,sha256=sha(path),donor_path=donor['path'],
                donor_sha256=donor['sha256'],solver_success=bool(result.success),solver_status=int(result.status),
                nfev=int(result.nfev),residual_calls=counter['used'],passed=passed,
                combined_residual_max=float(abs(error).max()),projected_force_max=float(abs(error[:8]).max()),
                height_error_m=float(error[-2]/100),pitch_rad=float(error[-1]),
                forward_qacc_max=float(abs(data.qacc).max()),**values)
            records.append(record)
            atomic_json(OUT/'progress.json',dict(completed_states=len(records),residual_calls=calls))
            print('VALIDATE',i,point['height'],point['speed'],passed,'nfev',result.nfev,'calls',calls,flush=True)
            if not passed:raise RuntimeError('Original gate failed; stop frozen validation batch')
        assert len(records)==20 and all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
        atomic_json(OUT/'completion.json',dict(
            verified=True,records=records,passed=20,residual_calls=calls,optimization_states=20,
            total_seconds=time.monotonic()-started,admission_sha256=sha(OUT/'source_admission.json'),
            no_fit=True,integration_steps=0,transitionFD_calls=0,new_training_samples=0,map_admitted=False,
            limits='Twentypredeclared developmentreference points,cachedsolver forward only. Independentfreshcold279 pending. No tablefit/controller/trajectory/finalOOD/PPO admission.'))
        print('DONE27820/20 cached originalgates; calls',calls,'/24000; no fit/steps/FD/PPO',flush=True)
    except BaseException as error:
        # Include an interrupted point's residual cost, even before its record is saved.
        used=sum(c['used'] for _,_,c,*_ in prepared)
        atomic_json(OUT/'failure.json',dict(error=repr(error),residual_calls=used,records=records,
            unattempted_points=p['points'][len(records):],implicit_retry=False))
        raise


if __name__=='__main__':
    run()
