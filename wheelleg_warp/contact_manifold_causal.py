"""Frozen CPU stage pipeline for bounded contact-manifold interventions."""
import json
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
from probe_holdout_factors import inputs as new_inputs
from check_contact_response import extract_current
from probe_contact_uncertainty import construct
from native.terrain import HeightTerrainScenario,model
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004'
OUT=BASE/'contact_manifold_causal_v1'
OLD=BASE/'joint_response_v1'
TOL=1e-12
FIELDS=['qpos','qvel','qacc','qacc_warmstart','ctrl','actuator_force','qfrc_constraint']


def entries():
    for label,i,state,ids,old,scene in new_inputs():
        yield 'new_saved_states',label,i,state,ids,old['contacts'],scene,old['case'],old['event']
    reg=json.loads((OLD/'registration.json').read_text());c=json.loads((OLD/'completion.json').read_text())
    total=0
    for job in c['records']:
        path=OLD/f'{job["label"]}_states.npz';assert sha(path)==job['states_sha256']
        with np.load(path,allow_pickle=False) as z:
            for i,state in enumerate(z['state']):
                world=int(z['world'][i]);scene=HeightTerrainScenario(**reg['cases'][world]['scenario']);total+=1
                # These old records are static fixture inputs, not newly qualified GPU evidence.
                expected=(state[7:17].astype(np.float32)+np.float32(.0005)*state[91:101].astype(np.float32)).astype(np.float32)
                np.testing.assert_array_equal(expected,state[75:85])
                yield 'old_critical_fixtures',job['label'],i,state,z['ids'],extract_current(scene,state[:17]),scene,reg['cases'][world]['seed'],int(z['event'][i])
    assert total==27


def data(m,state):
    assert m.nq==17 and m.nv==16 and m.na==m.nmocap==m.nplugin==0 and int(m.opt.integrator)==3
    assert mujoco.get_mjcb_control() is None and mujoco.get_mjcb_passive() is None
    d=mujoco.MjData(m);d.qpos[:]=state[:17];d.qvel[:]=state[17:33];d.ctrl[:]=state[50:56];d.qacc_warmstart[:]=0.;d.time=0.
    assert np.all(d.qfrc_applied==0) and np.all(d.xfrc_applied==0)
    return d


def advance(m,d,contact_override=None):
    mujoco.mj_fwdPosition(m,d)
    if contact_override is not None:contact_override(m,d)
    # Rebuild in every arm, including the untouched control; no later collision pass.
    mujoco.mj_makeConstraint(m,d);mujoco.mj_projectConstraint(m,d)
    mujoco.mj_fwdVelocity(m,d);mujoco.mj_fwdActuation(m,d)
    mujoco.mj_fwdAcceleration(m,d);mujoco.mj_fwdConstraint(m,d)
    assert int(m.opt.integrator)==3
    mujoco.mj_implicit(m,d)


def freeze():
    reg=json.loads((OUT/'registration.json').read_text());assert reg['total_states']==603 and reg['compatibility_step_budget']==2412 and not (OUT/'source_contract.json').exists()
    parent=json.loads((BASE/'contact_manifold_audit_v1/registration.json').read_text());sources={**parent['source_sha256'],'wheelleg_warp/contact_manifold_causal.py':sha(__file__)}
    assert all(sha(ROOT/n)==v for n,v in sources.items())
    for cohort in reg['state_cohorts']:assert sha(BASE/cohort['source'])==cohort['sha256']
    atomic_json(OUT/'source_contract.json',dict(registration_sha256=sha(OUT/'registration.json'),source_sha256=sources,identity_fields=FIELDS,full_qv_tolerance=TOL,
        pipeline='fwdPosition -> optional offline override -> makeConstraint/projectConstraint -> fwdVelocity/Actuation/Acceleration/Constraint -> implicit(integrator3)',
        legacy_fixture_provenance='Old collector pre/stats owners not explicitly retained. All27 saved hinge updates checked; use immutable numerical fixtures only, not new reliable GPU-event evidence. Original files preserved, no recapture.',
        standard_control='Fresh mj_step and reconstructed pipeline from separate identical MjData, no control/passive callback, activation, mocap, plugin or applied-force state.',training_updates=0))
    print('FROZEN CPU pipeline before any integration',flush=True)


def contract():
    p=json.loads((OUT/'source_contract.json').read_text());assert p['registration_sha256']==sha(OUT/'registration.json') and all(sha(ROOT/n)==v for n,v in p['source_sha256'].items());return p


def compatibility():
    contract();assert not (OUT/'compatibility_progress.json').exists();rows=[];steps=0;maxima={name:0. for name in FIELDS}
    try:
        for cohort,label,i,state,ids,contacts,scene,case,event in entries():
            parameters=[scene.mass,scene.mu_l,scene.mu_r,scene.drive_difference]
            for kind,m in [('actual_geometry',model(scene)),('local_plane',construct(state[:17],contacts,parameters))]:
                standard=data(m,state);manual=data(m,state)
                mujoco.mj_step(m,standard);steps+=1;advance(m,manual);steps+=1
                result=dict(cohort=cohort,label=label,index=i,case=case,event=event,model=kind,standard={},manual={},errors={})
                for name in FIELDS:
                    a=getattr(standard,name).copy();b=getattr(manual,name).copy();error=float(np.max(np.abs(a-b)));maxima[name]=max(maxima[name],error)
                    result['standard'][name]=a.tolist();result['manual'][name]=b.tolist();result['errors'][name]=error
                    np.testing.assert_allclose(b,a,rtol=0,atol=TOL)
                assert standard.time==manual.time==.0005 and standard.ncon==manual.ncon and standard.nefc==manual.nefc
                result.update(time_s=float(manual.time),ncon=int(manual.ncon),nefc=int(manual.nefc));rows.append(result)
            atomic_json(OUT/'compatibility_progress.json',dict(status='running',states=len(rows)//2,cpu_steps=steps,last_cohort=cohort,last_label=label))
            if i==143:print('COMPATIBLE',cohort,label,len(rows)//2,flush=True)
        assert len(rows)==1206 and steps==2412;contract()
        atomic_json(OUT/'compatibility.json',dict(verified=True,states=603,model_state_pairs=1206,cpu_steps=steps,records=rows,maxima=maxima,source_contract_sha256=sha(OUT/'source_contract.json'),training_updates=0,new_gpu_steps=0))
        atomic_json(OUT/'compatibility_progress.json',dict(status='complete',states=603,cpu_steps=steps));print('PASS603 full-state identity checks',maxima,flush=True)
    except BaseException as e:
        atomic_json(OUT/'compatibility_failure.json',dict(error=str(e),cpu_steps=steps,completed_model_state_pairs=len(rows),maxima=maxima,intervention_admitted=False))
        atomic_json(OUT/'compatibility_partial.json',dict(records=rows));raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','compatibility']);globals()[p.parse_args().command]()
