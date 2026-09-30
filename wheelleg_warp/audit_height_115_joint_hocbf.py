"""CPU/offline joint-constraint comparison; no physics or controller promotion."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from audit_height_115_control_timing import load_cases
from probe_height_115_action_predict_loow import sha
from probe_height_115_live_braking import solve_from_acceleration,joint_hocbf_requirement,JOINT_HOCBF_RATE
from probe_height_115_braking_budget import barriers,DT,ACTIVE
from probe_height_115_passive import geometry
from native.terrain import model,HeightTerrainScenario


def self_check():
    k=JOINT_HOCBF_RATE;h0=.1;v0=-.1;b=v0+k*h0
    for t in (0.,DT,.005,.02):
        e=math.exp(-k*t);h=(h0+b*t)*e
        v=(b-k*(h0+b*t))*e;a=(k*k*(h0+b*t)-2*k*b)*e
        psi,required=joint_hocbf_requirement(h,v)
        assert h>=0 and psi>=0 and abs(a-required)<1e-8
    psi,required=joint_hocbf_requirement(.45,-.3)
    assert psi>0 and -45>=required and -45<.3**2/(2*.45)
    assert .45-.3*.005-.5*45*.005**2>0
    assert joint_hocbf_requirement(.0001,-.03)[0]<0


def run(output):
    assert not output.exists(),output;self_check()
    m,G,reserve,limits,qa,va,samples,inputs=load_cases();rows=[]
    for kind,t,q,v,a0,nominal,expected in samples:
        old,_,_,error=solve_from_acceleration(m,q,v,a0,G,reserve,nominal)
        assert (old is not None)==(expected is not None),(kind,t,error)
        if expected is not None:np.testing.assert_allclose(old,expected,atol=1e-8,rtol=0)
        row=dict(case=kind,step=t,legacy_feasible=old is not None,legacy_error=error)
        for name,hocbf,public in [('immediate_public',False,True),('joint_hocbf_public',True,True)]:
            action,bs,_,error=solve_from_acceleration(m,q,v,a0,G,reserve,nominal,
                joint_hocbf=hocbf,public_motor_box=public)
            row[name]=dict(feasible=action is not None,error=error,
                action=action.tolist() if action is not None else None,
                minimum_joint_psi1=min(joint_hocbf_requirement(b['h'],b['rate'])[0] for b in bs[2:]))
        rows.append(row)
    # Actual accelerations in a previously validated safe 20ms oracle hold,
    # used for constraint diagnosis only, never for choosing a new action.
    folder=ROOT/'wheelleg_warp/results/height_115_contact_hold_v2_20260929'
    meta=json.loads((folder/'verification.json').read_text());trace=folder/'trace.npz'
    assert sha(trace)==meta['trace_sha256']
    with np.load(trace) as z:pre=z['pre'][1].copy();post=z['post'][1].copy();vn=z['post_velocity'][1].copy()
    held=[]
    for t in range(40):
        q=pre[t,:m.nq];v=pre[t,m.nq:m.nq+m.nv]
        acceleration=(vn[t,va]-v[va])/DT
        bs,_=barriers(q[qa],v[va],acceleration,np.zeros_like(G),reserve)
        immediate=[b['a0']-b['rate']**2/(2*b['h']) for b in bs[2:] if b['rate']<0 and b['h']>0]
        psi=[joint_hocbf_requirement(b['h'],b['rate'])[0] for b in bs[2:]]
        slacks=[b['a0']-joint_hocbf_requirement(b['h'],b['rate'])[1] for b in bs[2:]]
        joint_ids=[m.joint(n).id for n in ACTIVE]
        margin=float(np.minimum(post[t,m.jnt_qposadr[joint_ids]]-limits[:,0],limits[:,1]-post[t,m.jnt_qposadr[joint_ids]]).min())
        A=geometry(m,post[t])[1]
        assert margin>=0 and A.min()>=.1147044660616607
        held.append(dict(step=t,minimum_joint_margin_rad=margin,true_A_min_m=float(A.min()),
            immediate_joint_slack=min(immediate) if immediate else None,
            joint_hocbf_minimum_psi1=min(psi),joint_hocbf_minimum_acceleration_slack=min(slacks)))
    summary=dict(cases=len(rows),legacy_feasible=sum(r['legacy_feasible'] for r in rows),
        immediate_public_feasible=sum(r['immediate_public']['feasible'] for r in rows),
        joint_hocbf_public_feasible=sum(r['joint_hocbf_public']['feasible'] for r in rows),
        safe_hold_steps=len(held),safe_hold_immediate_joint_rejections=sum(r['immediate_joint_slack'] is not None and r['immediate_joint_slack']<0 for r in held),
        safe_hold_hocbf_joint_rejections=sum(r['joint_hocbf_minimum_psi1']<0 or r['joint_hocbf_minimum_acceleration_slack']<0 for r in held))
    inputs.extend([folder/'verification.json',trace])
    # Six-world alarm states, with the same fold G/reserves and no future outcome selection.
    base=ROOT/'wheelleg_warp/results'
    fitpath=base/'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    arcpath=base/'height_115_local_states_20260928/verification.json'
    alarmpath=base/'height_115_alarm_timing_integer_20260929/verification.json'
    fitted,archive,alarm=[json.loads(p.read_text()) for p in (fitpath,arcpath,alarmpath)]
    windows=arcpath.parent/'windows.npz'
    assert sha(windows)==archive['windows_sha256']==alarm['windows_sha256']
    assert sha(fitpath)==alarm['fitted_verification_sha256']
    world_rows=[];cols=archive['columns']
    with np.load(windows) as data:
        for world,event_id in enumerate(fitted['selected_event_ids']):
            event=archive['events'][event_id];wm=model(HeightTerrainScenario(**event['scenario']))
            pre=data[f'event_{event_id}_pre'];qa=np.array([wm.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([wm.joint(n).dofadr[0] for n in ACTIVE])
            fold=fitted['folds'][world];gain=np.array(fold['gain']);r=fold['reserves'];rsv=(r['actual_A_length_m'],r['eight_joint_margin_rad'])
            for label,time in [('15ms_before_failure',event['reference_s']-.015),('first_alarm',alarm['rows'][world]['first_alarm_s'])]:
                index=int(np.argmin(abs(pre[:,0]-time)));assert index>0 and abs(pre[index,0]-time)<1e-6
                q=pre[index,cols['qpos_start']:cols['qvel_start']];v=pre[index,cols['qvel_start']:cols['warmstart_start']]
                a0=(v[va]-pre[index-1,cols['qvel_start']+va])/DT
                nominal=pre[index,cols['ctrl_start']:cols['ctrl_start']+6]
                item=dict(world=world,state=label,time_s=time)
                for name,hocbf in [('immediate_public',False),('joint_hocbf_public',True)]:
                    c,bs,_,error=solve_from_acceleration(wm,q,v,a0,gain,rsv,nominal,joint_hocbf=hocbf,public_motor_box=True)
                    item[name]=dict(feasible=c is not None,error=error,action=c.tolist() if c is not None else None,
                        minimum_joint_psi1=min(joint_hocbf_requirement(b['h'],b['rate'])[0] for b in bs[2:]))
                world_rows.append(item)
    summary['six_world_states']={label:{name:sum(r[name]['feasible'] for r in world_rows if r['state']==label)
        for name in ('immediate_public','joint_hocbf_public')} for label in ('15ms_before_failure','first_alarm')}
    summary['joint_hocbf_initial_set_rejections']=sum(str(r['joint_hocbf_public']['error']).startswith('joint_velocity_outside') for r in world_rows)
    inputs.extend([fitpath,arcpath,alarmpath,windows])
    heldout=fitpath.parent/'heldout_inputs.npz';assert sha(heldout)==fitted['heldout_inputs_sha256']
    proxy_rows=[]
    with np.load(heldout) as z:
        for i in range(12):
            fold=fitted['folds'][i//2];gain=np.array(fold['gain']);r=fold['reserves']
            # Acceleration is a first-step interval average, not an instantaneous derivative certificate.
            observed=(z['first_step_active_v'][i]-z['v0'][i])/DT
            predicted=(z['v0'][i]-z['vprev'][i])/DT+z['actions'][i]@gain.T
            for arm in range(9):
                bs,_=barriers(z['q0'][i],z['v0'][i],predicted[arm],gain,(r['actual_A_length_m'],r['eight_joint_margin_rad']))
                h=np.array([b['h'] for b in bs[2:]]);rate=np.array([b['rate'] for b in bs[2:]])
                model_acc=np.stack([-predicted[arm],predicted[arm]],axis=1).ravel()
                true_acc=np.stack([-observed[arm],observed[arm]],axis=1).ravel()
                requirements=[joint_hocbf_requirement(x,v) for x,v in zip(h,rate)]
                member=bool(np.all(h>0) and min(p for p,_ in requirements)>=0)
                required=np.array([a for _,a in requirements])
                predicted_pass=bool(member and np.all(model_acc>=required))
                actual_pass=bool(member and np.all(true_acc>=required))
                proxy_rows.append(dict(world=i//2,lag_ms=(15,5)[i%2],arm=arm,initial_set_member=member,
                    predicted_joint_proxy_pass=predicted_pass,actual_joint_proxy_pass=actual_pass,
                    minimum_actual_joint_proxy_slack=float((true_acc-required).min())))
    summary['heldout_joint_first_step_proxy']=dict(actions=len(proxy_rows),
        initial_set_rejections=sum(not r['initial_set_member'] for r in proxy_rows),
        predicted_pass=sum(r['predicted_joint_proxy_pass'] for r in proxy_rows),
        actual_pass=sum(r['actual_joint_proxy_pass'] for r in proxy_rows),
        false_safe=sum(r['predicted_joint_proxy_pass'] and not r['actual_joint_proxy_pass'] for r in proxy_rows))
    inputs.append(heldout)
    paths=[Path(__file__),ROOT/'wheelleg_warp/probe_height_115_live_braking.py',ROOT/'wheelleg_warp/probe_height_115_braking_budget.py',
        ROOT/'wheelleg_warp/audit_height_115_control_timing.py',ROOT/'wheelleg_warp/native/environment.py']
    paths.extend(ROOT/p for p in ('wheelleg_warp/probe_height_115_contact_action_pair.py','wheelleg_warp/probe_height_115_radial_authority.py',
        'wheelleg_warp/probe_height_115_passive.py','wheelleg_warp/probe_height_115_margin.py',
        'wheelleg_warp/native/controller.py','wheelleg_warp/native/models.py','wheelleg_warp/native/terrain.py',
        'wheelleg_ppo/tools/state_estimation.py','wheelleg_ppo/tools/wheelleg_sim.py','wheelleg_ppo/tools/hardware_profile.py','wheelleg_ppo/xml/wheelleg.xml'))
    result=dict(role='offline_conditional_joint_HOCBF_ablation_not_safety_certificate',summary=summary,
        joint_hocbf_rate_per_s=JOINT_HOCBF_RATE,rate_rule='inverse existing5ms predictor horizon, fixed before outcomes',
        joint_condition='h>=0; psi1=rate+k*h>=0; acceleration+2*k*rate+k*k*h>=0',
        leg_condition='unchanged immediate constant-acceleration stopping rows',
        motor_condition='unchanged1Nm increment, absolute nominal command box tightened by public wheel gain upper1.05',
        scope='Archived44 feasible and2 failed LP fixtures, and old40-step safe oracle hold. No new physics, no current-controller performance or recursive-feasibility claim.',
        limitations='Continuous joint invariance is conditional on correct dynamics and maintained inequalities. Frozen G, a0 errors, positional reserves, passive limits, sample-and-hold/latency and initial oracle remain unvalidated. The initial psi1 check can reject recovery from outside the set.',
        rows=rows,safe_hold_rows=held,six_world_rows=world_rows,heldout_joint_proxy_rows=proxy_rows,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS',summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
