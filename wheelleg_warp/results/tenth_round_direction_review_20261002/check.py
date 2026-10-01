"""Recorded-evidence review and optimistic M3 actuator-authority bounds; no physics/learning."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.terrain import model,HeightTerrainScenario
from probe_height_115_contact_action_pair import basis
from probe_height_115_radial_authority import torque_box
from probe_height_115_action_predict_loow import sha
import wheelleg_sim as sim
OUT=Path(__file__).resolve().parent


def run():
    source=OUT.parent/'ninth_round_frozen_plan_panel_20261002'
    current=json.loads((source/'verification.json').read_text())
    old=json.loads((OUT.parent/'fifth_round_registered_robustness_20261002/verification.json').read_text())
    assert current['registration']['scenarios']==old['scenarios']
    assert all(r['reason']=='completed' for r in current['episodes']+old['candidate'])
    groups={}
    def yaw(row,scene):
        duration=3.5+1.5*row['task_goal_progress_m']/abs(scene['speed'])
        return row['rms_deg'][2]*np.sqrt(row['physical_steps']*.0005/duration)
    for group in dict.fromkeys(current['registration']['parameter_groups']):
        ids=[i for i,g in enumerate(current['registration']['parameter_groups']) if g==group]
        groups[group]=dict(old_tasks=sum(old['candidate'][i]['success'] for i in ids),new_tasks=sum(current['episodes'][i]['success'] for i in ids),
            full_gates=sum(current['original_full_gates'][i] for i in ids),
            old_mean_yaw_score_deg=float(np.mean([yaw(old['candidate'][i],old['scenarios'][i]) for i in ids])),
            new_mean_yaw_score_deg=float(np.mean([yaw(current['episodes'][i],old['scenarios'][i]) for i in ids])),
            minimum_active_design_margin_rad=min(current['active_design_margin_rad'][i] for i in ids))
    m=model(HeightTerrainScenario(stand_height_m=.115));authority=[]
    planfolder=OUT.parent/'braking_trajectory_slsqp_20261001'
    for world in range(2):
        p=planfolder/f'world{world}_best.npz';z=np.load(p,allow_pickle=False);steps=int(z['steps']);t=z['trace'][:steps]
        q=np.concatenate([z['initial_q'],t[:-1,:m.nq]]);v=np.concatenate([z['initial_v'],t[:-1,m.nq:m.nq+m.nv]])
        command=t[:,-12:-6];lambdas=[];amplitudes=[]
        for pose,speed,base in zip(q,v,command):
            b=basis(m,pose,speed)*np.array([1,1,-1,-1,1,-1])[:,None]
            b*=np.array([.1*sim.hw.DESIGN_MASS*9.81/2,1.,.3])[None,:]
            bounds=torque_box(m,speed)/np.array([1,1,1,1,1.05,1.05])
            assert np.max(abs(base)-bounds)<=1e-6
            row=[];peak=[]
            for axis in range(3):
                for sign in (1.,-1.):
                    delta=sign*b[:,axis];nonzero=abs(delta)>1e-12
                    lam=float(np.clip(np.min((bounds[nonzero]-np.sign(delta[nonzero])*base[nonzero])/abs(delta[nonzero])),0,1))
                    assert np.max(abs(base+lam*delta)-bounds)<=1e-6
                    row.append(lam);peak.append(float(np.max(abs(lam*delta))))
            lambdas.append(row);amplitudes.append(peak)
        a=np.array(lambdas);peak=np.array(amplitudes)
        elapsed=(z['initial_task'][0,0]+np.arange(steps))*.0005-z['initial_task'][0,1]
        stats={}
        for label,mask in [('whole_stop',np.ones(steps,bool)),('first_075s',elapsed<.75),('tail_from_15s',elapsed>=1.5)]:
            stats[label]={name:dict(mean_lambda=float(a[mask,j].mean()),fully_available_fraction=float(np.mean(a[mask,j]>=1-1e-10)),
                motor_effect_below_1e_minus6_Nm_fraction=float(np.mean(peak[mask,j]<=1e-6))) for j,name in enumerate(('F+','F-','H+','H-','yaw+','yaw-'))}
        authority.append(dict(world=world,samples=steps,stats=stats,input_sha256=sha(p)))
    result=dict(role='tenth_round_evidence_and_direction_review',paired_public_groups=groups,optimistic_m3_actuator_authority=authority,
        scope='Authority uses archived nominal predicted pre-step states and already accepted total commands as hypothetical shared Nom; it ignores original raw-base invalid flags and joint/geometry constraints, so is an upper bound, not current Actor authority or safe admissibility.',
        full_admission=False,learning_performed=False,primary_yaw_advantage_proven=False,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (source/'verification.json',source/'trace.npz',OUT.parent/'fifth_round_registered_robustness_20261002/verification.json')},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/probe_height_115_contact_action_pair.py',ROOT/'wheelleg_warp/probe_height_115_radial_authority.py')})
    report=OUT/'verification.json'
    if report.exists():
        recorded=json.loads(report.read_text())
        assert result['paired_public_groups']==recorded['paired_public_groups']
        assert result['optimistic_m3_actuator_authority']==recorded['optimistic_m3_actuator_authority']
        for name,h in recorded['input_sha256'].items():assert sha(ROOT/name)==h
        for name,h in recorded['source_sha256'].items():
            path=OUT/'source_at_run.py' if name==str(Path(__file__).relative_to(ROOT)) else ROOT/name
            assert sha(path)==h
    else:report.write_text(json.dumps(result,indent=2)+'\n')
    print('PAIRED',json.dumps(groups),flush=True)
    print('AUTHORITY',[(r['world'],r['stats']['first_075s']['yaw+'],r['stats']['first_075s']['F+']) for r in authority],flush=True)
    print('CHECKED frozen paired scope and original actuator inequalities; no learning/admission/advantage claim',flush=True)


if __name__=='__main__':run()
