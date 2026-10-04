"""Fixed2x2x2 offline factor study; actual factors are diagnostic privileges."""
from pathlib import Path
from dataclasses import replace
from itertools import product
import json,sys
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
from native.terrain import HeightTerrainScenario,model
from native.models import build_spec,compile_spec

DATA=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_response_v1'
OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/prediction_factors_v1'


def construct(scene,q,actual_geometry,actual_parameters):
    s=scene if actual_parameters else replace(scene,mass=7.,mu_l=.8,mu_r=.8,drive_difference=0.)
    if actual_geometry:return model(s)
    # Same failed lower-flat hypothesis, geometry from current pose only.
    s=replace(s,terrain='legacy',height_l=0.,height_r=0.,grade_deg=0.,roughness_m=0.,step_height_m=0.,transition_run_m=0.,lateral_margin_m=0.)
    spec=build_spec(s);initial=compile_spec(spec,s);d=mujoco.MjData(initial);d.qpos[:]=q;mujoco.mj_kinematics(initial,d)
    bottoms=[]
    for side in ['L','R']:
        g=initial.geom('wheel_collide_'+side).id;axis=d.geom_xmat[g].reshape(3,3)[:,1];size=initial.geom_size[g]
        support=np.sqrt(size[0]**2+(size[1]**2-size[0]**2)*axis[2]**2)
        bottoms.append(float(d.geom_xpos[g,2]-support))
    spec.geom('floor').pos[2]=min(bottoms)
    return compile_spec(spec,s)


def run():
    prior=json.loads((DATA/'registration.json').read_text());review=json.loads((DATA/'review.json').read_text());ledger=json.loads((DATA/'completion.json').read_text())
    assert review['verified']
    OUT.mkdir(parents=True,exist_ok=False)
    reg=dict(version='prediction-factorial-v1',factors=['actual_geometry','actual_parameters_mass_mu_drive','actual_warmstart'],factor_levels=[False,True],states=27,cpu_steps=216,
        selection='All27 prior captured states, no model/state/error filtering.',
        baseline='State-derived lower flat plane, fixed7kg/friction.8/drive0, zero warmstart. This repeats an already failed hypothesis solely as the0/0/0 factor control.',
        privileges='Every actual factor is offline diagnostic truth, never actor/controller input or a deployable predictor; no factor winner is promoted.',
        source_sha256=sha(__file__),response_review_sha256=sha(DATA/'review.json'),nominal_review_sha256=sha(ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_response_v1/review.json'),training_updates=0,old_gate_or_final_used=False)
    (OUT/'registration.json').write_text(json.dumps(reg,indent=2)+'\n');records=[]
    for rec in ledger['records']:
        label=rec['label'];path=DATA/(label+'_states.npz');assert sha(path)==rec['states_sha256'];z=np.load(path,allow_pickle=False);ids=z['ids']
        for i,state in enumerate(z['state']):
            scene=HeightTerrainScenario(**prior['cases'][int(z['world'][i])]['scenario'])
            for geometry,parameters,warm in product([False,True],repeat=3):
                m=construct(scene,state[:17],geometry,parameters);d=mujoco.MjData(m)
                d.qpos[:]=state[:17];d.qvel[:]=state[17:33];d.ctrl[:]=state[50:56];d.qacc_warmstart[:]=state[33:49] if warm else 0.;d.time=state[49]
                mujoco.mj_step(m,d);pred=d.qpos[ids[:4]];actual=state[68:85][ids[:4]];error=pred-actual
                records.append(dict(label=label,index=i,event=int(z['event'][i]),world=int(z['world'][i]),
                    factors=[geometry,parameters,warm],predicted_q=pred.tolist(),actual_gpu_q=actual.tolist(),error_rad=error.tolist(),max_error_rad=float(abs(error).max()),
                    false_safe=bool(abs(pred).max()<=1.4 and abs(actual).max()>1.4)))
        print('COMPLETED factors',label,len(z['state'])*8,flush=True)
    assert len(records)==216
    summaries={}
    for flags in product([False,True],repeat=3):
        subset=[r for r in records if r['factors']==list(flags)];key='/'.join('1' if f else '0' for f in flags)
        summaries[key]=dict(states=len(subset),max_error_rad=max(r['max_error_rad'] for r in subset),mean_max_error_rad=float(np.mean([r['max_error_rad'] for r in subset])),false_safe=sum(r['false_safe'] for r in subset))
    effects=[]
    for factor in range(3):
        diffs=[]
        for flags in product([False,True],repeat=3):
            if flags[factor]:continue
            changed=list(flags);changed[factor]=True;a='/'.join('1' if f else '0' for f in flags);b='/'.join('1' if f else '0' for f in changed)
            diffs.append(dict(other_factors=list(flags),max_error_change_rad=summaries[b]['max_error_rad']-summaries[a]['max_error_rad'],false_safe_change=summaries[b]['false_safe']-summaries[a]['false_safe']))
        effects.append(dict(factor=reg['factors'][factor],stratified_changes=diffs))
    (OUT/'result.json').write_text(json.dumps(dict(verified_producer=True,cpu_steps=216,records=records,summary=summaries,stratified_effects=effects,registration_sha256=sha(OUT/'registration.json'),
        limits='Factor interactions and support-model changes are descriptive finite same-state CPU interventions. Grouped dynamics parameters not separately identified; no online measurement claim or error bound.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':run()
