"""Declared parameter-domain corner envelope and prespecified interior checks.

Corner extrema are finite diagnostics, not a continuous uncertainty proof.
"""
from pathlib import Path
from itertools import product
import json,sys
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
from native.terrain import HeightTerrainScenario,model
from native.models import build_spec,compile_spec

DATA=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_response_v1'
CONTACT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/contact_response_v1'
OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/contact_uncertainty_v1'


def construct(q,contacts,parameters):
    mass,mul,mur,drive=parameters
    s=HeightTerrainScenario(speed=.7,mass=float(mass),height_l=0.,height_r=0.,center=1.8,offset=0.,
        mu_l=float(mul),mu_r=float(mur),drive_difference=float(drive),delay_ms=0.,terrain='legacy',stand_height_m=.3,solver_iterations=100)
    spec=build_spec(s);initial=compile_spec(spec,s);d=mujoco.MjData(initial);d.qpos[:]=q;mujoco.mj_kinematics(initial,d)
    for name in ['floor','bump_L','bump_R']:spec.geom(name).contype=0;spec.geom(name).conaffinity=0
    for i,c in enumerate(contacts):
        side=c['side'];g=initial.geom('wheel_collide_'+side).id;n=np.asarray(c['normal_toward_wheel'])
        support=np.linalg.norm(initial.geom_size[g]*(d.geom_xmat[g].reshape(3,3).T@n));offset=float(n@d.geom_xpos[g]-support-c['signed_gap_m'])
        name=f'current_support_{i}';spec.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_PLANE,pos=(n*offset).tolist(),zaxis=n.tolist(),size=[1,1,.01],contype=0,conaffinity=0)
        mu=mul if side=='L' else mur
        spec.add_pair(name=f'current_pair_{i}',geomname1=name,geomname2='wheel_collide_'+side,condim=int(initial.geom_condim[g]),
            friction=[mu,mu,.02,.001,.001],solref=initial.geom_solref[g].tolist(),solimp=initial.geom_solimp[g].tolist(),margin=0.,gap=0.)
    return compile_spec(spec,s)


def step(m,state,ids):
    d=mujoco.MjData(m);d.qpos[:]=state[:17];d.qvel[:]=state[17:33];d.ctrl[:]=state[50:56];d.qacc_warmstart[:]=0;d.time=0
    mujoco.mj_step(m,d);return d.qpos[ids[:4]].copy()


def run():
    cr=json.loads((CONTACT/'review.json').read_text());source=json.loads((CONTACT/'result.json').read_text());states_reg=json.loads((DATA/'registration.json').read_text());ledger=json.loads((DATA/'completion.json').read_text());assert cr['verified']
    OUT.mkdir(parents=True,exist_ok=False)
    bounds=np.array([[7.,7.5],[.6,1.],[.6,1.],[-.03,.03]])
    corners=[list(p) for p in product(*bounds.tolist())]
    rng=np.random.default_rng(129001);interiors=(bounds[:,0]+rng.uniform(.05,.95,(8,4))*(bounds[:,1]-bounds[:,0])).tolist()
    reserve=float(np.spacing(np.float32(1.4)))
    reg=dict(version='contact-parameter-envelope-v1',states=27,corners=corners,interiors=interiors,interior_seed=129001,parameter_order=['mass','mu_left','mu_right','drive_difference'],
        bounds=bounds.tolist(),numerical_reserve_rad=reserve,cpu_model_steps=27*(16+8),cpu_oracle_validation_steps=27*8,training_updates=0,
        inputs='Same declared ideal fullstate/current normal-gap/known command. Prediction consumes all public corners, never the actual parameter value; interior parameter labels supplied only to offline verification models.',
        corner_rule='Take component-wise min/max of16 CPU corner next-q values; reserve one positionULP for separate GPU arithmetic comparison. No enlargement based on test scores.',
        validation='8 fixed interior parameter vectors before any score, same27 states, compare current-contact model to actual geometry oracle at same interior parameters, plus observedGPU next state. Every miss retained.',
        limits='Existing states are not independent state holdout. Corner coverage of these interior points is empirical, not a continuum/recursive robustness certificate.',
        contact_review_sha256=sha(CONTACT/'review.json'),source_sha256=sha(__file__),old_gate_or_final_used=False)
    (OUT/'registration.json').write_text(json.dumps(reg,indent=2)+'\n');records=[];prediction_steps=0;oracle_steps=0
    for rec in ledger['records']:
        label=rec['label'];path=DATA/(label+'_states.npz');assert sha(path)==rec['states_sha256'];z=np.load(path,allow_pickle=False);ids=z['ids']
        for i,state in enumerate(z['state']):
            contacts=next(r['contacts'] for r in source['records'] if r['label']==label and r['index']==i)
            predictions=[step(construct(state[:17],contacts,p),state,ids) for p in corners];prediction_steps+=16
            values=np.asarray(predictions);lower=values.min(axis=0);upper=values.max(axis=0);checks=[]
            world=int(z['world'][i]);scene=HeightTerrainScenario(**states_reg['cases'][world]['scenario'])
            from dataclasses import replace
            for j,parameters in enumerate(interiors):
                predicted=step(construct(state[:17],contacts,parameters),state,ids);prediction_steps+=1
                actual_scene=replace(scene,mass=parameters[0],mu_l=parameters[1],mu_r=parameters[2],drive_difference=parameters[3])
                actual=step(model(actual_scene),state,ids);oracle_steps+=1
                checks.append(dict(index=j,parameter_vector=parameters,predicted_q=predicted.tolist(),oracle_q=actual.tolist(),
                    contact_oracle_difference_rad=float(abs(predicted-actual).max()),model_inside_raw_corner=bool(np.all(predicted>=lower-1e-12)&np.all(predicted<=upper+1e-12)),
                    oracle_inside_reserved_corner=bool(np.all(actual>=lower-reserve)&np.all(actual<=upper+reserve))))
            gpu=state[68:85][ids[:4]];records.append(dict(label=label,index=i,event=int(z['event'][i]),corner_q=values.tolist(),lower_q=lower.tolist(),upper_q=upper.tolist(),interior_checks=checks,
                observed_gpu_q=gpu.tolist(),gpu_inside_reserved_corner=bool(np.all(gpu>=lower-reserve)&np.all(gpu<=upper+reserve))))
        print('COMPLETED envelope',label,len(z['state']),flush=True)
    assert len(records)==27 and prediction_steps==648 and oracle_steps==216
    checks=[c for r in records for c in r['interior_checks']]
    summary=dict(interior_checks=len(checks),model_corner_misses=sum(not c['model_inside_raw_corner'] for c in checks),oracle_reserved_corner_misses=sum(not c['oracle_inside_reserved_corner'] for c in checks),
        observed_gpu_corner_misses=sum(not r['gpu_inside_reserved_corner'] for r in records),max_contact_oracle_difference_rad=max(c['contact_oracle_difference_rad'] for c in checks),
        max_corner_width_rad=max(max(np.asarray(r['upper_q'])-np.asarray(r['lower_q'])) for r in records))
    (OUT/'result.json').write_text(json.dumps(dict(verified_producer=True,cpu_prediction_steps=prediction_steps,cpu_oracle_validation_steps=oracle_steps,records=records,summary=summary,registration_sha256=sha(OUT/'registration.json'),
        limits='Even zero finite misses does not establish a continuous parameter-domain bound or an independent state validation. No model deployment, layer safety or newPPO is admitted by this report.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':run()
