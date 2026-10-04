"""Current contact normal/gap measurement hypothesis; no future scene model."""
from pathlib import Path
import json,sys
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
from native.terrain import HeightTerrainScenario,model
from native.models import build_spec,compile_spec

DATA=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_response_v1'
OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/contact_response_v1'


def extract_current(scene,q):
    # Oracle used solely to emulate the declared ideal current tactile signal.
    m=model(scene);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_fwdPosition(m,d)
    wheels={m.geom('wheel_collide_'+s).id:s for s in ['L','R']};records=[]
    for i in range(d.ncon):
        c=d.contact[i];a,b=int(c.geom[0]),int(c.geom[1])
        if a in wheels and m.geom_bodyid[b]==0:wheel=a;normal=-c.frame[:3]
        elif b in wheels and m.geom_bodyid[a]==0:wheel=b;normal=c.frame[:3]
        else:continue
        records.append(dict(side=wheels[wheel],normal_toward_wheel=np.asarray(normal).tolist(),signed_gap_m=float(c.dist)))
    return records


def build_contact_model(q,contacts):
    # Constructor has no actual scenario/plant arguments. All parameters fixed.
    scene=HeightTerrainScenario(speed=.7,mass=7.,height_l=0.,height_r=0.,center=1.8,offset=0.,mu_l=.8,mu_r=.8,drive_difference=0.,delay_ms=0.,terrain='legacy',stand_height_m=.3,solver_iterations=100)
    spec=build_spec(scene);baseline=compile_spec(spec,scene);data=mujoco.MjData(baseline);data.qpos[:]=q;mujoco.mj_kinematics(baseline,data)
    for name in ['floor','bump_L','bump_R']:
        geom=spec.geom(name);geom.contype=0;geom.conaffinity=0
    for i,c in enumerate(contacts):
        side=c['side'];wheel=baseline.geom('wheel_collide_'+side).id
        normal=np.asarray(c['normal_toward_wheel'],float);assert abs(np.linalg.norm(normal)-1)<1e-8
        center=data.geom_xpos[wheel];rotation=data.geom_xmat[wheel].reshape(3,3);radii=baseline.geom_size[wheel]
        support=np.linalg.norm(radii*(rotation.T@normal))
        offset=float(normal@center-support-c['signed_gap_m']);name=f'measured_support_{i}'
        spec.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_PLANE,pos=(normal*offset).tolist(),zaxis=normal.tolist(),size=[1.,1.,.01],contype=0,conaffinity=0)
        spec.add_pair(name=f'current_pair_{i}',geomname1=name,geomname2='wheel_collide_'+side,
            condim=int(baseline.geom_condim[wheel]),friction=[.8,.8,.02,.001,.001],solref=baseline.geom_solref[wheel].tolist(),solimp=baseline.geom_solimp[wheel].tolist(),margin=0.,gap=0.)
    return compile_spec(spec,scene)


def run():
    p=json.loads((DATA/'registration.json').read_text());review=json.loads((DATA/'review.json').read_text());ledger=json.loads((DATA/'completion.json').read_text());assert review['verified']
    OUT.mkdir(parents=True,exist_ok=False);budget=float(np.spacing(np.float32(1.4)))
    reg=dict(version='current-contact-response-v1',states=27,cpu_steps=27,training_updates=0,
        measurement='Ideal current wheel/world-support contact side, outward normal and signed gap only. Emulated using actual scene collision geometry offline; no future shape/terrain map passed to predictor.',
        inputs='Same ideal current full q17/v16/command as prior contract, plus currentcontact descriptor. This is an additional tactile/state-estimation assumption, absent from old actor39; any future control comparison must give it equally to every method.',
        fixed_design=dict(mass_kg=7.,friction=.8,drive_difference=0.,warmstart=0.,solver_iterations=100,timestep_s=.0005),
        construction='Infinite plane for each current normal/gap, collision explicitly paired only with its wheel; other floor/bump collision disabled. No descriptor means no wheel-support assumption.',
        maximum_q_error_budget_rad=budget,false_safe_allowed=0,scope='Existing diagnostic states, no independent generalization or deployment admission without further held-out evidence.',
        source_sha256={'wheelleg_warp/check_contact_response.py':sha(__file__),'wheelleg_warp/native/models.py':sha(ROOT/'wheelleg_warp/native/models.py')},response_review_sha256=sha(DATA/'review.json'),old_gate_or_final_used=False)
    (OUT/'registration.json').write_text(json.dumps(reg,indent=2)+'\n');records=[]
    for rec in ledger['records']:
        label=rec['label'];path=DATA/(label+'_states.npz');assert sha(path)==rec['states_sha256'];z=np.load(path,allow_pickle=False);ids=z['ids']
        for i,state in enumerate(z['state']):
            scene=HeightTerrainScenario(**p['cases'][int(z['world'][i])]['scenario']);contacts=extract_current(scene,state[:17])
            m=build_contact_model(state[:17],contacts);d=mujoco.MjData(m);d.qpos[:]=state[:17];d.qvel[:]=state[17:33];d.ctrl[:]=state[50:56];d.qacc_warmstart[:]=0.;d.time=0.
            mujoco.mj_step(m,d);pred=d.qpos[ids[:4]];actual=state[68:85][ids[:4]];error=pred-actual
            records.append(dict(label=label,index=i,event=int(z['event'][i]),contacts=contacts,predicted_q=pred.tolist(),observed_gpu_q=actual.tolist(),error_rad=error.tolist(),
                max_q_error_rad=float(abs(error).max()),false_safe=bool(abs(pred).max()<=1.4 and abs(actual).max()>1.4)))
        print('COMPLETED contact prediction',label,len(z['state']),flush=True)
    assert len(records)==27
    maximum=max(r['max_q_error_rad'] for r in records);false=sum(r['false_safe'] for r in records)
    result=dict(verified_producer=True,cpu_steps=27,records=records,max_q_error_rad=maximum,false_safe=false,development_gate=maximum<=budget and false==0,
        registration_sha256=sha(OUT/'registration.json'),limits='Current ideal descriptor is not ordinary encoder-only observability or validated real tactile measurement. Plane approximation omits future contacts and curvature; fixed parameters remain uncertain. No safe-state/control guarantee.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':run()
