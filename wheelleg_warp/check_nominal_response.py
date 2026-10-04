"""Fixed-design state-only support hypotheses, not a terrain-aware oracle."""
from pathlib import Path
import json,sys
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
from native.models import build_spec,compile_spec
from native.terrain import HeightTerrainScenario

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_response_v1'
DATA=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_response_v1'


def nominal_model(q,hypothesis):
    # No recorded case/scenario argument: nominal physical parameters are fixed.
    scene=HeightTerrainScenario(speed=.7,mass=7.,height_l=0.,height_r=0.,center=1.8,offset=0.,
        mu_l=.8,mu_r=.8,drive_difference=0.,delay_ms=0.,terrain='legacy',stand_height_m=.3,solver_iterations=100)
    spec=build_spec(scene);initial=compile_spec(spec,scene);d=mujoco.MjData(initial);d.qpos[:]=q;mujoco.mj_kinematics(initial,d)
    lower=[];positions=[]
    for side in ['L','R']:
        g=initial.geom('wheel_collide_'+side).id;position=d.geom_xpos[g].copy();axis=d.geom_xmat[g].reshape(3,3)[:,1]
        size=initial.geom_size[g];support=np.sqrt(size[0]**2+(size[1]**2-size[0]**2)*axis[2]**2)
        lower.append(float(position[2]-support));positions.append(position)
    floor=min(lower);spec.geom('floor').pos[2]=floor
    if hypothesis=='two_current_supports':
        # Reuse empty bump geometry as horizontal current supports only.
        for i,side in enumerate(['L','R']):
            geom=spec.geom('bump_'+side);geom.pos=[positions[i][0],positions[i][1],lower[i]-.005]
            geom.size=[.25,.075,.005]
    elif hypothesis!='lower_flat_plane':raise ValueError(hypothesis)
    return compile_spec(spec,scene),dict(wheel_bottom_estimates=lower,common_floor_height=floor)


def run():
    original=json.loads((DATA/'registration.json').read_text());review=json.loads((DATA/'review.json').read_text());completed=json.loads((DATA/'completion.json').read_text())
    assert review['verified']
    OUT.mkdir(parents=True,exist_ok=False)
    budget=float(np.spacing(np.float32(1.4)))
    reg=dict(version='nominal-support-response-v1',hypotheses=['lower_flat_plane','two_current_supports'],snapshots=27,cpu_steps=54,training_updates=0,
        inputs='Ideal current full17 generalized positions and16 velocities, command6; includes passive hinges and ideal base pose/velocity estimate beyond old actor39. Any future control comparison must grant the same declared state-estimator contract to every method.',
        excluded_inputs='No scene/mass/friction/drive/terrain parameters, no obstacle preview, no actual qacc_warmstart; nominal model warmstart0.',
        fixed_design=dict(mass_kg=7.,friction=.8,drive_difference=0.,solver_iterations=100,timestep_s=.0005),
        supports='Either common horizontal floor at minimum FK wheel-bottom z, or two horizontal pads at each FK wheel-bottom z; these are support assumptions, not measurements of actual terrain/contact.',
        maximum_q_error_budget_rad=budget,false_safe_allowed=0,
        admission='Each hypothesis separately: all27 errors<=one float32 position ULP at1.4, zero falsely predicted safe cases; only then consider independent held-out error testing. No parameter search or best-hypothesis promotion.',
        data_review_sha256=sha(DATA/'review.json'),source_sha256={'wheelleg_warp/check_nominal_response.py':sha(__file__),
            'wheelleg_warp/native/models.py':sha(ROOT/'wheelleg_warp/native/models.py')},old_gate_or_final_used=False)
    (OUT/'registration.json').write_text(json.dumps(reg,indent=2)+'\n')
    results=[]
    for record in completed['records']:
        label=record['label'];path=DATA/(label+'_states.npz');assert sha(path)==record['states_sha256']
        z=np.load(path,allow_pickle=False);ids=z['ids']
        for i,s in enumerate(z['state']):
            for hypothesis in reg['hypotheses']:
                m,support=nominal_model(s[:17],hypothesis);d=mujoco.MjData(m)
                d.qpos[:]=s[:17];d.qvel[:]=s[17:33];d.ctrl[:]=s[50:56];d.time=0.;d.qacc_warmstart[:]=0
                mujoco.mj_step(m,d)
                predicted=d.qpos[ids[:4]];actual=s[68:85][ids[:4]];qerror=predicted-actual
                predmargin=float(1.4-abs(predicted).max());actualmargin=float(1.4-abs(actual).max())
                results.append(dict(label=label,state_index=i,event=int(z['event'][i]),hypothesis=hypothesis,support=support,
                    predicted_active_q=predicted.tolist(),observed_gpu_active_q=actual.tolist(),q_error_rad=qerror.tolist(),
                    max_q_error_rad=float(abs(qerror).max()),predicted_margin_rad=predmargin,observed_margin_rad=actualmargin,
                    false_safe=bool(predmargin>=0 and actualmargin<0)))
        print('COMPLETED',label,'nominal54step budget',flush=True)
    assert len(results)==54
    summary={}
    for h in reg['hypotheses']:
        rows=[r for r in results if r['hypothesis']==h];maximum=max(r['max_q_error_rad'] for r in rows);false=sum(r['false_safe'] for r in rows)
        summary[h]=dict(states=27,max_q_error_rad=maximum,false_safe=false,admitted=maximum<=budget and false==0)
    report=dict(verified_producer=True,cpu_steps=54,training_updates=0,summary=summary,records=results,registration_sha256=sha(OUT/'registration.json'),
        limits='Finite existing diagnostic states, not independent/generalization or a certified error bound. Support reconstruction uses only stated ideal state and fixed robot geometry; it may miss contact normals/slip/impulses. Fullstate assumption not silently attributed to actor39.')
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':run()
