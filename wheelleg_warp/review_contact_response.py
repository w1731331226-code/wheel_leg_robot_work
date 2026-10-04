"""Current-contact descriptor and fixed-model prediction reconstruction."""
from pathlib import Path
import json
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
from check_contact_response import extract_current,build_contact_model
from native.terrain import HeightTerrainScenario

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/contact_response_v1'
DATA=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_response_v1'


def run():
    reg=json.loads((OUT/'registration.json').read_text());result=json.loads((OUT/'result.json').read_text());prior=json.loads((DATA/'registration.json').read_text());ledger=json.loads((DATA/'completion.json').read_text())
    assert result['verified_producer'] and result['cpu_steps']==reg['cpu_steps']==27 and reg['training_updates']==0 and not reg['old_gate_or_final_used']
    assert result['registration_sha256']==sha(OUT/'registration.json') and all(sha(ROOT/n)==v for n,v in reg['source_sha256'].items())
    assert reg['response_review_sha256']==sha(DATA/'review.json') and reg['maximum_q_error_budget_rad']==float(np.spacing(np.float32(1.4)))
    states={}
    for rec in ledger['records']:
        path=DATA/(rec['label']+'_states.npz');assert sha(path)==rec['states_sha256'];z=np.load(path,allow_pickle=False)
        for i,state in enumerate(z['state']):states[(rec['label'],i)]=(state,z['ids'],int(z['world'][i]),int(z['event'][i]))
    factor=json.loads((ROOT/'wheelleg_warp/results/paper_recovery_20261004/prediction_factors_v1/result.json').read_text());maximum_factor_difference=0.;seen=set()
    for row in result['records']:
        key=(row['label'],row['index']);assert key not in seen;seen.add(key);state,ids,world,event=states[key];assert event==row['event']
        contacts=extract_current(HeightTerrainScenario(**prior['cases'][world]['scenario']),state[:17]);assert contacts==row['contacts']
        assert all(set(c)=={'side','normal_toward_wheel','signed_gap_m'} for c in contacts)
        m=build_contact_model(state[:17],contacts);assert m.opt.iterations==100 and m.opt.timestep==.0005 and m.npair==len(contacts)
        for side in ['L','R']:
            np.testing.assert_array_equal(m.geom_friction[m.geom('wheel_collide_'+side).id],[.8,.02,.001]);assert m.actuator_gainprm[m.actuator('motor_wheel'+side).id,0]==1
        for name in ['floor','bump_L','bump_R']:
            g=m.geom(name).id;assert m.geom_contype[g]==m.geom_conaffinity[g]==0
        d=mujoco.MjData(m);d.qpos[:]=state[:17];d.qvel[:]=state[17:33];d.ctrl[:]=state[50:56];d.qacc_warmstart[:]=0;d.time=0
        mujoco.mj_step(m,d);pred=d.qpos[ids[:4]];actual=state[68:85][ids[:4]]
        np.testing.assert_allclose(pred,row['predicted_q'],rtol=0,atol=1e-12);np.testing.assert_array_equal(actual,row['observed_gpu_q'])
        np.testing.assert_allclose(pred-actual,row['error_rad'],rtol=0,atol=1e-12)
        assert np.isclose(abs(pred-actual).max(),row['max_q_error_rad'],rtol=0,atol=1e-12) and row['false_safe']==bool(abs(pred).max()<=1.4 and abs(actual).max()>1.4)
        reference=next(x for x in factor['records'] if x['label']==key[0] and x['index']==key[1] and x['factors']==[True,False,False])
        maximum_factor_difference=max(maximum_factor_difference,float(abs(pred-np.asarray(reference['predicted_q'])).max()))
    assert len(seen)==len(states)==27
    maximum=max(r['max_q_error_rad'] for r in result['records']);false=sum(r['false_safe'] for r in result['records']);gate=maximum<=reg['maximum_q_error_budget_rad'] and false==0
    assert maximum==result['max_q_error_rad'] and false==result['false_safe'] and gate==result['development_gate']
    review=dict(verified=True,cpu_predictions_reconstructed=27,max_q_error_rad=maximum,false_safe=false,development_gate=gate,
        max_difference_from_true_geometry_fixed_parameter_predictions_rad=maximum_factor_difference,
        result_sha256=sha(OUT/'result.json'),registration_sha256=sha(OUT/'registration.json'),reviewer_sha256=sha(__file__),
        interpretation='Current ideal normal/gap representation matches true-geometry fixed-parameter prediction in this finite sample within reported error; full future map not needed for this observation. Fixed dynamics uncertainty remains above precision budget.',
        limits='Descriptor itself emulated from actual collision geometry; not a validated real tactile sensor or old actor39 input. Contact planes cannot guarantee future contact changes or general support modes. No model/learning deployment admitted.')
    (OUT/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n');print(json.dumps(review))


if __name__=='__main__':run()
