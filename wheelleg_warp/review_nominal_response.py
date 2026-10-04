"""Reconstruct state-only nominal hypothesis predictions and original admission."""
from pathlib import Path
import json
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
from check_nominal_response import nominal_model

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_response_v1'
DATA=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_response_v1'


def run():
    p=json.loads((OUT/'registration.json').read_text());r=json.loads((OUT/'result.json').read_text());prior=json.loads((DATA/'completion.json').read_text())
    assert r['verified_producer'] and r['cpu_steps']==54 and r['training_updates']==0 and r['registration_sha256']==sha(OUT/'registration.json')
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items()) and p['data_review_sha256']==sha(DATA/'review.json')
    assert p['fixed_design']==dict(mass_kg=7.,friction=.8,drive_difference=0.,solver_iterations=100,timestep_s=.0005)
    assert p['maximum_q_error_budget_rad']==float(np.spacing(np.float32(1.4))) and p['false_safe_allowed']==0
    reference={}
    for record in prior['records']:
        label=record['label'];path=DATA/(label+'_states.npz');assert sha(path)==record['states_sha256'];z=np.load(path,allow_pickle=False)
        for i,state in enumerate(z['state']):reference[(label,i)]=(state,z['ids'],int(z['event'][i]))
    assert len(reference)==27 and len(r['records'])==54
    seen=set()
    for row in r['records']:
        key=(row['label'],row['state_index']);state,ids,event=reference[key];h=row['hypothesis'];assert h in p['hypotheses'] and row['event']==event
        assert (key,h) not in seen;seen.add((key,h))
        # Builder signature accepts only q and declared support hypothesis, not
        # recorded actual model/scenario or solver warmstart.
        m,support=nominal_model(state[:17],h);assert support==row['support']
        assert m.opt.iterations==100 and m.opt.timestep==.0005 and m.na==0 and m.nmocap==0
        for side in ['L','R']:
            np.testing.assert_array_equal(m.geom_friction[m.geom('wheel_collide_'+side).id],[.8,.02,.001])
            assert m.actuator_gainprm[m.actuator('motor_wheel'+side).id,0]==1
        data=mujoco.MjData(m);data.qpos[:]=state[:17];data.qvel[:]=state[17:33];data.ctrl[:]=state[50:56];data.time=0;data.qacc_warmstart[:]=0
        mujoco.mj_step(m,data);predicted=data.qpos[ids[:4]];actual=state[68:85][ids[:4]]
        np.testing.assert_allclose(predicted,row['predicted_active_q'],rtol=0,atol=1e-12);np.testing.assert_array_equal(actual,row['observed_gpu_active_q'])
        np.testing.assert_allclose(predicted-actual,row['q_error_rad'],rtol=0,atol=1e-12)
        assert np.isclose(abs(predicted-actual).max(),row['max_q_error_rad'],rtol=0,atol=1e-12)
        predmargin=1.4-abs(predicted).max();actualmargin=1.4-abs(actual).max()
        assert np.isclose(predmargin,row['predicted_margin_rad'],rtol=0,atol=1e-12) and np.isclose(actualmargin,row['observed_margin_rad'],rtol=0,atol=1e-12)
        assert row['false_safe']==bool(predmargin>=0 and actualmargin<0)
    summary={}
    for h in p['hypotheses']:
        rows=[x for x in r['records'] if x['hypothesis']==h];maximum=max(x['max_q_error_rad'] for x in rows);false=sum(x['false_safe'] for x in rows)
        summary[h]=dict(states=27,max_q_error_rad=maximum,false_safe=false,admitted=maximum<=p['maximum_q_error_budget_rad'] and false==0)
    assert summary==r['summary']
    review=dict(verified=True,nominal_cpu_steps_reconstructed=54,summary=summary,result_sha256=sha(OUT/'result.json'),registration_sha256=sha(OUT/'registration.json'),reviewer_sha256=sha(__file__),
        scope='Every state-only prediction reconstructed with fixed design/friction/gain/zero warmstart and no actual-scene argument; source and actual-state identity checked. Ideal fullstate measurement assumption remains stronger than actor39.',
        disposition='Both hypotheses fail; no deployment, new PPO or empirical-error-as-robust-bound claim.')
    (OUT/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n');print(json.dumps(review))


if __name__=='__main__':run()
