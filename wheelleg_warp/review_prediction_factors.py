"""Independent reconstruction of all prescribed offline factor combinations."""
from pathlib import Path
from itertools import product
import json
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
from probe_prediction_factors import construct
from native.terrain import HeightTerrainScenario

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/prediction_factors_v1'
DATA=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_response_v1'


def run():
    p=json.loads((OUT/'registration.json').read_text());r=json.loads((OUT/'result.json').read_text());cases=json.loads((DATA/'registration.json').read_text())['cases'];ledger=json.loads((DATA/'completion.json').read_text())
    assert r['verified_producer'] and r['cpu_steps']==p['cpu_steps']==216 and p['training_updates']==0 and not p['old_gate_or_final_used']
    assert p['source_sha256']==sha(ROOT/'wheelleg_warp/probe_prediction_factors.py') and r['registration_sha256']==sha(OUT/'registration.json')
    assert p['response_review_sha256']==sha(DATA/'review.json') and p['nominal_review_sha256']==sha(ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_response_v1/review.json')
    states={}
    for rec in ledger['records']:
        path=DATA/(rec['label']+'_states.npz');assert sha(path)==rec['states_sha256'];z=np.load(path,allow_pickle=False)
        for i,state in enumerate(z['state']):states[(rec['label'],i)]=(state,int(z['world'][i]),int(z['event'][i]),z['ids'])
    assert len(states)==27 and len(r['records'])==216;seen=set()
    for row in r['records']:
        key=(row['label'],row['index']);state,world,event,ids=states[key];flags=tuple(row['factors']);assert flags in list(product([False,True],repeat=3))
        assert (key,flags) not in seen;seen.add((key,flags));assert row['event']==event and row['world']==world
        scene=HeightTerrainScenario(**cases[world]['scenario']);m=construct(scene,state[:17],flags[0],flags[1]);data=mujoco.MjData(m)
        assert m.na==0 and m.opt.iterations==100 and m.opt.timestep==.0005
        data.qpos[:]=state[:17];data.qvel[:]=state[17:33];data.ctrl[:]=state[50:56];data.qacc_warmstart[:]=state[33:49] if flags[2] else 0.;data.time=state[49]
        mujoco.mj_step(m,data);pred=data.qpos[ids[:4]];actual=state[68:85][ids[:4]]
        np.testing.assert_allclose(pred,row['predicted_q'],rtol=0,atol=1e-12);np.testing.assert_array_equal(actual,row['actual_gpu_q'])
        np.testing.assert_allclose(pred-actual,row['error_rad'],rtol=0,atol=1e-12)
        assert np.isclose(abs(pred-actual).max(),row['max_error_rad'],rtol=0,atol=1e-12)
        assert row['false_safe']==bool(abs(pred).max()<=1.4 and abs(actual).max()>1.4)
    summary={}
    for flags in product([False,True],repeat=3):
        rows=[x for x in r['records'] if x['factors']==list(flags)];key='/'.join('1' if f else '0' for f in flags)
        summary[key]=dict(states=27,max_error_rad=max(x['max_error_rad'] for x in rows),mean_max_error_rad=float(np.mean([x['max_error_rad'] for x in rows])),false_safe=sum(x['false_safe'] for x in rows))
    assert summary==r['summary']
    prior=json.loads((ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_response_v1/result.json').read_text())
    assert summary['0/0/0']['max_error_rad']==prior['summary']['lower_flat_plane']['max_q_error_rad'] and summary['0/0/0']['false_safe']==3
    response=json.loads((DATA/'review.json').read_text());maximum=max(s['maximum_cpu_gpu_q_error_rad'] for s in response['summary'].values())
    assert summary['1/1/1']['max_error_rad']==maximum
    warm_changes=[a for a in r['records'] if a['factors'][2]]
    for a in warm_changes:
        b=next(x for x in r['records'] if x['label']==a['label'] and x['index']==a['index'] and x['factors']==[a['factors'][0],a['factors'][1],False])
        np.testing.assert_array_equal(a['predicted_q'],b['predicted_q'])
    review=dict(verified=True,cpu_steps_reconstructed=216,states=27,summary=summary,
        registration_sha256=sha(OUT/'registration.json'),result_sha256=sha(OUT/'result.json'),reviewer_sha256=sha(__file__),
        conclusions='In these finite states, replacing support geometry removes most error and false-safe classifications; true dynamics parameters remove additional error. Warmstart levels give identical results here. No real-geometry factor is promoted as an allowed predictor.',
        limits='Actual geometry includes complete simulated scene; this factor does not prove current-contact-only measurement suffices. Mass/friction/drive grouped, not individually attributed. Finite oracle intervention is not an online uncertainty bound or globally independent experiment.')
    (OUT/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n');print(json.dumps(review))


if __name__=='__main__':run()
