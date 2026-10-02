"""Frozen failure: compare identical model histories with13 versus1 arm, no new plant step."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from forecaster import Forecaster
from native.environment import NativeEnv
from probe_height_115_margin import cases
OUT=Path(__file__).resolve().parent


def check():
    r=json.loads((OUT/'verification.json').read_text());s=np.load(OUT/'current_pre.npz');history=np.load(OUT/'actual.npz')['trace']
    request=np.array(r['decisions'][-1]['request']);env=NativeEnv.height115_candidate(n=2,scenario=cases()[4:6],residual_scale=0,nominal_correction=True)
    try:
        nq,nv=env.cpu.nq,env.cpu.nv
        np.testing.assert_array_equal(history[-11,:,:nq],s['q'])
        past=history[-12];past_q=past[:,:nq];past_v=past[:,nq:nq+nv]
        env.data.qpos.assign(s['q']);env.data.qvel.assign(s['v']);env.data.sensordata.assign(s['sensor']);env.k['state'].assign(s['memory'])
        env.state.assign(s['task']);env.stopped_q.assign(past_q);env.stopped_v.assign(past_v);env.data.ctrl.assign(s['previous_command'])
        predictions=[]
        for arms in (13,1):
            p=Forecaster(env,candidate_count=arms,nominal_boundary=True)
            predictions.append(p.forecast_nominal(env,np.repeat(request,arms,axis=0))[:,::arms].copy())
        # Read the already recorded actual future only after both predictions are fixed.
        truth=np.load(OUT/'failed_pair.npz')['actual'];rows=[]
        for arms,pred in zip((13,1),predictions):
            qe=float(abs(pred[:,:,:nq]-truth[:,:,:nq]).max());ve=float(abs(pred[:,:,nq:nq+nv]-truth[:,:,nq:nq+nv]).max());ce=float(abs(pred[:,:,-12:-6]-truth[:,:,-12:-6]).max())
            rows.append(dict(arms=arms,qpos=qe,qvel=ve,command=ce,pass_original=bool(qe<=2e-6 and ve<=1e-3 and ce<=1e-5)))
        np.savez_compressed(OUT/'batch_predictions.npz',arms13=predictions[0],arms1=predictions[1])
        result=dict(role='fixed_failure_predictor_batching_diagnostic',rows=rows,actual_future_not_used_for_prediction=True,new_plant_rollout=False,original_gates_unchanged=True)
        (OUT/'batch_check.json').write_text(json.dumps(result,indent=2)+'\n');print('BATCH CHECK',rows,flush=True)
    finally:env.close()


if __name__=='__main__':check()
