"""Frozen inputs: reused, reused after public reset_data, and fresh predictors."""
from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import mujoco_warp as mjw
from native.environment import NativeEnv
from probe_height_115_margin import cases
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'thirteenth_round_feedback_checked_20261002'
spec=importlib.util.spec_from_file_location('archived_predictor',SOURCE/'forecaster.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)


def run():
    assert not (OUT/'reset_check.json').exists()
    r=json.loads((SOURCE/'verification.json').read_text());s=np.load(SOURCE/'current_pre.npz');history=np.load(SOURCE/'actual.npz')['trace']
    request=np.array(r['decisions'][-1]['request']);env=NativeEnv.height115_candidate(n=2,scenario=cases()[4:6],residual_scale=0,nominal_correction=True)
    try:
        nq,nv=env.cpu.nq,env.cpu.nv
        np.testing.assert_array_equal(history[-11,:,:nq],s['q']);past=history[-12]
        env.data.qpos.assign(s['q']);env.data.qvel.assign(s['v']);env.data.sensordata.assign(s['sensor']);env.k['state'].assign(s['memory']);env.state.assign(s['task'])
        env.stopped_q.assign(past[:,:nq]);env.stopped_v.assign(past[:,nq:nq+nv]);env.data.ctrl.assign(s['previous_command'])
        p=mod.Forecaster(env,candidate_count=13,nominal_boundary=True);inputs=np.repeat(request,13,axis=0);results=[]
        fresh=p.forecast_nominal(env,inputs)[:,::13].copy()
        for repetition in range(16):
            # Fixed history stress: nearby legal inputs, no outcome-dependent choice.
            sample=inputs.copy();axis=repetition%6;sample[:,axis]=np.clip(sample[:,axis]+(.001 if repetition%2==0 else -.001),-1,1)
            p.forecast_nominal(env,sample)
            results.append((repetition,'reused',p.forecast_nominal(env,inputs)[:,::13].copy()))
            mjw.reset_data(p.model,p.data)
            results.append((repetition,'public_reset',p.forecast_nominal(env,inputs)[:,::13].copy()))
        # Only now read recorded actual future.
        truth=np.load(SOURCE/'failed_pair.npz')['actual'];rows=[]
        for rep,label,pred in [(-1,'fresh',fresh)]+results:
            qe=float(abs(pred[:,:,:nq]-truth[:,:,:nq]).max());ve=float(abs(pred[:,:,nq:nq+nv]-truth[:,:,nq:nq+nv]).max());ce=float(abs(pred[:,:,-12:-6]-truth[:,:,-12:-6]).max())
            rows.append(dict(repetition=rep,mode=label,qpos=qe,qvel=ve,command=ce,original_pass=bool(qe<=2e-6 and ve<=1e-3 and ce<=1e-5)))
        np.savez_compressed(OUT/'reset_predictions.npz',fresh=fresh,history=np.stack([x[2] for x in results]))
        result=dict(role='public_reset_frozen_predictor_state_comparison',rows=rows,history_stress_registered=True,actual_future_not_prediction_input=True,new_plant_rollout=False,learning=False,default_promoted=False,
            input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (SOURCE/'current_pre.npz',SOURCE/'actual.npz',SOURCE/'failed_pair.npz',SOURCE/'verification.json')},
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),SOURCE/'forecaster.py')})
        (OUT/'reset_check.json').write_text(json.dumps(result,indent=2)+'\n');print('RESET CHECK',[(name,sum(x['original_pass'] for x in rows if x['mode']==name),max(x['command'] for x in rows if x['mode']==name)) for name in ('fresh','reused','public_reset')],flush=True)
    finally:env.close()


if __name__=='__main__':run()
