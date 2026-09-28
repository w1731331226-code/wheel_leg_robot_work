"""只读对照height-v1探针检查点与零残差在20 mm边界的请求/执行链。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize
from benchmark_parallel import TimedPPO
from native.terrain import HeightTerrainScenario
from trace_failure_chain import COL,RecordedEnv


SELECTED=(0,5,6,11,12,17,18,23)


def run(output,policy):
    panel=ROOT/'wheelleg_warp/results/height_scope_recheck_20260928/boundary_run1/verification.json'
    rows=json.loads(panel.read_text())['rows'];cases=[HeightTerrainScenario(**row['scenario']) for row in rows]
    assert len(cases)==48 and all(cases[i].height_l==.02 for i in SELECTED)
    checkpoint=ROOT/'wheelleg_warp/results/height_v1_capability_pilot_20260928/policy_102400'
    torch.set_num_threads(1)
    raw=RecordedEnv(48,scenario=cases,height_conditioned=True,residual_scale=0 if policy=='zero' else 1)
    if policy=='final':
        env=VecNormalize.load(str(checkpoint)+'.pkl',raw);env.training=False;env.norm_reward=False
        model=TimedPPO.load(str(checkpoint)+'.zip',device='cpu')
    else:env=raw;model=None
    obs=env.reset();zero=np.zeros((48,3),np.float32)
    touched={i:None for i in SELECTED};windows={i:[] for i in SELECTED};results={i:None for i in SELECTED}
    try:
        for step in range(700):
            action=zero if model is None else model.predict(obs,deterministic=True)[0]
            env.step_async(action)
            states=raw.state.numpy();trace=raw.trace.numpy()
            for i in SELECTED:
                if touched[i] is None and int(states[i,14])&1:touched[i]=step
                if touched[i] is not None and 0<=step-touched[i]<7:windows[i].append(trace[:,i].copy())
            obs,_,done,infos=env.step_wait()
            for i in SELECTED:
                if done[i] and results[i] is None:
                    info=infos[i];results[i]=dict(success=info['success'],reason=info['reason'],peak_deg=info['peak_deg'],
                        height_rmse_m=info['height_rmse_m'])
            if all(value is not None for value in results.values()):break
        assert all(value is not None for value in results.values())
        assert all(len(chunks)==7 for chunks in windows.values())
    finally:env.close()
    packed=np.stack([np.concatenate(windows[i],axis=0) for i in SELECTED]);col={name:COL.index(name) for name in COL}
    summaries=[]
    for k,i in enumerate(SELECTED):
        a=packed[k];requested=np.linalg.norm(a[:,47:53],axis=1);executed=np.linalg.norm(a[:,53:59],axis=1)
        summaries.append(dict(index=i,scenario=rows[i]['scenario'],result=results[i],
            contact_bin_end_s=round((touched[i]+1)*.02,3),lambda_zero_fraction=float(np.mean(a[:,col['lambda']]<.5)),
            mean_requested_norm_Nm=float(requested.mean()),mean_executed_norm_Nm=float(executed.mean()),
            requested_nonzero_steps=int(np.count_nonzero(requested>1e-4)),
            executed_nonzero_steps=int(np.count_nonzero(executed>1e-4)),
            peak_wheel1_rpm=float(np.max(abs(a[:,col['motor_speed_4']]))*60/(2*np.pi))))
    names=('wheelleg_warp/trace_height_capability.py','wheelleg_warp/trace_failure_chain.py',
           'wheelleg_warp/native/environment.py','wheelleg_warp/native/controller.py','wheelleg_warp/native/terrain.py')
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'window.npz',trace=packed)
    (output/'summary.json').write_text(json.dumps(dict(policy=policy,source_sha256=hashes,
        panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
        checkpoint_sha256={suffix:hashlib.sha256(Path(str(checkpoint)+suffix).read_bytes()).hexdigest()
                           for suffix in ('.zip','.pkl')} if policy=='final' else None,
        columns=COL,selected=SELECTED,rows=summaries),ensure_ascii=False,indent=2)+'\n')
    print('PASS',policy,sum(x['result']['success'] for x in summaries),'/',len(summaries),output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',choices=('zero','final'),required=True);args=parser.parse_args()
    run(args.output,args.policy)
