"""同一训练范围地形配对：连续腿长目标与固定0.30 m名义控制。"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import numpy as np
from native.environment import NativeEnv
from native.terrain import bank_height_v3,sample_height_terrain_v3


def evaluate(scenarios):
    n=len(scenarios);env=NativeEnv(n=n,scenario=scenarios,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=0)
    env.reset();rows=[None]*n;zero=np.zeros((n,3),np.float32)
    for _ in range(700):
        _,_,done,infos=env.step(zero)
        for i in np.flatnonzero(done):
            if rows[i] is None:
                info=infos[i]
                rows[i]=dict(success=info['success'],reason=info['reason'],peak_deg=info['peak_deg'],
                    height_rmse_m=info['height_rmse_m'],velocity_rmse=info['velocity_rmse'])
        if all(row is not None for row in rows):break
    assert all(row is not None for row in rows)
    return rows


def run(output,seed):
    assert seed in (1135000,1136000)
    cases=[sample_height_terrain_v3(seed+i,stage=3,split='train') for i in range(128)]
    variable=evaluate(cases);fixed=evaluate([replace(s,stand_height_m=.3) for s in cases])
    rows=[dict(seed=seed+i,scenario=asdict(s),variable=a,fixed_0_30=b) for i,(s,a,b) in enumerate(zip(cases,variable,fixed))]
    root=Path(__file__).resolve().parents[1]
    names=('wheelleg_warp/probe_height_iid_pair.py','wheelleg_warp/native/environment.py','wheelleg_warp/native/controller.py',
           'wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py','wheelleg_ppo/tools/ppo_env.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    (output/'verification.json').write_text(json.dumps(dict(protocol='public train-seed height pairing, zero residual, no PPO',
        seed_start=seed,n=128,source_sha256=hashes,variable_success=sum(x['success'] for x in variable),
        fixed_0_30_success=sum(x['success'] for x in fixed),rows=rows),ensure_ascii=False,indent=2)+'\n')
    print('PASS',seed,'variable',sum(x['success'] for x in variable),'fixed',sum(x['success'] for x in fixed),output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(1135000,1136000),required=True);args=parser.parse_args()
    run(args.output,args.seed)
