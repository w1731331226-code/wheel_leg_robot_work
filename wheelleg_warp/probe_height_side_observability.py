"""成对左/右障碍在首次接触前的32维Actor观测可区分性。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import numpy as np
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario,bank_height_v3


def run(output,same_side=False):
    heights=(.16,.20,.25,.30,.35,.38);speeds=(.5,-.5,1.,-1.)
    cases=[]
    for speed in speeds:
        for h in heights:
            cases.extend((HeightTerrainScenario(speed=speed,height_l=.02,stand_height_m=h),
                          HeightTerrainScenario(speed=speed,height_l=.02 if same_side else 0.,
                                                height_r=0. if same_side else .02,stand_height_m=h)))
    n=len(cases);env=NativeEnv(n=n,scenario=cases,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=0)
    initial=env.reset();first=[None]*n;differences=[];zero=np.zeros((n,3),np.float32)
    assert np.array_equal(initial[::2],initial[1::2])
    for step in range(700):
        env.step_async(zero)
        obs=env.obs.numpy();state=env.state.numpy()
        row=np.full((n//2,32),np.nan,np.float32)
        for pair in range(n//2):
            left,right=2*pair,2*pair+1
            if first[left] is None and int(state[left,14])&1:first[left]=round(float(state[left,0]*.0005),3)
            right_mask=1 if same_side else 2
            if first[right] is None and int(state[right,14])&right_mask:
                first[right]=round(float(state[right,0]*.0005),3)
            if first[left] is None and first[right] is None:row[pair]=obs[left]-obs[right]
        differences.append(row)
        env.step_wait()
        if all(t is not None for t in first):break
    assert all(t is not None for t in first)
    diff=np.stack(differences);rows=[]
    for pair in range(n//2):
        clean=diff[:,pair][np.isfinite(diff[:,pair,0])]
        assert len(clean)>10
        rows.append(dict(speed=cases[2*pair].speed,height_m=cases[2*pair].stand_height_m,
                         left_contact_bin_s=first[2*pair],right_contact_bin_s=first[2*pair+1],
                         precontact_policy_steps=len(clean),max_abs_observation_difference=float(np.max(abs(clean))),
                         per_channel_max_abs=np.max(abs(clean),axis=0).tolist()))
    source=Path(__file__).resolve().parent
    names=('probe_height_side_observability.py','native/environment.py','native/controller.py','native/terrain.py','native/models.py')
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'precontact_differences.npz',differences=diff)
    (output/'verification.json').write_text(json.dumps(dict(protocol='paired same-side numerical control' if same_side else
        'paired obstacle side only; zero residual; compare 32 delayed actor inputs before either bump contact',
        same_side_control=same_side,
        source_sha256=hashes,pair_count=len(rows),max_abs_difference=max(r['max_abs_observation_difference'] for r in rows),
        rows=rows),ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(rows),max(r['max_abs_observation_difference'] for r in rows),output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--same-side',action='store_true');args=parser.parse_args()
    run(args.output,args.same_side)
