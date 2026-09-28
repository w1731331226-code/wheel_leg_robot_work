"""20 mm多高度边界：原M3轮差矩的接触前固定时窗特权上界。"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import numpy as np
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario,bank_height_v3


def rollout(cases,contact_times=None,polarity=0):
    n=len(cases);env=NativeEnv(n=n,scenario=cases,bank_factory=bank_height_v3,height_conditioned=True,
                               residual_scale=1 if polarity else 0)
    env.reset();times=[None]*n;results=[None]*n;filtered_at_contact=[None]*n
    directions=np.sign([s.speed for s in cases]).astype(np.float32)
    for step in range(700):
        time=step*.02;actions=np.zeros((n,3),np.float32)
        if contact_times is not None:
            mask=np.asarray([(t-.1<=time<t+.05) for t in contact_times])
            actions[:,2]=mask*directions*polarity
        env.step_async(actions)
        state=env.state.numpy();filtered=env.k['state'].numpy()
        for i,s in enumerate(cases):
            required=1 if s.height_l else 2
            if times[i] is None and int(state[i,14])&required:
                times[i]=round(float(state[i,0]*.0005),3)
            if contact_times is not None and filtered_at_contact[i] is None and time+.02>=contact_times[i]:
                filtered_at_contact[i]=float(filtered[i,18])
        _,_,done,infos=env.step_wait()
        for i in np.flatnonzero(done):
            if results[i] is None:
                info=infos[i]
                results[i]=dict(success=info['success'],reason=info['reason'],peak_deg=info['peak_deg'],
                    height_rmse_m=info['height_rmse_m'],velocity_rmse=info['velocity_rmse'],
                    terrain_evidence_passed=info['terrain_evidence_passed'])
        if all(row is not None for row in results):break
    assert all(row is not None for row in results)
    if contact_times is None:assert all(t is not None for t in times)
    if polarity:assert all(value is None or abs(value)>=.99 for value in filtered_at_contact),filtered_at_contact
    return [dict(scenario=asdict(s),contact_bin_s=t,result=row,filtered_at_reference_contact=filtered)
            for s,t,row,filtered in zip(cases,times,results,filtered_at_contact)]


def run(output):
    panel=Path(__file__).resolve().parents[1]/'wheelleg_warp/results/height_scope_recheck_20260928/boundary_run1/verification.json'
    cases=[HeightTerrainScenario(**row['scenario']) for row in json.loads(panel.read_text())['rows']]
    assert len(cases)==48 and all(max(s.height_l,s.height_r)==.02 for s in cases)
    baseline=rollout(cases);reference=[row['contact_bin_s'] for row in baseline]
    positive=rollout(cases,reference,1);negative=rollout(cases,reference,-1)
    root=Path(__file__).resolve().parents[1]
    names=('wheelleg_warp/probe_height_precontact.py','wheelleg_warp/native/environment.py',
           'wheelleg_warp/native/controller.py','wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    groups=dict(baseline=baseline,positive=positive,negative=negative)
    summary={name:dict(success=sum(row['result']['success'] for row in rows),total=len(rows),
                       by_height={str(h):sum(row['result']['success'] for row in rows if row['scenario']['stand_height_m']==h)
                                  for h in (.16,.20,.25,.30,.35,.38)}) for name,rows in groups.items()}
    output.mkdir(parents=True,exist_ok=False)
    (output/'verification.json').write_text(json.dumps(dict(protocol='privileged zero-action contact schedule ±100/50ms M3 channel3',
        source_sha256=hashes,panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),summary=summary,rows=groups,
        training=False,holdout_opened=False),ensure_ascii=False,indent=2)+'\n')
    print('PASS',summary,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.output)
