"""Validate the integrated profile through ordinary40-substep NativeEnv.step."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'twentyfourth_round_operating_regions_20261002'


def run(mode):
    output=OUT/mode;output.mkdir()
    prior=json.loads((SOURCE/mode/'registered_cases.json').read_text())
    scenes=[HeightTerrainScenario(**s) for s in prior['scenarios']]
    (output/'registered_cases.json').write_text(json.dumps(prior,indent=2)+'\n')
    env=NativeEnv.height115_candidate(n=len(scenes),scenario=scenes,residual_scale=0,shared_reference=True)
    try:
        env.reset();rows=[None]*len(scenes)
        deadline=int(np.ceil((env.param.numpy()[:,3].max()+2.)/.02))+2
        for _ in range(deadline):
            _,_,done,infos=env.step(np.zeros((len(scenes),3),np.float32))
            for w in np.flatnonzero(done):
                if rows[w] is None:rows[w]={k:v for k,v in infos[w].items() if k!='terminal_observation'}
            if all(r is not None for r in rows):break
        assert all(r is not None for r in rows)
        assert all(r['physical_steps']==r['physical_evidence_steps'] and r['shared_reference_contract']=='public-region-v2-state-phase-vmc' for r in rows)
        result=dict(episodes=rows,total=len(rows),physical=sum(r['physical_safety_passed'] for r in rows),
            design=sum(r['design_joint_passed'] for r in rows),success=sum(r['success'] for r in rows),
            baseline_version=env.baseline_version,host_request_injection=False,learning=False,full_admission=False,
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/shared_reference.py',ROOT/'wheelleg_warp/native/shared_reference.npz')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('RESULT native',mode,result['physical'],result['design'],result['success'],'/',len(rows),flush=True)
        print('FAILED',[(w,r['reason'],r['velocity_rmse'],r['peak_deg'],r['min_active_design_margin_rad']) for w,r in enumerate(rows) if not r['success']],flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('speed_boundary_v2','broad_v2','registered_v2','random_v2','nearcut_v2'));run(p.parse_args().mode)
