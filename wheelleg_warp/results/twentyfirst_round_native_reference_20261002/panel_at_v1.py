"""Actual NativeEnv.step coverage, with no host-side request injection."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario
from native.braking_reference import load_reference
from probe_height_115_margin import cases
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent


def run(mode):
    output=OUT/mode;output.mkdir()
    if mode=='registered':
        registry=json.loads((ROOT/'wheelleg_warp/results/fifth_round_registered_robustness_20261002/registered_cases.json').read_text())
        scenes=[HeightTerrainScenario(**s) for s in registry['scenarios']];labels=registry['labels']
    else:
        variations=[('nominal',{}),('training_corner',dict(mass=7.5,mu_l=.6,mu_r=1.,drive_difference=.03,delay_ms=10.))]
        scenes=[];labels=[]
        for label,changes in variations:
            for height in (.115,.12,.1375,.16,.25,.30,.38):
                for speed in (.5,.75,1.,-.5,-.75,-1.):
                    scenes.append(replace(cases()[4],stand_height_m=height,speed=speed,**changes));labels.append(label)
    registration=dict(mode=mode,scenarios=[asdict(s) for s in scenes],labels=labels,
        scope='Existing18-case panel or preregistered flat7-height/6-speed/two-parameter grid; no holdout, no learning, no continuous-domain proof.')
    (output/'registered_cases.json').write_text(json.dumps(registration,indent=2)+'\n')
    env=NativeEnv.height115_candidate(n=len(scenes),scenario=scenes,residual_scale=0,braking_reference=True)
    plan=load_reference();errors=[];checked=0
    try:
        env.reset();rows=[None]*len(scenes);params=env.param.numpy()
        weight=np.clip((abs(params[:,0])-.5)/.5,0,1)*np.clip((.16-params[:,13])/.045,0,1)
        for step in range(700):
            _,_,done,infos=env.step(np.zeros((len(scenes),3),np.float32))
            state=env.state.numpy();actual=env.nominal_correction.numpy();expected=np.zeros_like(actual)
            for w in range(len(scenes)):
                if done[w]:
                    if rows[w] is None:rows[w]={k:v for k,v in infos[w].items() if k!='terminal_observation'}
                    assert not actual[w].any()
                elif state[w,1]>=0:
                    start=((round(state[w,1]*2000)+9)//10)*10
                    last_update=int(state[w,0])-10
                    if last_update>=start:
                        index=min((last_update-start)//10,399);direction=int(params[w,0]<0)
                        expected[w]=weight[w]*plan[index,direction]
                checked+=1
            error=float(abs(actual-expected).max());errors.append(error);assert error<=1e-14,error
            if all(r is not None for r in rows):break
        assert all(r is not None for r in rows)
        assert all(r['physical_steps']==r['physical_evidence_steps'] and r['design_joint_contract']=='active-1p4-v1' for r in rows)
        groups={label:dict(total=labels.count(label),physical=sum(r['physical_safety_passed'] for r,l in zip(rows,labels) if l==label),
            design=sum(r['design_joint_passed'] for r,l in zip(rows,labels) if l==label),task=sum(r['success'] for r,l in zip(rows,labels) if l==label)) for label in dict.fromkeys(labels)}
        result=dict(registration=registration,episodes=rows,groups=groups,reference_samples_checked=checked,
            maximum_native_request_error_Nm=max(errors),manual_injection=False,learning=False,full_admission=False,
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/braking_reference.py',ROOT/'wheelleg_warp/native/braking_reference.npz')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('COMPLETED',mode,groups,'request_error',max(errors),flush=True)
        print('FAILURES',[(w,scenes[w].stand_height_m,scenes[w].speed,rows[w]['reason'],rows[w]['velocity_rmse'],rows[w]['stop_distance_m'],rows[w]['min_active_design_margin_rad']) for w in range(len(rows)) if not rows[w]['success']],flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('registered','grid'));run(parser.parse_args().mode)
