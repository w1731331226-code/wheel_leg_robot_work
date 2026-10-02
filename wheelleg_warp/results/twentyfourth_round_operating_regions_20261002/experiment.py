"""Whole-controller selection by fixed public height, never parameter labels."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
from native.environment import NativeEnv
from native.terrain import model,HeightTerrainScenario,sample_height_terrain_115
from probe_height_115_margin import cases
from select_braking_common_action import lqr_cost
from model_lqr import vmc_coordinates
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent
PRIOR=OUT.parent/'twentythird_round_virtual_reference_20261002'
spec=importlib.util.spec_from_file_location('vmc_reference_core',PRIOR/'experiment.py')
core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)


def calibration(refined=False):
    heights=[.1181875,.11825,.118375] if refined else [.118,.118125,.11825,.1185]
    velocities=(.5,.75,.875,.9375,1.,-.5,-.75,-.875,-.9375,-1.) if refined else (.9375,1.,-.9375,-1.)
    variations=[('nominal',{}),('training_corner',dict(mass=7.5,mu_l=.6,mu_r=1.,drive_difference=.03,delay_ms=10.))]
    scenes=[replace(cases()[4],stand_height_m=h,speed=v,**change) for _,change in variations for h in heights for v in velocities]
    r=dict(heights=heights,scenarios=[asdict(s) for s in scenes],labels=[name for name,_ in variations for _ in range(len(heights)*len(velocities))],
        rule='Require at least two adjacent calibration heights where both complete controllers pass every registered speed and parameter group; choose midpoint of longest such contiguous run, earliest run breaks ties. Freeze before validation. No mixing or hidden-parameter selection.',learning=False,full_admission=False)
    path=OUT/('calibration_refined.json' if refined else 'calibration.json')
    if path.exists():assert json.loads(path.read_text())==r
    else:path.write_text(json.dumps(r,indent=2)+'\n')
    return scenes


def choose(refined=False):
    target=OUT/('selector_refined.json' if refined else 'selector.json')
    assert not target.exists()
    base=json.loads((OUT/('calibration_refined.json' if refined else 'calibration.json')).read_text())
    modes=('refined_zero','refined_virtual') if refined else ('overlap_zero','overlap_virtual')
    results=[json.loads((OUT/m/'verification.json').read_text()) for m in modes]
    good=[]
    for h in base['heights']:
        ids=[i for i,s in enumerate(base['scenarios']) if s['stand_height_m']==h]
        good.append(all(all(r['episodes'][i]['success'] for i in ids) for r in results))
    runs=[];start=None
    for i,passed in enumerate(good+[False]):
        if passed and start is None:start=i
        elif not passed and start is not None:
            if i-start>=2:runs.append((base['heights'][i-1]-base['heights'][start],start,i-1))
            start=None
    if not runs:
        result=dict(accepted=False,good_heights=[h for h,g in zip(base['heights'],good) if g],reason='No adjacent jointly passing calibration heights',full_admission=False)
    else:
        _,a,b=max(runs,key=lambda t:(t[0],-t[1]));lo,hi=base['heights'][a],base['heights'][b]
        result=dict(accepted=True,switch_height=(lo+hi)/2,calibration_overlap=[lo,hi],
            selector='virtual reference if public target height is below switch_height; otherwise zero correction',
            parameter_truth_used=False,validation_performed=False,full_admission=False)
    target.write_text(json.dumps(result,indent=2)+'\n');print('FROZEN SELECTOR',result,flush=True)


def run(mode):
    output=OUT/mode;output.mkdir()
    if mode.startswith('overlap_') or mode.startswith('refined_'):scenes=calibration(mode.startswith('refined_'));selection=None
    else:
        selection=json.loads((OUT/('selector_v2.json' if mode.endswith('_v2') else 'selector_refined.json')).read_text());assert selection['accepted']
        cut=selection['switch_height']
        if mode in ('random_v2','nearcut_v2'):
            seeds=range(1154000,1154064) if mode=='random_v2' else range(1154100,1154164)
            scenes=[sample_height_terrain_115(seed,3,'development') for seed in seeds]
            if mode=='nearcut_v2':
                rng=np.random.default_rng(240024)
                scenes=[replace(s,stand_height_m=float(rng.uniform(.115,.12)),terrain='legacy',grade_deg=0.,roughness_m=0.,step_height_m=0.,height_l=0.,height_r=0.) for s in scenes]
        elif mode in ('boundary','boundary_v2','speed_boundary_v2','random_v2','nearcut_v2'):
            heights=[.115,.1175,cut-1e-6,cut,cut+1e-6,.12] if mode=='speed_boundary_v2' else sorted(set([.1175,.118,.118125,.11825,.1185,.11875,cut-1e-6,cut,cut+1e-6]))
            velocities=(.5,.749999,.75,.750001,.8,1.,-.5,-.749999,-.75,-.750001,-.8,-1.) if mode=='speed_boundary_v2' else (.5,.75,.875,.9375,1.,-.5,-.75,-.875,-.9375,-1.)
            variations=[{},dict(mass=7.5,mu_l=.6,mu_r=1.,drive_difference=.03,delay_ms=10.)]
            scenes=[replace(cases()[4],stand_height_m=h,speed=v,**change) for change in variations for h in heights for v in velocities]
        elif mode in ('registered','registered_v2'):
            r=json.loads((OUT.parent/'fifth_round_registered_robustness_20261002/registered_cases.json').read_text())
            scenes=[HeightTerrainScenario(**s) for s in r['scenarios']]
        else:
            r=json.loads((OUT.parent/'twentyfirst_round_native_reference_20261002/grid_v2/registered_cases.json').read_text())
            scenes=[HeightTerrainScenario(**s) for s in r['scenarios']]
    registration=dict(mode=mode,scenarios=[asdict(s) for s in scenes],selector=selection,
        selection_before_rollout=True,learning=False,full_admission=False)
    (output/'registered_cases.json').write_text(json.dumps(registration,indent=2)+'\n')
    plans=[np.load(OUT.parent/f'eighteenth_round_robust_reference_20261002/solve_world{d}/best.npz')['schedule'] for d in range(2)]
    saved=np.load(PRIOR/'template/trajectory.npz');templates=[saved[f'features{d}'] for d in range(2)]
    ref,Q,R,P=lqr_cost(model(HeightTerrainScenario(stand_height_m=.115)));c,_,_=vmc_coordinates(ref);lift=np.linalg.pinv(c[:6]);metric=lift.T@P@lift
    use=np.array([mode in ('overlap_virtual','refined_virtual') if selection is None else s.stand_height_m<selection['switch_height'] and abs(s.speed)>selection.get('reference_min_speed',0.) for s in scenes])
    np.savez_compressed(output/'metric.npz',metric=metric,P=P,C=c[:6],lift=lift)
    env=NativeEnv.height115_candidate(n=len(scenes),scenario=scenes,residual_scale=0,nominal_correction=True);n=len(scenes)
    try:
        env.reset();execution=core.graph(env);feature=wp.zeros((n,15),dtype=core.D)
        args=[env.data.qpos,env.data.qvel,env.data.sensordata,env.ids,env.k['heights'],env.k['angles'],env.k['state'],feature]
        cursor=np.full(n,-1,int);phase=np.zeros(n,int);entry=np.zeros(n);finished=np.zeros(n,bool);current=np.zeros((n,6));records=[];requests=[]
        max_iterations=int(np.ceil((env.param.numpy()[:,3].max()+2.)/.005))+2
        for iteration in range(max_iterations):
            state=env.state.numpy();active=env.active.numpy()!=0;wp.launch(core.features,n,args);x=feature.numpy();target=current.copy()
            for w in np.flatnonzero(active&(state[:,1]>=0)):
                cursor[w]+=1;d=int(scenes[w].speed<0)
                if cursor[w]==0:entry[w]=x[w,3]
                assert abs(entry[w])>1e-6
                finished[w] |= x[w,6]>0
                if not use[w] or finished[w]:target[w]=0;continue
                template=templates[d];end=np.flatnonzero(template[:,6]>0);last=int(end[0])-1 if len(end) else len(template)-1
                choices=np.arange(phase[w],min(phase[w]+2,last)+1)
                if not len(choices):choices=np.array([phase[w]])
                query=x[w,:6].copy();query[3]*=template[0,3]/entry[w];delta=template[choices,:6]-query
                phase[w]=int(choices[np.argmin(np.einsum('ni,ij,nj->n',delta,metric,delta))])
                reference_J=template[phase[w],7:15].reshape(2,2,2)
                virtual=np.linalg.solve(reference_J,plans[d][phase[w],:4].reshape(2,2,1))[:,:,0].mean(axis=0)
                target[w,:4]=(x[w,7:15].reshape(2,2,2)@virtual).reshape(4);target[w,4:]=plans[d][phase[w],4:].mean()
                target[w]/=max(1.,float(abs(target[w]).max()))
            delta=target-current;current=np.clip(current+delta*np.minimum(1.,.1/np.maximum(abs(delta).sum(axis=1),1e-30))[:,None],-1,1)
            env.set_nominal_correction(current);wp.capture_launch(execution)
            records.append(np.c_[state[:,0],state[:,1],cursor,phase,finished,entry,x]);requests.append(current.copy())
            if env.done.numpy().all():break
            if iteration%500==0:print('PROGRESS',mode,iteration,flush=True)
        assert env.done.numpy().all()
        rows=[{k:v for k,v in r.items() if k!='terminal_observation'} for r in env.step_wait()[3]]
        assert all(not e['success'] or e['physical_steps']==e['physical_evidence_steps'] for e in rows)
        np.savez_compressed(output/'trajectory.npz',records=np.array(records),requests=np.array(requests),use_reference=use)
        result=dict(total=n,episodes=rows,physical=sum(e['physical_safety_passed'] for e in rows),design=sum(e['design_joint_passed'] for e in rows),success=sum(e['success'] for e in rows),
            controller_selection=use.tolist(),learning=False,full_admission=False,trajectory_sha256=sha(output/'trajectory.npz'),
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),PRIOR/'experiment.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('RESULT',mode,result['physical'],result['design'],result['success'],'/',n,flush=True)
        print('FAILED',[(w,scenes[w].stand_height_m,scenes[w].speed,e['stop_distance_m'],e['tail_speed_m_s'],e['min_active_design_margin_rad']) for w,e in enumerate(rows) if not e['success']],flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('overlap_zero','overlap_virtual','refined_zero','refined_virtual','choose','choose_refined','boundary','registered','broad','boundary_v2','registered_v2','broad_v2','speed_boundary_v2','random_v2','nearcut_v2'));mode=p.parse_args().mode
    choose(mode=='choose_refined') if mode in ('choose','choose_refined') else run(mode)
