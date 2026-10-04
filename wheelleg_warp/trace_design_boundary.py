"""Registered read-only2kHz low-height working-domain boundary measurement."""
from pathlib import Path
import argparse,json
import numpy as np
import warp as wp
from stable_baselines3 import PPO
import route_pilot_env as experiment
from native.controller import D,control_physical_nominal
from native.environment import after,collect_physical
from smoke_reward_training import weight_digest
from training_contract import digest
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/route_pilot_v1'
COL=['valid','time_s']+[f'pre_q_{i}' for i in range(4)]+[f'pre_v_{i}' for i in range(4)]+[f'post_q_{i}' for i in range(4)]+[f'post_v_{i}' for i in range(4)]+[f'nominal_command_{i}' for i in range(6)]+[f'actor_command_{i}' for i in range(6)]+[f'total_command_{i}' for i in range(6)]+['lambda','contact_mask','running_min_leg_m','running_physical_joint_margin_rad','running_design_margin_rad']+[f'filtered_request_{i}' for i in range(3)]+[f'target_request_{i}' for i in range(3)]+['instant_design_margin_rad']+[f'actual_actuator_torque_{i}' for i in range(6)]


@wp.kernel
def pre_record(slot:int,q:wp.array2d[float],v:wp.array2d[float],ids:wp.array[int],active:wp.array[int],out:wp.array3d[D]):
    w=wp.tid()
    if active[w]==0:return
    for j in range(4):out[slot,w,2+j]=D(q[w,ids[j]]);out[slot,w,6+j]=D(v[w,ids[4+j]])


@wp.kernel
def post_record(slot:int,q:wp.array2d[float],v:wp.array2d[float],ids:wp.array[int],active:wp.array[int],
                task:wp.array2d[D],memory:wp.array2d[D],diag:wp.array2d[D],ctrl:wp.array2d[float],
                force:wp.array2d[float],targets:wp.array2d[float],flags:wp.array2d[int],out:wp.array3d[D]):
    w=wp.tid();out[slot,w,0]=D(active[w])
    if active[w]==0:return
    out[slot,w,1]=(task[w,0]+D(1))*D(.0005);margin=D(1.e30)
    for j in range(4):
        out[slot,w,10+j]=D(q[w,ids[j]]);out[slot,w,14+j]=D(v[w,ids[4+j]])
        margin=wp.min(margin,D(1.4)-wp.abs(D(q[w,ids[j]])))
    for j in range(6):
        out[slot,w,18+j]=diag[w,j];out[slot,w,24+j]=diag[w,6+j];out[slot,w,30+j]=D(ctrl[w,j]);out[slot,w,48+j]=D(force[w,j])
    out[slot,w,36]=diag[w,12];out[slot,w,37]=D(flags[w,1])
    out[slot,w,38]=wp.min(task[w,31],task[w,32]);out[slot,w,39]=task[w,33];out[slot,w,40]=task[w,38]
    for j in range(3):out[slot,w,41+j]=memory[w,16+j];out[slot,w,44+j]=D(targets[w,j])
    out[slot,w,47]=margin


def measured(cases,model,normalization,classical,out,label):
    factory_before=experiment.raw_env;launch_before=wp.launch;chunks=[[] for _ in cases]
    def factory(selected,mode):
        assert mode=='diff3' and selected==cases
        n=len(cases);buffer=wp.zeros((40,n,len(COL)),dtype=D);slot=0;force=None;control=None;pre_count=0
        def launch(kernel,dim,inputs=None,**kwargs):
            nonlocal slot,force,control,pre_count
            if kernel is control_physical_nominal:
                assert slot<40;control=inputs
                launch_before(pre_record,n,[slot,inputs[0],inputs[1],inputs[7],inputs[5],buffer]);pre_count+=1
            answer=launch_before(kernel,dim,**kwargs) if inputs is None else launch_before(kernel,dim,inputs,**kwargs)
            if kernel is collect_physical:force=inputs[2]
            if kernel is after:
                # Inject before after() for the final active sample; the original
                # call has already been captured, so place via the branch below.
                slot+=1
            return answer
        def capture_launch(kernel,dim,inputs=None,**kwargs):
            if kernel is after:
                assert force is not None and control is not None and slot<40
                launch_before(post_record,n,[slot,inputs[0],inputs[1],inputs[6],inputs[13],inputs[9],inputs[10],inputs[11],control[14],force,control[3],inputs[5],buffer])
            return launch(kernel,dim,inputs,**kwargs)
        wp.launch=capture_launch
        try:env=factory_before(cases,mode)
        finally:wp.launch=launch_before
        assert slot==pre_count==40
        pending=set(range(n));wait=env.step_wait
        def step_wait():
            result=wait();data=buffer.numpy()
            for w in list(pending):
                trace=data[data[:,w,0]>0,w].copy();assert len(trace)>0
                chunks[w].append(trace)
                if result[2][w]:pending.remove(w)
            return result
        env.step_wait=step_wait
        return env
    experiment.raw_env=factory
    before=(model.num_timesteps,model._n_updates,weight_digest(model)) if model else None
    try:result=experiment.evaluate(cases,'L2-route' if model else 'M3-route',model,normalization,classical)
    finally:experiment.raw_env=factory_before;wp.launch=launch_before
    if model:assert before==(model.num_timesteps,model._n_updates,weight_digest(model))
    arrays=[np.concatenate(c) for c in chunks]
    for r,t in zip(result['runs'],arrays):
        assert len(t)==r['physical_steps']==r['physical_evidence_steps']
        np.testing.assert_allclose(t[:,1],np.arange(1,len(t)+1)*.0005,rtol=0,atol=1e-9)
        np.testing.assert_allclose(t[:,40],np.minimum.accumulate(t[:,47]),rtol=0,atol=1e-12)
        assert r['design_joint_passed']==bool(t[:,47].min()>=0)
        assert np.isfinite(t).all()
        if model:assert np.all(t[:,28:30]==0) and np.all(t[:,43]==0) and np.all(t[:,46]==0)
    np.savez_compressed(out/(label+'_trace.npz'),columns=np.array(COL),offsets=np.cumsum([0]+[len(t) for t in arrays]),trace=np.concatenate(arrays))
    atomic_json(out/(label+'.json'),result)
    return dict(summary=result['summary'],physical=result['physical'],design=result['design'],physics_samples=sum(len(t) for t in arrays))


def run(out):
    p=json.loads((PILOT/'proposal.json').read_text());review=json.loads((PILOT/'study_review.json').read_text())
    assert review['verified'] and all(digest(ROOT/n)==v for n,v in p['source_sha256'].items())
    cases=[r for panel in ['regular','controlled'] for r in p[panel] if r['scenario']['stand_height_m']<.16]
    assert len(cases)==27 and len(set(r['seed'] for r in cases))==27
    out.mkdir(parents=True,exist_ok=False);models=[]
    for seed in p['training_seeds']:
        prefix=PILOT/'runs/L2-route'/str(seed)/'step_200000';c=json.loads(prefix.with_suffix('.json').read_text())['checkpoint']
        assert digest(prefix.with_suffix('.zip'))==c['checkpoint_sha256'] and digest(prefix.with_suffix('.pkl'))==c['normalization_sha256']
        models.append(dict(seed=seed,prefix=str(prefix),checkpoint=c))
    reg=dict(version='design-boundary-2khz-v1',cases=cases,models=models,classical={k:p['classical'][k] for k in ['B0','B1']},columns=COL,
        selection='All19 regular and8 controlled development cases with requested height<160mm; three fixed L2 final models and two unchanged classic references; not selected by failures.',
        budget_episodes=135,training_updates=0,physics_hz=2000,actor_hz=50,
        injection='Pre kernel before original controller, post kernel after physics/contact/physical evidence and before native.after; read-only private output buffers. No model/controller/reward/order change.',
        interpretation='First crossing and pre/post joint motion, accepted Nom/Actor command decomposition and contact flags. Torque sign alone does not determine acceleration in coupled closed-chain/contact dynamics.',
        source_sha256={**p['source_sha256'],'wheelleg_warp/trace_design_boundary.py':digest(__file__)},pilot_review_sha256=digest(PILOT/'study_review.json'),old_gate_or_final_used=False)
    atomic_json(out/'registration.json',reg);records=[];completed=0
    jobs=[(k,None,c) for k,c in reg['classical'].items()]+[(str(m['seed']),m,None) for m in models]
    try:
        for label,m,c in jobs:
            atomic_json(out/'progress.json',dict(status='running',completed_episodes=completed,pending_job=label))
            assert all(digest(ROOT/n)==v for n,v in reg['source_sha256'].items())
            model=PPO.load(m['prefix']+'.zip',device='cuda') if m else None
            stats=measured(cases,model,m['prefix']+'.pkl' if m else None,c,out,label)
            records.append(dict(label=label,stats=stats,result_sha256=digest(out/(label+'.json')),trace_sha256=digest(out/(label+'_trace.npz'))))
            completed+=len(cases);atomic_json(out/'completed_jobs.json',dict(completed_episodes=completed,records=records));print('COMPLETED',completed,label,stats,flush=True)
        assert completed==135
        atomic_json(out/'completion.json',dict(completed_episodes=completed,training_updates=0,records=records,registration_sha256=digest(out/'registration.json')))
        atomic_json(out/'progress.json',dict(status='complete',completed_episodes=completed))
    except BaseException as e:
        atomic_json(out/'interruption.json',dict(completed_episodes=completed,error=str(e),pending_job_consumption_unknown=True,silently_resumable=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
