"""Frozen new-state holdout collector; read-only events, no controller changes."""
from pathlib import Path
import json
import numpy as np
import warp as wp
from stable_baselines3 import PPO
import route_pilot_env as experiment
from native.controller import D,control_physical_nominal
from native.environment import command_step,after
from probe_joint_response import save_pre,WIDTH
from smoke_reward_training import weight_digest
from training_contract import digest
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/contact_state_holdout_v1'
SNAP_WIDTH=WIDTH+2


@wp.kernel
def save_holdout(q:wp.array2d[float],v:wp.array2d[float],active:wp.array[int],flags:wp.array2d[int],
                 pre:wp.array2d[D],ever:wp.array[int],filled:wp.array2d[int],snapshots:wp.array3d[D]):
    w=wp.tid()
    if active[w]==0:return
    contact=flags[w,1];old=ever[w]
    for event in range(3):
        selected=(event==0 and pre[w,49]>=D(1.5)) or (event==1 and pre[w,49]>=D(3.5)) or (event==2 and old==0 and contact!=0)
        if selected and filled[w,event]==0:
            filled[w,event]=1
            for j in range(68):snapshots[w,event,j]=pre[w,j]
            for j in range(17):snapshots[w,event,68+j]=D(q[w,j])
            for j in range(16):snapshots[w,event,85+j]=D(v[w,j])
            snapshots[w,event,101]=D(old);snapshots[w,event,102]=D(contact)
    ever[w]=old|contact


def capture(cases,agent,normalization,classical,out,label):
    original_factory=experiment.raw_env;original_launch=wp.launch;captured={}
    def factory(selected,mode):
        assert selected==cases and mode=='diff3'
        n=len(cases);pre=wp.zeros((n,WIDTH),dtype=D);filled=wp.zeros((n,3),dtype=int);ever=wp.zeros(n,dtype=int);snap=wp.zeros((n,3,SNAP_WIDTH),dtype=D)
        warm=None;task=None
        def launch(kernel,dim,inputs=None,**kwargs):
            nonlocal warm,task
            if kernel is command_step:warm=inputs[6];task=inputs[0]
            if kernel is after:original_launch(save_holdout,n,[inputs[0],inputs[1],inputs[13],inputs[5],pre,ever,filled,snap])
            result=original_launch(kernel,dim,**kwargs) if inputs is None else original_launch(kernel,dim,inputs,**kwargs)
            if kernel is control_physical_nominal:original_launch(save_pre,n,[inputs[0],inputs[1],warm,inputs[14],inputs[15],task,inputs[5],pre])
            return result
        wp.launch=launch
        try:env=original_factory(cases,mode)
        finally:wp.launch=original_launch
        assert env.cpu.nq==17 and env.cpu.nv==16 and env.cpu.na==0 and env.cpu.nmocap==0
        assert np.all(env.data.qfrc_applied.numpy()==0) and np.all(env.data.xfrc_applied.numpy()==0)
        saved=[];unavailable=[];pending=set(range(n));wait=env.step_wait
        def step_wait():
            result=wait()
            for w in list(pending):
                if result[2][w]:
                    marks=filled.numpy();data=snap.numpy()
                    for e in range(3):
                        if marks[w,e]:saved.append(dict(world=w,event=e,state=data[w,e].copy()))
                        else:unavailable.append(dict(world=w,event=e,reason=result[3][w]['reason'],duration_s=result[3][w]['duration_s']))
                    marks[w,:]=1;filled.assign(marks);pending.remove(w)
            return result
        env.step_wait=step_wait;captured.update(saved=saved,unavailable=unavailable,ids=env.ids.numpy());return env
    experiment.raw_env=factory;before=(agent.num_timesteps,agent._n_updates,weight_digest(agent)) if agent else None
    try:result=experiment.evaluate(cases,'L2-route' if agent else 'M3-route',agent,normalization,classical)
    finally:experiment.raw_env=original_factory;wp.launch=original_launch
    if agent:assert before==(agent.num_timesteps,agent._n_updates,weight_digest(agent))
    saved=captured['saved'];states=np.stack([r['state'] for r in saved]) if saved else np.empty((0,SNAP_WIDTH))
    worlds=np.array([r['world'] for r in saved],int);events=np.array([r['event'] for r in saved],int)
    assert len(saved)+len(captured['unavailable'])==len(cases)*3 and np.isfinite(states).all()
    for s,e in zip(states,events):
        if e in [0,1]:assert [1.5,3.5][e]-1e-10<=s[49]<[1.5,3.5][e]+.0005+1e-10
        else:assert s[101]==0 and s[102]>0
    np.savez_compressed(out/(label+'_states.npz'),state=states,world=worlds,event=events,ids=captured['ids'])
    atomic_json(out/(label+'_episodes.json'),result);atomic_json(out/(label+'_missing_events.json'),dict(events=captured['unavailable']))
    return dict(snapshots=len(saved),missing_events=len(captured['unavailable']),summary=result['summary'],physical=result['physical'],design=result['design'])


def freeze():
    p=json.loads((OUT/'registration.json').read_text());assert p['rollout_budget']==192 and not (OUT/'collector_contract.json').exists()
    pilot=json.loads((OUT.parent/'route_pilot_v1/proposal.json').read_text())
    files=['collect_contact_holdout.py','probe_joint_response.py','route_pilot_env.py','route_state.py','probe_packet_odometry.py']
    sources={**pilot['source_sha256'],**{f'wheelleg_warp/{n}':digest(ROOT/'wheelleg_warp'/n) for n in files}}
    assert all(digest(ROOT/n)==v for n,v in sources.items())
    atomic_json(OUT/'collector_contract.json',dict(registration_sha256=digest(OUT/'registration.json'),source_sha256=sources,
        classical_B1=pilot['classical']['B1'],events='Scheduled pre1.5/3.5 and first new target contact; ordinary floor support is not included in target mask.',snapshot_width=SNAP_WIDTH,
        limits='Derived time/static direct motors; model-state flags and predictor inputs unchanged. Fullstate only logged offline.',training_updates=0))
    print('FROZEN independent state collector before any new rollout',flush=True)


def run():
    p=json.loads((OUT/'registration.json').read_text());contract=json.loads((OUT/'collector_contract.json').read_text())
    assert contract['registration_sha256']==digest(OUT/'registration.json') and all(digest(ROOT/n)==v for n,v in contract['source_sha256'].items())
    out=OUT/'collection';out.mkdir(exist_ok=False);records=[];completed=0
    jobs=[('B1',None)]+[(str(m['seed']),m) for m in p['models']]
    try:
        for label,m in jobs:
            atomic_json(OUT/'collection_progress.json',dict(status='running',completed_episodes=completed,pending_job=label))
            if m:
                prefix=Path(m['prefix']);assert digest(prefix.with_suffix('.zip'))==m['checkpoint']['checkpoint_sha256'] and digest(prefix.with_suffix('.pkl'))==m['checkpoint']['normalization_sha256']
                agent=PPO.load(str(prefix)+'.zip',device='cuda');norm=str(prefix)+'.pkl';classic=None
            else:agent=None;norm=None;classic=contract['classical_B1']
            result=capture(p['cases'],agent,norm,classic,out,label)
            record=dict(label=label,result=result,states_sha256=digest(out/(label+'_states.npz')),episodes_sha256=digest(out/(label+'_episodes.json')),missing_sha256=digest(out/(label+'_missing_events.json')))
            records.append(record);completed+=48;atomic_json(OUT/'collection_jobs.json',dict(completed_episodes=completed,records=records));print('COMPLETED holdout',completed,label,result,flush=True)
        assert completed==192 and sum(r['result']['snapshots']+r['result']['missing_events'] for r in records)==576
        assert all(digest(ROOT/n)==v for n,v in contract['source_sha256'].items())
        atomic_json(OUT/'collection_completion.json',dict(completed_episodes=completed,records=records,training_updates=0,collector_contract_sha256=digest(OUT/'collector_contract.json')))
        atomic_json(OUT/'collection_progress.json',dict(status='complete',completed_episodes=completed))
    except BaseException as e:
        atomic_json(OUT/'collection_interruption.json',dict(completed_episodes=completed,error=str(e),pending_job_consumption_unknown=True,silently_resumable=False));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args();globals()[args.command]()
