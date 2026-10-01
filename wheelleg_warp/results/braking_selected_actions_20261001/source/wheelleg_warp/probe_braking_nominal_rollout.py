"""Causal5ms nominal-command rollout from known controller memory; fixed flat model predictor."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco_warp as mjw
import numpy as np
import warp as wp
from native.controller import D,V6,command_bounds,control_physical
from native.models import batch
from native.terrain import model,HeightTerrainScenario
from probe_braking_phase_chart import physical_metrics
from probe_height_115_contact_action_pair import basis
from probe_height_115_action_predict_loow import sha


@wp.kernel
def apply_extra(v:wp.array2d[float],ids:wp.array[int],upper:wp.array2d[D],extra:wp.array2d[D],ctrl:wp.array2d[float]):
    w=wp.tid();speed=V6();gain=V6()
    for j in range(6):speed[j]=D(v[w,ids[4+j]]);gain[j]=upper[w,j]
    bound=command_bounds(speed,gain)
    for j in range(6):ctrl[w,j]=float(wp.clamp(D(ctrl[w,j])+extra[w,j],-bound[j],bound[j]))


def run(source,output,cold_start=False,model_start=False,common_mixes=False,selection=None):
    assert not (cold_start and model_start)
    assert not (common_mixes and selection is not None)
    source=source.resolve();output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    meta=json.loads((source/'verification.json').read_text());col=meta['columns']
    cfg=np.load(source/'controller_inputs.npz',allow_pickle=False);window=np.load(source/'windows.npz',allow_pickle=False)
    scenes=[HeightTerrainScenario(**s) for s in meta['scenarios']];models=[model(s) for s in scenes]
    samples=[];starts=[];states=[];references=[]
    for w in range(6):
        r=window[f'w{w}_braking_radial'];arrival=float(r[:,3].max())
        for elapsed in (.1,.5,1.):
            index=int(np.argmin(abs(r[:,0]-arrival-elapsed)));assert abs(r[index,0]-arrival-elapsed)<1e-9 and r[index,2]==0
            samples.append(dict(world=w,elapsed_s=elapsed));starts.append(window[f'w{w}_braking_pre'][index])
            states.append(window[f'w{w}_braking_controller_after_nominal'][index]);references.append(cfg['reference'][w])
    starts=np.array(starts);n=len(starts);assert n==18
    extra=np.zeros((n,6));arm_names=['zero']
    if selection is not None:
        selection=selection.resolve();chosen=np.load(selection/'selection.npz',allow_pickle=False)
        record=json.loads((selection/'verification.json').read_text())
        assert record['actual_future_used_for_selection'] is False and len(record['rows'])==n
        for a,b in zip(samples,record['rows']):assert a['world']==b['world'] and a['elapsed_s']==b['elapsed_s']
        extra=chosen['extra'];assert extra.shape==(n,6) and np.isfinite(extra).all() and np.max(np.sum(abs(extra),axis=1))<=.10000001
        samples=[dict(**s,arm=r['selected_arm']) for s,r in zip(samples,record['rows'])];arm_names=['frozen_model_selected_action']
    if common_mixes:
        arm_names=['zero']+[f'{name}{sign}' for name in ('F','H','W','FH','HW','FW') for sign in ('+','-')]
        actions=[]
        for i,sample in enumerate(samples):
            b=basis(models[sample['world']],starts[i,1:col['qvel_start']],starts[i,col['qvel_start']:col['warmstart_start']])
            b=b/np.sum(abs(b),axis=0)*.1
            directions=[b[:,j] for j in range(3)]+[(b[:,0]+b[:,1])/2,(b[:,1]+b[:,2])/2,(b[:,0]+b[:,2])/2]
            actions.append(np.array([np.zeros(6)]+[signed for d in directions for signed in (d,-d)]))
        extra=np.concatenate(actions);assert np.max(np.sum(abs(extra),axis=1))<=.10000000001
        starts=np.repeat(starts,13,axis=0);states=np.repeat(states,13,axis=0);references=np.repeat(references,13,axis=0)
        samples=[dict(**s,arm=arm) for s in samples for arm in arm_names];n=len(starts);assert n==234
    flat=HeightTerrainScenario(stand_height_m=.115);nominal=model(flat)
    cpus=[models[s['world']] for s in samples]+[nominal]*n
    q=starts[:,1:col['qvel_start']].copy();v=starts[:,col['qvel_start']:col['warmstart_start']]
    phase=[nominal.joint(name).qposadr[0] for name in ('wheel1','wheel2')];q[:,phase]=(q[:,phase]+np.pi)%(2*np.pi)-np.pi
    Q=np.tile(q,(2,1)).astype(np.float32);V=np.tile(v,(2,1)).astype(np.float32)
    W=np.tile(starts[:,col['warmstart_start']:col['ctrl_start']],(2,1)).astype(np.float32)
    if cold_start or model_start:W[n:]=0
    wp.init();wp.set_device('cuda:0');_,wm,data,_=batch(cpus,[scenes[s['world']] for s in samples]+[flat]*n)
    data.qpos.assign(Q);data.qvel.assign(V);data.qacc_warmstart.assign(W);data.ctrl.assign(np.tile(starts[:,col['ctrl_start']:],(2,1)).astype(np.float32))
    ids=wp.array(cfg['ids'],dtype=int);gain_upper=wp.array(np.tile(cfg['public_gain_upper'],(2*n,1)),dtype=D)
    known_extra=wp.array(np.tile(extra,(2,1)),dtype=D)
    wp.launch(apply_extra,2*n,[data.qvel,ids,gain_upper,known_extra,data.ctrl])
    if model_start:
        # Initialize numerical state from the predictor's OWN current-state solve.
        # No integration, future observation or actual-plant warmstart is used.
        _,seed_model,seed,_=batch([nominal]*n,[flat]*n)
        seed.qpos.assign(Q[n:]);seed.qvel.assign(V[n:]);seed.qacc_warmstart.assign(np.zeros_like(W[n:]));seed.ctrl.assign(data.ctrl.numpy()[n:])
        mjw.forward(seed_model,seed)
        assert np.array_equal(seed.qpos.numpy(),Q[n:]) and np.array_equal(seed.qvel.numpy(),V[n:])
        W[n:]=seed.qacc.numpy();assert np.isfinite(W[n:]).all();data.qacc_warmstart.assign(W)
    state=wp.array(np.tile(states,(2,1)),dtype=D);reference=wp.array(np.tile(references,(2,1)),dtype=D)
    targets=wp.zeros((2*n,3));command=wp.zeros(2*n,dtype=D);active=wp.ones(2*n,dtype=int);diag=wp.zeros((2*n,38),dtype=D)
    fixed={k:wp.array(cfg[k],dtype=D) for k in ('heights','gains','feed','angles','yaw')}
    assert int(cfg['ids'][10])==int(nominal.sensor('body_gyro').adr[0])
    rows=[];qs=[];vs=[];commands=[];physical=[];contacts=[];previous_v=V
    for step in range(10):
        issued=data.ctrl.numpy().astype(float);mjw.step(wm,data);q=data.qpos.numpy().astype(float);v=data.qvel.numpy().astype(float);force=data.actuator_force.numpy().astype(float)
        c=data.contact;count=int(data.nacon.numpy()[0]);sets=[set() for _ in cpus]
        for w,pair in zip(c.worldid.numpy()[:count],c.geom.numpy()[:count]):
            if w>=0:sets[int(w)].add(tuple(sorted(map(int,pair))))
        contacts.extend([[step,w,*pair] for w,pairs in enumerate(sets) for pair in sorted(pairs)])
        for i,sample in enumerate(samples):
            qe=float(abs(q[i]-q[n+i]).max());ve=float(abs(v[i]-v[n+i]).max());ce=float(abs(issued[i]-issued[n+i]).max());equal=sets[i]==sets[n+i]
            rows.append(dict(step=step+1,**sample,qpos_error=qe,qvel_error=ve,issued_command_error_Nm=ce,contacts_match=equal,
                             passed=qe<=2e-6 and ve<=1e-3 and equal and ce<=1e-5))
        for i,m in enumerate(cpus):
            sl=slice(i,i+1);physical.append(dict(step=step+1,world=i,**physical_metrics(m,q[sl],v[sl],force[sl],previous_v[sl],issued[sl])))
        qs.append(q);vs.append(v);commands.append(issued);previous_v=v
        if step<9:
            wp.launch(control_physical,2*n,[data.qpos,data.qvel,data.sensordata,targets,command,active,state,ids,
                fixed['heights'],fixed['gains'],fixed['feed'],fixed['angles'],reference,fixed['yaw'],data.ctrl,diag,0,0,gain_upper],block_dim=32)
            wp.launch(apply_extra,2*n,[data.qvel,ids,gain_upper,known_extra,data.ctrl])
    np.savez_compressed(output/'rollout.npz',initial_qpos=Q,initial_qvel=V,initial_warmstart=W,initial_controller=np.tile(states,(2,1)),known_extra=np.tile(extra,(2,1)),qpos=qs,qvel=vs,commands=commands,contacts=contacts)
    result=dict(role='causal_nominal_controller_memory_5ms_rollout',samples=samples,rows=rows,physical=physical,
        full_pair_pass_count=sum(r['passed'] for r in rows),total_pairs=len(rows),all_pairs_pass=all(r['passed'] for r in rows),
        all_physical_steps_pass=all(p['safe_arms']==1 for p in physical),original_gate=dict(qpos=2e-6,qvel=1e-3,contacts_equal=True,issued_command=1e-5),
        base_states=18,arms=arm_names,known_extra_L1_limit_Nm=.1 if common_mixes or selection is not None else 0.,
        prediction_plant='one fixed public7kg flat nominal model; no per-world terrain layout, random actuator gain or future trajectory',
        predictor_initial_warmstart='own current-state forward solve; no state integration' if model_start else 'zero; predictor evolves its own numerical state thereafter' if cold_start else 'actual previous numerical warmstart copied at initial state',
        future_nominal_source='same known controller code, public tables and past internal state, evaluated on predictor OWN predicted q/v and modeled sensors after each step; stop command0 known throughout5ms',
        limitations='18 finite braking states, known extra0 or L1<=0.1Nm held5ms, future Nom recalculated every0.5ms then total command allocated in same public box. Full simulation q/v still privileged for hardware; initial warmstart labeled separately. Both copies use Warp; separate from CPU/Warp physics pairing. No sensor-only, model-uncertainty, complete timing, feedback braking or full-task claim.',
        input_sha256={str((source/f).relative_to(ROOT)):sha(source/f) for f in ('verification.json','windows.npz','controller_inputs.npz')},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_braking_phase_chart.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/models.py',ROOT/'wheelleg_warp/native/terrain.py')})
    if selection is not None:result['selection_sha256']={str((selection/f).relative_to(ROOT)):sha(selection/f) for f in ('verification.json','selection.npz')}
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED pairs',result['full_pair_pass_count'],'/',len(rows),'physical',result['all_physical_steps_pass'],flush=True)
    print('max q/v/command',*[max(r[k] for r in rows) for k in ('qpos_error','qvel_error','issued_command_error_Nm')],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--cold-start',action='store_true');mode.add_argument('--model-start',action='store_true')
    action=parser.add_mutually_exclusive_group();action.add_argument('--common-mixes',action='store_true');action.add_argument('--selection',type=Path)
    args=parser.parse_args();run(args.source,args.output,args.cold_start,args.model_start,args.common_mixes,args.selection)
