"""Locate the first 2 kHz difference in paired public GPU worlds."""
from collections import Counter
from pathlib import Path
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from check_repeatability_v2 import initial_arrays, delta
from native.controller import control, D
from native.environment import begin, command_step, reduce_contacts, after
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from training_contract import TASK_CONTRACT_VERSION, source_hashes
from dashboard.live_env import atomic_json as write

PANEL = ROOT / 'wheelleg_warp/results/contract_v2_baseline_checked_20260923/protocol.json'
SELECTED = (700202, 700203, 700300, 700312)


@wp.kernel
def snapshot(slot:int, q:wp.array2d[float], v:wp.array2d[float], ctrl:wp.array2d[float],
             state:wp.array2d[D], done:wp.array[int], flags:wp.array2d[int], diag:wp.array2d[D],
             out:wp.array3d[D]):
    w=wp.tid();nq=q.shape[1];nv=v.shape[1];nu=ctrl.shape[1]
    out[slot,w,0]=state[w,0]*D(.0005)
    for j in range(nq):out[slot,w,1+j]=D(q[w,j])
    for j in range(nv):out[slot,w,1+nq+j]=D(v[w,j])
    for j in range(nu):out[slot,w,1+nq+nv+j]=D(ctrl[w,j])
    out[slot,w,1+nq+nv+nu]=state[w,14]
    out[slot,w,2+nq+nv+nu]=D(done[w])
    out[slot,w,3+nq+nv+nu]=D(flags[w,0])
    out[slot,w,4+nq+nv+nu]=D(flags[w,1])
    out[slot,w,5+nq+nv+nu]=diag[w,12]
    out[slot,w,6+nq+nv+nu]=diag[w,13]


@wp.kernel
def ordered_contacts(slot:int, ncon:wp.array[int], world:wp.array[int], geom:wp.array[wp.vec2i], out:wp.array3d[int]):
    j=wp.tid();out[slot,j,0]=-1;out[slot,j,1]=-1;out[slot,j,2]=-1
    if j<ncon[0]:
        out[slot,j,0]=world[j];out[slot,j,1]=geom[j][0];out[slot,j,2]=geom[j][1]


def graph(env, record_contacts):
    n=env.num_envs;data=env.data;c=data.contact
    width=7+data.qpos.shape[1]+data.qvel.shape[1]+data.ctrl.shape[1]
    states=wp.zeros((40,n,width),dtype=D)
    pairs=wp.zeros((40,data.naconmax,3),dtype=wp.int32) if record_contacts else None
    with wp.ScopedCapture() as captured:
        wp.launch(begin,n,[env.reward])
        for slot in range(40):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,
                data.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,
                env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],
                env.k['reference'],env.k['yaw'],data.ctrl,env.diag,int(env.project_clipped_base),int(env.grouped_residual)],block_dim=32)
            mjw.step(env.model,data)
            if record_contacts:
                wp.launch(ordered_contacts,data.naconmax,[slot,data.nacon,c.worldid,c.geom,pairs])
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,c.worldid,c.geom,env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,
                env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,
                env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,
                env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
            wp.launch(snapshot,n,[slot,data.qpos,data.qvel,data.ctrl,env.state,env.done,env.contact_flags,env.diag,states])
    return captured.graph,states,pairs


def first_order_difference(left,right,cases):
    for step in range(40):
        a=[[] for _ in cases];b=[[] for _ in cases]
        for row in left[step]:
            if 0<=row[0]<len(cases):a[int(row[0])].append(tuple(int(x) for x in row[1:]))
        for row in right[step]:
            if 0<=row[0]<len(cases):b[int(row[0])].append(tuple(int(x) for x in row[1:]))
        for world,case in enumerate(cases):
            pa=a[world];pb=b[world]
            if pa!=pb:
                canonical=lambda items:Counter(tuple(sorted(pair)) for pair in items)
                return dict(physical_step=step+1,seed=case.terrain_seed,
                    pair_multisets_equal=canonical(pa)==canonical(pb),pairs=[pa,pb])
    return None


def run(output):
    frozen=json.loads(PANEL.read_text())
    assert frozen['task_contract_version']==TASK_CONTRACT_VERSION and len(frozen['panels'])==160
    for item in frozen['checkpoints'].values():
        for suffix,key in (('.zip','checkpoint_sha256'),('.pkl','normalization_sha256')):
            assert hashlib.sha256(Path(item['path']+suffix).read_bytes()).hexdigest()==item[key]
    cases=[TerrainScenario(**row['scenario']) for row in frozen['panels']]
    selected=[next(i for i,case in enumerate(cases) if case.terrain_seed==seed) for seed in SELECTED]
    assert len(set(selected))==len(SELECTED)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(task_contract_version=TASK_CONTRACT_VERSION,
        panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),source_sha256=source_hashes(__file__),
        checkpoint=frozen['checkpoints']['terrain_v3'],worlds=160,selected=SELECTED,
        training=False,holdout_evaluated=False,
        note='Two fresh identical 160-world banks, independent deterministic policy actions after their observations diverge; no step_wait/reset during the first episode.'))
    a=TerrainEnv(160,scenario=cases);b=TerrainEnv(160,scenario=cases)
    checkpoint=frozen['checkpoints']['terrain_v3']['path']
    na=VecNormalize.load(checkpoint+'.pkl',a);nb=VecNormalize.load(checkpoint+'.pkl',b)
    na.training=nb.training=False;na.norm_reward=nb.norm_reward=False
    torch.set_num_threads(1);model=PPO.load(checkpoint+'.zip',device='cpu')
    policy=[];physical_a=[];physical_b=[];covered=[]
    try:
        a.reset();b.reset();initial_a=initial_arrays(a);initial_b=initial_arrays(b)
        mismatches={name:delta(initial_a[name],initial_b[name]) for name in initial_a
                    if not np.array_equal(initial_a[name],initial_b[name])}
        assert set(mismatches)<={'sensordata'},mismatches
        with (output/'initial.npz').open('xb') as f:
            np.savez_compressed(f,**{'a_'+k:v for k,v in initial_a.items()},**{'b_'+k:v for k,v in initial_b.items()})
        first_graph_a,first_states_a,first_pairs_a=graph(a,True)
        first_graph_b,first_states_b,first_pairs_b=graph(b,True)
        later_graph_a,later_states_a,_=graph(a,False)
        later_graph_b,later_states_b,_=graph(b,False)
        nq=a.data.qpos.shape[1];nv=a.data.qvel.shape[1]
        first_state=None;first_action=None;outcomes=None
        for step in range(1,1001):
            obs_a=a.obs.numpy();obs_b=b.obs.numpy()
            action_a=model.predict(na.normalize_obs(obs_a),deterministic=True)[0]
            action_b=model.predict(nb.normalize_obs(obs_b),deterministic=True)[0]
            if first_action is None and not np.array_equal(action_a,action_b):
                i=int(np.flatnonzero(np.any(action_a!=action_b,axis=1))[0])
                first_action=dict(policy_step=step,seed=cases[i].terrain_seed,
                                  max_abs=float(np.max(abs(action_a[i]-action_b[i]))))
            a.targets.assign(action_a);b.targets.assign(action_b)
            if step==1:
                wp.capture_launch(first_graph_a);wp.capture_launch(first_graph_b)
                sa,sb=first_states_a.numpy(),first_states_b.numpy()
                ca,cb=first_pairs_a.numpy(),first_pairs_b.numpy()
                with (output/'first_20ms.npz').open('xb') as f:
                    np.savez_compressed(f,state_a=sa,state_b=sb,contacts_a=ca,contacts_b=cb)
                difference=np.any(sa[:,:,1:1+nq+nv]!=sb[:,:,1:1+nq+nv],axis=2)
                if difference.any():
                    slot,i=np.argwhere(difference)[0]
                    first_state=dict(physical_step=int(slot+1),seed=cases[i].terrain_seed,
                        qpos_max_abs=delta(sa[slot,i,1:1+nq],sb[slot,i,1:1+nq]),
                        qvel_max_abs=delta(sa[slot,i,1+nq:1+nq+nv],sb[slot,i,1+nq:1+nq+nv]),
                        worlds_different=int(difference[slot].sum()))
                first_contact=first_order_difference(ca,cb,cases)
            else:
                wp.capture_launch(later_graph_a);wp.capture_launch(later_graph_b)
            state_a=a.state.numpy();state_b=b.state.numpy()
            qa=a.data.qpos.numpy()[selected];qb=b.data.qpos.numpy()[selected]
            va=a.data.qvel.numpy()[selected];vb=b.data.qvel.numpy()[selected]
            policy.append(dict(step=step,time_s=step*.02,
                max_qpos_delta=[delta(x,y) for x,y in zip(qa,qb)],
                max_qvel_delta=[delta(x,y) for x,y in zip(va,vb)],
                contact_mask_a=state_a[selected,14].astype(int).tolist(),
                contact_mask_b=state_b[selected,14].astype(int).tolist(),
                action_max_delta=[delta(x,y) for x,y in zip(action_a[selected],action_b[selected])]))
            near=any(-.7<=min(state_a[i,24:26])<=.7 or -.7<=min(state_b[i,24:26])<=.7 for i in selected)
            if near:
                sa=(first_states_a if step==1 else later_states_a).numpy()
                sb=(first_states_b if step==1 else later_states_b).numpy()
                physical_a.append(sa[:,selected,:].copy());physical_b.append(sb[:,selected,:].copy());covered.append(step)
            done_a=a.done.numpy();done_b=b.done.numpy()
            if np.all(done_a!=0) and np.all(done_b!=0):
                outcomes=dict(a=[dict(seed=case.terrain_seed,reason=int(done_a[i]),success=bool(state_a[i,19]))
                                 for i,case in enumerate(cases)],
                              b=[dict(seed=case.terrain_seed,reason=int(done_b[i]),success=bool(state_b[i,19]))
                                 for i,case in enumerate(cases)])
                break
        if outcomes is None:raise RuntimeError('First episode did not terminate in 20 s')
        with (output/'selected_2khz.npz').open('xb') as f:
            np.savez_compressed(f,a=np.concatenate(physical_a),b=np.concatenate(physical_b),
                                policy_steps=np.asarray(covered),selected_seeds=np.asarray(SELECTED))
        flips=[dict(seed=x['seed'],first=x['success'],second=y['success'])
               for x,y in zip(outcomes['a'],outcomes['b']) if x['success']!=y['success']]
        write(output/'summary.json',dict(initial_mismatch=mismatches,first_state_difference=first_state,
            first_contact_order_difference=first_contact,first_policy_action_difference=first_action,
            selected_policy_trace=policy,selected_2khz_policy_steps=covered,
            outcomes=outcomes,success_flips=flips,steps=step,training=False,
            note='Earliest visible divergence and closed-loop growth; contact ordering is not proven as the unique cause.'))
        print('first',first_state,'contact',first_contact and (first_contact['physical_step'],first_contact['seed']),
              'flips',len(flips),'steps',step,flush=True)
    finally:na.close();nb.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.output)
