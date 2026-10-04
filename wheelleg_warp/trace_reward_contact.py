"""Exploratory worst-pair fixed-policy trajectory replay; no policy updates."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from train_height_comparison import raw_env,summary
from training_contract import digest
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

S=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reward_pilot_v1'
COL=['end_s','body_x_m','body_y_m','yaw_rad','contact_mask','pre_step_value_prediction',
     'action_force_diff','action_leg_moment_diff','action_wheel_diff','pre_step_body_y_m']

def run(out):
    proposal=json.loads((S/'proposal.json').read_text());cases=proposal['development']
    contract=json.loads((S/'trainer_contract.json').read_text())
    assert contract['proposal_sha256']==digest(S/'proposal.json')
    assert all(digest(ROOT/name)==value for name,value in contract['source_sha256'].items())
    models={}
    for arm in ['original','potential']:
        p=S/'runs'/arm/'1610';record=json.loads((p/'step_200000.json').read_text())
        assert digest(p/'step_200000.zip')==record['checkpoint']['checkpoint_sha256']
        assert digest(p/'step_200000.pkl')==record['checkpoint']['normalization_sha256']
        models[arm]=dict(prefix=str(p/'step_200000'),checkpoint=record['checkpoint'])
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'registration.json',dict(role='Exploratory after failed pilot; seed1610 selected because largest decline, not independent validation',
        cases=cases,models=models,policy_steps=200000,actions='fixed deterministic means',sample_hz=50,physics_hz=2000,
        columns=COL,source_sha256=digest(Path(__file__)),trainer_contract_sha256=digest(S/'trainer_contract.json'),
        original_pilot_scores_not_overwritten=True,policy_updates=0,normalization_updates=0,
        value_scope='Predictions of critic trained under each arm/stochastic policy, not calibrated deterministic-policy returns or MonteCarlo targets'))
    results={}
    for arm in ['original','potential']:
        raw=raw_env(cases,'diff3');env=VecNormalize.load(models[arm]['prefix']+'.pkl',VecCheckNan(raw,raise_exception=True))
        env.training=False;env.norm_reward=False;agent=PPO.load(models[arm]['prefix']+'.zip',device='cuda')
        before=(agent.num_timesteps,agent._n_updates);stats=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count)
        rows=[None]*len(cases);pending=set(range(len(cases)));traces=[[] for _ in cases];ids=raw.ids.numpy()
        try:
            obs=env.reset();deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
            for _ in range(deadline):
                pre_y=raw.data.qpos.numpy()[:,1].copy();action=agent.predict(obs,deterministic=True)[0]
                with torch.no_grad():value=agent.policy.predict_values(torch.as_tensor(obs,device=agent.device)).cpu().numpy().reshape(-1)
                obs,_,done,infos=env.step(action);qpos=raw.data.qpos.numpy();state=raw.state.numpy();stopped=raw.stopped_q.numpy()
                for w in list(pending):
                    info=infos[w] if done[w] else None;q=stopped[w] if info else qpos[w]
                    time=info['duration_s'] if info else float(state[w,0]*.0005)
                    mask=(info['touched_contact_mask']|info['touched_terrain_contact_mask']) if info else int(state[w,14])
                    qw,qx,qy,qz=map(float,q[3:7]);yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
                    traces[w].append([time,float(q[0]),float(q[1]),float(yaw),mask,float(value[w]),*action[w].tolist(),float(pre_y[w])])
                    if not info:continue
                    t=np.asarray(traces[w]);assert np.isfinite(t).all() and np.all(np.diff(t[:,0])>0)
                    assert np.max(abs(t[:,6:9]))<=1
                    touch=np.flatnonzero(t[:,4].astype(int)&12)
                    center=np.flatnonzero(np.sign(cases[w]['scenario']['speed'])*t[:,1]>=cases[w]['scenario']['center'])
                    assert len(center)
                    length=float(np.mean([sim.fk_joints(float(q[ids[2*s]]),float(q[ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    rows[w]=dict(**cases[w],**{k:v for k,v in info.items() if k!='terminal_observation'},final_mean_fk_leg_m=length,
                        max_abs_body_y_m=float(abs(t[:,2]).max()),body_y_at_center_progress_m=float(t[center[0],2]),
                        first_sampled_terrain_contact_s=float(t[touch[0],0]) if len(touch) else None)
                    pending.remove(w)
                if not pending:break
            assert not pending and before==(agent.num_timesteps,agent._n_updates)
            np.testing.assert_array_equal(stats[0],env.obs_rms.mean);np.testing.assert_array_equal(stats[1],env.obs_rms.var);assert stats[2]==env.obs_rms.count
            arrays=[np.asarray(t) for t in traces]
            np.savez_compressed(out/(arm+'_trace.npz'),columns=np.array(COL),offsets=np.cumsum([0]+[len(t) for t in arrays]),trace=np.concatenate(arrays))
            stats_out=dict(summary=summary(rows),mean_max_abs_body_y_m=float(np.mean([r['max_abs_body_y_m'] for r in rows])),
                terrain_missing=sum(not r['terrain_passed'] for r in rows),mean_abs_y_at_center_m=float(np.mean([abs(r['body_y_at_center_progress_m']) for r in rows])),runs=rows)
            atomic_json(out/(arm+'.json'),stats_out);results[arm]={k:v for k,v in stats_out.items() if k!='runs'}
            print(arm,results[arm],flush=True)
        finally:env.close()
    atomic_json(out/'result.json',dict(verified=True,results=results,replayed_episodes=192,unique_cases=96,
        inference='50Hz root-body position/contact association, not exact tire/obstacle intersection or causal explanation. First-contact timing bracket20ms. FreshGPU replay may differ from original pilot; original gate/disposition remain unchanged.',
        source_sha256=digest(Path(__file__)),policy_updates=0,normalization_updates=0))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
