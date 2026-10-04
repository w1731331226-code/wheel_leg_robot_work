"""Public-development trajectory diagnostic; original physics/reward stay unchanged."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from train_height_comparison import protocol,raw_env,summary,b1_action
from training_contract import digest,verify_checkpoint
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
COL=['time_s','body_x_m','body_y_m','yaw_rad','return','yaw_cost','dense_return',
     'contact_mask','design_margin_rad','min_actual_chain_m','irreversible_constraint_breach']

def run(out):
    config=protocol(P);cases=config['selection'];classic=json.loads((P/'classical_selection.json').read_text())['selected']['candidate']
    jobs=[('B1','diff3',None)]
    for method,mode in config['methods'].items():
        record=json.loads((P/'runs'/method/'1610/step_2000000.json').read_text())
        verify_checkpoint(record['path'],record);jobs.append((method,mode,record))
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'preregistration.json',dict(role='Exploratory diagnosis after failed old gate; public32 only',
        policy_choice='B1 plus all three final2M policies of seed1610, chosen before new results',
        cases=cases,jobs=[dict(label=l,mode=m,checkpoint=r) for l,m,r in jobs],
        protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__),
        telemetry_hz=50,physical_and_reward_hz=2000,training=False,
        old_gate_or_sealed_final_replayed=False,no_control_reward_or_constraints_changed=True))
    results={}
    for label,mode,record in jobs:
        raw=raw_env(cases,mode);env=raw;agent=None;stats=None
        if record:
            env=VecNormalize.load(record['path']+'.pkl',VecCheckNan(raw,raise_exception=True))
            env.training=False;env.norm_reward=False
            stats=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count)
            agent=PPO.load(record['path']+'.zip',device='cuda')
        param=raw.param.numpy();ids=raw.ids.numpy()
        pending=set(range(len(cases)));rows=[None]*len(cases);traces=[[] for _ in cases]
        returned=np.zeros(len(cases));first_breach=[None]*len(cases)
        try:
            obs=env.reset();deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
            for step in range(deadline):
                actions=agent.predict(obs,deterministic=True)[0] if agent else np.stack([b1_action(o,classic) for o in obs])
                obs,reward,done,infos=env.step(actions)
                state=raw.state.numpy();qpos=raw.data.qpos.numpy();stopped=raw.stopped_q.numpy()
                for w in list(pending):
                    returned[w]+=float(reward[w]);info=infos[w] if done[w] else None
                    q=stopped[w] if info else qpos[w]
                    qw,qx,qy,qz=map(float,q[3:7]);yaw=float(np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
                    if info:
                        t=info['duration_s'];total=info['episode']['r'];terminal=10 if info['success'] else -10
                        yaw_cost=np.deg2rad(info['rms_deg'][2])**2*t/.08726646**2
                        mask=info['touched_contact_mask']|info['touched_terrain_contact_mask']
                        design=info['min_active_design_margin_rad'];chain=info['min_leg_m']
                        breach=not info['physical_safety_passed'] or not info['design_joint_passed']
                        peaks=info['relative_peak_deg'] if cases[w]['scenario']['relative_attitude'] else info['peak_deg']
                        breach=breach or max(peaks)>5 or max(info['peak_deg'][:2])>10
                    else:
                        t=float(state[w,0]*.0005);total=float(state[w,20]);terminal=0
                        yaw_cost=float(state[w,13]/.08726646**2);mask=int(state[w,14])
                        design=float(state[w,38]);chain=float(min(state[w,31],state[w,32]))
                        peaks=state[w,21:24] if cases[w]['scenario']['relative_attitude'] else state[w,8:11]
                        breach=design<0 or chain<param[w,15] or state[w,33]<0 or max(state[w,35:37])>1e-6
                        breach=breach or max(peaks)>np.deg2rad(5) or max(state[w,8:10])>np.deg2rad(10)
                    if breach and first_breach[w] is None:first_breach[w]=float(t)
                    traces[w].append([t,float(q[0]),float(q[1]),yaw,float(total),float(yaw_cost),
                                      float(total-terminal),mask,design,chain,int(breach)])
                    if not info:continue
                    pending.remove(w)
                    mean_fk=float(np.mean([sim.fk_joints(float(q[ids[2*s]]),float(q[ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    trace=np.asarray(traces[w]);assert np.isfinite(trace).all() and np.all(np.diff(trace[:,0])>0)
                    np.testing.assert_allclose(returned[w],total,atol=2e-5,rtol=0)
                    delay=float(t-first_breach[w]) if first_breach[w] is not None else None
                    rows[w]=dict(**cases[w],**{k:v for k,v in info.items() if k!='terminal_observation'},
                        final_mean_fk_leg_m=mean_fk,maximum_body_cross_track_m=float(np.max(abs(trace[:,2]))),
                        first_irreversible_constraint_breach_policy_s=first_breach[w],
                        terminal_feedback_delay_from_breach_s=delay,
                        exact_dense_return=float(total-terminal),exact_yaw_penalty=float(yaw_cost),
                        snapshot_return_after_first_breach=float(total-terminal-trace[np.flatnonzero(trace[:,10])[0],6]) if first_breach[w] is not None else None)
                if not pending:break
            assert not pending,'Diagnostic did not collect every first episode'
            result=summary(rows)
            if stats:
                np.testing.assert_array_equal(stats[0],env.obs_rms.mean);np.testing.assert_array_equal(stats[1],env.obs_rms.var);assert stats[2]==env.obs_rms.count
            offsets=np.cumsum([0]+[len(t) for t in traces])
            np.savez_compressed(out/(label+'_trace.npz'),columns=np.asarray(COL),offsets=offsets,
                                trace=np.concatenate([np.asarray(t) for t in traces]))
            results[label]=dict(summary=result,runs=rows)
            atomic_json(out/(label+'.json'),results[label])
            print(label,result,'mean|max cross-track',float(np.mean([r['maximum_body_cross_track_m'] for r in rows])),max(r['maximum_body_cross_track_m'] for r in rows),flush=True)
        finally:env.close()
    atomic_json(out/'summary.json',dict(results={l:r['summary'] for l,r in results.items()},
        additional_public_diagnostic_episodes=128,training_updates=0,old_gate_or_final_replayed=False,
        inference='50Hz breach timing bracket, unchanged2kHz integrated return/gates; trajectory/return association is not causal proof',
        protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
