"""GPU no-update online reward/endpoint/rollout audit on all public32 cases."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from train_height_comparison import protocol,raw_env,b1_action,summary
from training_contract import digest,verify_checkpoint
from dashboard.live_env import atomic_json
from reward_potential import FailurePotential
import wheelleg_sim as sim

P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
COL=['start_s','original_reward','shaped_reward_delivered','shaped_reward_exact','phi_before','phi_after']

def run(out):
    config=protocol(P);cases=config['selection'];gamma=config['ppo']['gamma'];horizon=config['ppo']['n_steps']
    classic=json.loads((P/'classical_selection.json').read_text())['selected']['candidate']
    jobs=[('B1','diff3',None)]
    for method,mode in config['methods'].items():
        checkpoint=json.loads((P/'runs'/method/'1610/step_2000000.json').read_text());verify_checkpoint(checkpoint['path'],checkpoint)
        jobs.append((method,mode,checkpoint))
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'registration.json',dict(cases=cases,jobs=[dict(method=m,mode=v,checkpoint=r) for m,v,r in jobs],
        gamma=gamma,beta=10.,rollout_steps=horizon,training_updates=0,old_gate_or_final_replayed=False,
        intervention='Only reward potential wrapper, original38 policy/value inputs and all physics/control/constraints/gates',
        endpoint_rule='Phi(reset)=Phi(any terminal)=0; native historical irreversible breach at50Hz gives intermediatePhi=-10',
        fixture='After completed collection, inject stale potential=-10 then reset; verify one zero-action step has no carry-over',
        source_sha256={name:digest(ROOT/'wheelleg_warp'/name) for name in ['check_reward_potential.py','reward_potential.py']},
        protocol_sha256=digest(P/'protocol.json')))
    results={};all_metrics=[];probe_worlds=0
    for method,mode,checkpoint in jobs:
        raw=raw_env(cases,mode);original_wait=raw.step_wait;probe={}
        def spy():
            answer=original_wait();probe['obs']=answer[0].copy();probe['reward']=answer[1].copy();probe['done']=answer[2].copy()
            probe['physical']=[raw.data.qpos.numpy(),raw.data.qvel.numpy(),raw.data.ctrl.numpy(),raw.state.numpy()]
            return answer
        raw.step_wait=spy;shaped=FailurePotential(raw,gamma=gamma,beta=10.);env=shaped;agent=None;stats=None
        if checkpoint:
            env=VecNormalize.load(checkpoint['path']+'.pkl',VecCheckNan(shaped,raise_exception=True))
            env.training=False;env.norm_reward=False
            stats=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count)
            agent=PPO.load(checkpoint['path']+'.zip',device='cuda');before=(agent.num_timesteps,agent._n_updates)
        traces=[[] for _ in cases];pending=set(range(len(cases)));rows=[None]*len(cases);ids=raw.ids.numpy()
        try:
            obs=env.reset();assert np.all(shaped.potential==0)
            deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
            for _ in range(deadline):
                start=raw.state.numpy()[:,0]*.0005
                action=agent.predict(obs,deterministic=True)[0] if agent else np.stack([b1_action(o,classic) for o in obs])
                obs,rewards,done,infos=env.step(action)
                expected_obs=env.normalize_obs(probe['obs']) if agent else probe['obs']
                np.testing.assert_array_equal(obs,expected_obs);np.testing.assert_array_equal(done,probe['done'])
                np.testing.assert_array_equal(rewards,shaped.last_exact_reward.astype(shaped.last_original_reward.dtype))
                for a,b in zip(probe['physical'],[raw.data.qpos.numpy(),raw.data.qvel.numpy(),raw.data.ctrl.numpy(),raw.state.numpy()]):np.testing.assert_array_equal(a,b)
                assert np.all(shaped.potential[done]==0)
                for w in list(pending):
                    traces[w].append([start[w],probe['reward'][w],rewards[w],shaped.last_exact_reward[w],shaped.last_before[w],shaped.last_after[w]])
                    if not done[w]:continue
                    stopped=raw.stopped_q.numpy()[w]
                    length=float(np.mean([sim.fk_joints(float(stopped[ids[2*s]]),float(stopped[ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    rows[w]=dict(**cases[w],**{k:v for k,v in infos[w].items() if k!='terminal_observation'},final_mean_fk_leg_m=length)
                    pending.remove(w)
                if not pending:break
            assert not pending
            group=summary(rows);metrics=[]
            for row,values in zip(rows,traces):
                t=np.asarray(values);n=len(t);weights=gamma**np.arange(n)
                assert n==row['episode']['l'] and t[0,4]==0 and t[-1,5]==0
                np.testing.assert_allclose(t[:,0],np.arange(n)*.02,atol=1e-9,rtol=0)
                np.testing.assert_array_equal(t[1:,4],t[:-1,5])
                np.testing.assert_allclose(t[:,3]-t[:,1],gamma*t[:,5]-t[:,4],atol=1e-12,rtol=0)
                np.testing.assert_allclose(t[:,1].sum(),row['episode']['r'],atol=2e-5,rtol=0)
                exact_error=abs(float(weights@(t[:,3]-t[:,1])))
                delivered_error=abs(float(weights@(t[:,2]-t[:,1])))
                rounding_bound=float(weights@abs(t[:,2]-t[:,3]))
                assert exact_error<1e-10 and delivered_error<=rounding_bound+1e-10
                ppo_raw=t[:,1].astype(np.float32).astype(float)
                ppo_shaped=t[:,2].astype(np.float32).astype(float)
                ppo_error=abs(float(weights@(ppo_shaped-ppo_raw)))
                ppo_bound=float(weights@(abs(ppo_shaped-t[:,3])+abs(ppo_raw-t[:,1])))
                assert ppo_error<=ppo_bound+1e-10
                boundary_error=0.
                # All50 possible episode/rollout alignments, including nonterminal cuts.
                for phase in range(horizon):
                    starts=[0]+list(range(horizon-phase,n,horizon))+[n]
                    for lo,hi in zip(starts,starts[1:]):
                        expected=-t[lo,4]+gamma**(hi-lo)*t[hi-1,5]
                        got=float(gamma**np.arange(hi-lo)@(t[lo:hi,3]-t[lo:hi,1]))
                        boundary_error=max(boundary_error,abs(got-expected))
                assert boundary_error<1e-10
                metrics.append(dict(seed=row['seed'],exact_discount_error=exact_error,delivered_discount_error=delivered_error,
                    delivered_rounding_bound=rounding_bound,ppo_float32_discount_error=ppo_error,ppo_float32_rounding_bound=ppo_bound,
                    all_rollout_alignments_boundary_error=boundary_error,nonzero_potential_steps=int(np.count_nonzero(t[:,5]))))
            # Lifecycle fixture, not a new successful/failed full episode.
            shaped.potential.fill(-10);obs=env.reset();assert np.all(shaped.potential==0)
            env.step(np.zeros((len(cases),raw.action_dim),np.float32));probe_worlds+=len(cases)
            assert np.all(shaped.last_before==0) and np.all(shaped.last_after==0)
            np.testing.assert_array_equal(shaped.last_original_reward,shaped.last_exact_reward)
            if agent:
                assert before==(agent.num_timesteps,agent._n_updates)
                np.testing.assert_array_equal(stats[0],env.obs_rms.mean);np.testing.assert_array_equal(stats[1],env.obs_rms.var);assert stats[2]==env.obs_rms.count
            arrays=[np.asarray(t) for t in traces]
            np.savez_compressed(out/(method+'_trace.npz'),columns=np.array(COL),offsets=np.cumsum([0]+[len(t) for t in arrays]),trace=np.concatenate(arrays))
            atomic_json(out/(method+'.json'),dict(summary=group,runs=rows,metrics=metrics))
            results[method]=group;all_metrics.extend(metrics)
            print(method,group,'potential episodes',sum(m['nonzero_potential_steps']>0 for m in metrics),flush=True)
        finally:env.close()
    result=dict(verified=True,completed_episodes=128,reset_one_step_probe_worlds=probe_worlds,results=results,
        maximum_exact_discount_error=max(m['exact_discount_error'] for m in all_metrics),
        maximum_delivered_discount_error=max(m['delivered_discount_error'] for m in all_metrics),
        maximum_ppo_float32_discount_error=max(m['ppo_float32_discount_error'] for m in all_metrics),
        maximum_rollout_boundary_error=max(m['all_rollout_alignments_boundary_error'] for m in all_metrics),
        observations_and_physical_state_unchanged_by_reward_wrapper=True,policy_updates=0,normalization_updates=0,
        scope='Same32 public cases for four fixed controllers; freshGPU replay, not bitwise old-trajectory replication or learned benefit. Physical comparisons occur before/after reward processing within each same transition. Rollout identity is algebra, not a measured PPO critic guarantee.',
        source_sha256={name:digest(ROOT/'wheelleg_warp'/name) for name in ['check_reward_potential.py','reward_potential.py']})
    atomic_json(out/'result.json',result);print('PASS ONLINE POTENTIAL',result,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
