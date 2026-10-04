"""Read-only causal position estimate from the existing delayed actor packet.

Initial route origin is known. Unknown initial lateral offsets cannot be
recovered by velocity integration; this is not a new localization algorithm.
"""
from pathlib import Path
import argparse, json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
from train_height_comparison import raw_env, summary, b1_action
from training_contract import digest
from smoke_reward_training import weight_digest
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'wheelleg_warp/results/paper_recovery_20261004/yaw_sector_v1'
COL=['time_s','estimated_y_m','true_y_m','packet_yaw_rad','packet_vx_m_s','packet_vy_m_s','terminal']


def lateral_velocity(packet):
    if packet.ndim!=2 or packet.shape[1]!=38 or not np.isfinite(packet).all():
        raise ValueError('Expected finite request_state_v1 packets')
    return np.sin(packet[:,2])*packet[:,6]+np.cos(packet[:,2])*packet[:,7]


class PacketOdometry:
    def __init__(self, packet):
        self.y=np.zeros(len(packet),np.float64)
        self.last_velocity=lateral_velocity(packet)
    def update(self, packet, dt):
        dt=np.asarray(dt,dtype=np.float64)
        if dt.shape!=self.y.shape or not np.isfinite(dt).all() or np.any(dt<=0):
            raise ValueError('Expected one positive elapsed interval per world')
        velocity=lateral_velocity(packet)
        self.y+=.5*(self.last_velocity+velocity)*dt
        self.last_velocity=velocity
        return self.y.copy()


def self_check():
    packet=np.zeros((2,38));packet[:,2]=[0,np.pi/2];packet[:,6]=[2,1];packet[:,7]=[1,0]
    state=PacketOdometry(packet)
    np.testing.assert_allclose(state.update(packet,np.array([.02,.02])),[.02,.02],atol=1e-15)
    packet[:,6]*=-1;packet[:,7]*=-1
    np.testing.assert_allclose(state.update(packet,np.array([.02,.02])),[.02,.02],atol=1e-15)
    # Same velocity packets cannot reveal an unobserved initial 120mm shift.
    assert abs(state.y[0]-(state.y[0]+.12))>.119


def collect(cases,model,candidate,out,label):
    raw=raw_env(cases,'diff3');env=raw;agent=None
    if model:
        prefix=Path(model['prefix'])
        env=VecNormalize.load(str(prefix)+'.pkl',VecCheckNan(raw,raise_exception=True))
        env.training=False;env.norm_reward=False;agent=PPO.load(str(prefix)+'.zip',device='cuda')
        before=(agent.num_timesteps,agent._n_updates,weight_digest(agent))
        rms=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count)
    pending=np.ones(len(cases),bool);rows=[None]*len(cases);traces=[[] for _ in cases]
    try:
        obs=env.reset();packet=env.unnormalize_obs(obs) if model else obs
        state=PacketOdometry(packet);clock=np.zeros(len(cases));ids=raw.ids.numpy()
        initial=raw.data.qpos.numpy()
        assert np.all(initial[:,1]==0)
        for w in range(len(cases)):traces[w].append([0.,0.,float(initial[w,1]),*packet[w,[2,6,7]].tolist(),0.])
        deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
        for _ in range(deadline):
            action=agent.predict(obs,deterministic=True)[0] if model else np.stack([b1_action(o,candidate) for o in obs])
            obs,_,done,infos=env.step(action)
            packet=env.unnormalize_obs(obs) if model else obs.copy()
            dt=np.full(len(cases),.02)
            for w in np.flatnonzero(done & pending):
                terminal=np.asarray(infos[w]['terminal_observation'])[None,:]
                packet[w]=(env.unnormalize_obs(terminal) if model else terminal)[0]
                dt[w]=infos[w]['duration_s']-clock[w]
            estimate=state.update(packet,dt);clock+=dt
            q=raw.data.qpos.numpy();stopped=raw.stopped_q.numpy() if done.any() else None
            for w in np.flatnonzero(pending):
                position=stopped[w] if done[w] else q[w]
                traces[w].append([clock[w],estimate[w],float(position[1]),*packet[w,[2,6,7]].tolist(),float(done[w])])
                if done[w]:
                    length=float(np.mean([sim.fk_joints(float(position[ids[2*s]]),float(position[ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    rows[w]=dict(**cases[w],**{k:v for k,v in infos[w].items() if k!='terminal_observation'},final_mean_fk_leg_m=length)
                    pending[w]=False
            if not pending.any():break
        assert not pending.any()
        if model:
            assert before==(agent.num_timesteps,agent._n_updates,weight_digest(agent))
            np.testing.assert_array_equal(rms[0],env.obs_rms.mean);np.testing.assert_array_equal(rms[1],env.obs_rms.var);assert rms[2]==env.obs_rms.count
        arrays=[np.asarray(t) for t in traces]
        np.savez_compressed(out/(label+'_trace.npz'),columns=np.array(COL),offsets=np.cumsum([0]+[len(t) for t in arrays]),trace=np.concatenate(arrays))
        errors=[float(abs(t[t[:,6]==0,1]-t[t[:,6]==0,2]).max()) for t in arrays]
        result=dict(summary=summary(rows),runs=rows,max_nonterminal_y_error_m=max(errors),per_case_max_error_m=errors)
        atomic_json(out/(label+'.json'),result)
        return {k:v for k,v in result.items() if k!='runs'}
    finally:env.close()


def run(out):
    self_check();prior=json.loads((PRIOR/'registration.json').read_text())
    assert all(digest(ROOT/n)==v for n,v in prior['source_sha256'].items())
    for model in prior['models']:
        prefix=Path(model['prefix']);record=model['checkpoint']
        assert digest(prefix.with_suffix('.zip'))==record['checkpoint_sha256'] and digest(prefix.with_suffix('.pkl'))==record['normalization_sha256']
    out.mkdir(parents=True,exist_ok=False)
    reg=dict(version='causal-packet-odometry-v1',cases=prior['cases'],models=prior['models'],candidate=prior['classical'],
        columns=COL,budget_episodes=128,training_steps=0,prior_registration_sha256=digest(PRIOR/'registration.json'),
        role='Known-zero-origin development sensor/memory audit; estimator never enters control or policy.',
        formula='v_y_world=sin(packet_yaw)*packet_vx+cos(packet_yaw)*packet_vy; trapezoidal integration using actor period20ms, terminal partial dt only after episode end',
        inputs='Existing actor packet only, including normalized/clipped packet inversion for learned models; no hidden delay value, contact, parameter or position input.',
        accuracy_budget_m=.005,accuracy_rule='Every scored nonterminal sample in all128 episodes within5mm. Engineering budget only, not sufficient for collision/contact or a learning-admission gate.',
        limits='Ideal simulated body velocities already in original observation; not wheel encoder localization or sim-to-real proof; unknown initial offsets remain aliased. No explicit delay compensation.',
        old_gate_or_final_used=False,source_sha256={**prior['source_sha256'],'wheelleg_warp/probe_packet_odometry.py':digest(__file__)})
    atomic_json(out/'registration.json',reg);results={};records=[]
    jobs=[('B1',None)]+[(str(m['seed']),m) for m in reg['models']]
    try:
        for label,model in jobs:
            atomic_json(out/'progress.json',dict(status='running',completed_episodes=len(records)*32,pending_job=label))
            assert all(digest(ROOT/n)==v for n,v in reg['source_sha256'].items())
            results[label]=collect(reg['cases'],model,reg['candidate'],out,label)
            records.append(dict(label=label,result_sha256=digest(out/(label+'.json')),trace_sha256=digest(out/(label+'_trace.npz'))))
            atomic_json(out/'completed_jobs.json',dict(records=records,completed_episodes=len(records)*32))
            print('COMPLETED',len(records)*32,label,results[label]['max_nonterminal_y_error_m'],flush=True)
        atomic_json(out/'result.json',dict(completed_episodes=128,training_steps=0,results=results,
            accuracy_budget_passed=all(r['max_nonterminal_y_error_m']<=reg['accuracy_budget_m'] for r in results.values()),registration_sha256=digest(out/'registration.json')))
        atomic_json(out/'progress.json',dict(status='complete',completed_episodes=128))
    except BaseException as e:
        atomic_json(out/'interruption.json',dict(completed_episodes=len(records)*32,error=str(e),silently_resumable=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:self_check();print('PASS frame rotation, trapezoidal integration and unknown-origin limitation')
    else:
        if args.output is None:parser.error('--output required')
        run(args.output)
