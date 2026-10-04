"""Bounded three-seed route/action pilot; no old-gate or final-set selection."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import torch
from route_pilot_env import ARMS,training_env,agent,evaluate,self_check
from train_height_comparison import protocol,entry,equal
from native.terrain import sample_height_terrain_115
from reward_pilot_train import Continuous
from smoke_reward_training import weight_digest,check_agent
from training_contract import digest
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/route_pilot_v1'
FILES=['route_pilot.py','route_pilot_env.py','route_state.py','probe_packet_odometry.py',
       'probe_route_feedback.py','reward_pilot_train.py','reward_potential.py','smoke_reward_training.py','audit_initial_actions.py','test_route_state.py']


def spec():
    d=json.loads((OUT/'proposal.json').read_text())
    assert all(digest(ROOT/n)==v for n,v in d['source_sha256'].items())
    assert d['base_protocol_sha256']==digest(BASE/'protocol.json')
    return d


def register():
    p=protocol(BASE);OUT.mkdir(parents=True,exist_ok=False)
    prior=json.loads((OUT.parent/'route_feedback_v1/registration.json').read_text())
    d=dict(version='route-action-pilot-v1',arms=ARMS,training_seeds=[1609,1610,1611],environments=100,
        policy_steps_per_run=200000,total_training_budget=2400000,curriculum_milestones=p['curriculum_milestones'],checkpoint_every=20000,
        training_banks='Reuse the exact frozen100-world/stage/seed banks; same episode-fixed heights. No geometry expansion in this mechanism pilot.',
        ppo=p['ppo'],normalization=p['normalization'],initial_log_std={**p['initial_log_std'],'L2':p['initial_log_std']['M3'][:2]},
        regular=[entry(i,sample_height_terrain_115(i,3,'development')) for i in range(4700000,4700096)],controlled=prior['sets']['controlled'],
        development_scope='Fresh regular96; reused public controlled40. Both development, not an independent/final test. Every final200k model evaluated once; no checkpoint/seed/case selection.',
        classical={'B0':dict(candidate=dict(name='B0',kp=0.,kd=0.,roll_gain=0.),arm=0),
                   'B1':dict(candidate=prior['candidate'],arm=0),'B1-route':dict(candidate=prior['candidate'],arm=1)},
        evaluation_budget_episodes=(3+12)*(96+40),
        hypothesis='Primary paired information effect M3-route minus M3-zero at regular96; action-space contrasts V6/L2 are whole parameterization effects, not isolated physical-prior causality.',
        information_gate=dict(mean_regular_success_gain=.05,positive_seed_pairs_at_least=2,controlled_success_nondegradation=True,
            physical_design_nondegradation='No increase in mean failure counts vsM3-zero separately by panel; disclose individual seed/case swaps'),
        formal_expansion_gate=dict(candidates=['M3-route','L2-route'],
            rule='For each of3 seeds preserve every success/physical/design pass of B0,B1,B1-route in both panels, all trajectories complete; mean controlled success gain over the strongest of all3 fixed classical counts>=.05; regular Jpsi no worse than B1+.05deg; V6/zero contrasts reported independently. Candidate precedence M3 then L2, no five-seed extension merely for information-gate success.'),
        initialization_scope='M3-zero/route identical39D network seed/std; V6 inherits original calibration. L2 truncates the two leg std coordinates and removes wheel authority; effort covariance/rank/normalization histories differ and are reported, no full equivalence claim.',
        lifecycle='One continuous200k learn/run; save after actual PPO updates. Do not restart physics, route memory or RNG midrun. Interrupted run preserved and nonresumable as an exact trajectory.',
        failure_rule='No silent retry, budget/gain sweep or renewed old gates. Runtime/source error stops queue and preserves all consumption. Negative pilot does not justify5seed expansion.',
        scope='Pure simulation, original actuator/physical/design/task gates, fixed nominal7kg and39 causal inputs; no new PPO or localization algorithm claim.',
        old_gate_or_final_used=False,base_protocol_sha256=digest(BASE/'protocol.json'),
        source_sha256={**p['source_sha256'],**{f'wheelleg_warp/{n}':digest(ROOT/'wheelleg_warp'/n) for n in FILES}})
    atomic_json(OUT/'proposal.json',d);print('REGISTERED4arms x3seeds x200k =2.4M, 2040development evaluations',flush=True)


class RouteLedger(Continuous):
    def __init__(self,raw,route,directory,interval,leg_only=False):
        super().__init__(None,raw,directory,interval);self.route=route;self.leg_only=leg_only
    def _on_step(self):
        value=super()._on_step()
        if self.leg_only:
            assert np.all(self.raw.venv.k['state'].numpy()[:,18]==0)
            assert np.all(self.raw.venv.diag.numpy()[:,10:12]==0)
        return value
    def save(self):
        before=[self.route.odometry.y.copy(),self.route.odometry.last_velocity.copy(),self.route.clock.copy()]
        super().save()
        equal(before,[self.route.odometry.y,self.route.odometry.last_velocity,self.route.clock])
        record=self.saved[-1];record['route_memory_unchanged_by_save']=True
        record['ongoing_route_y_m']=before[0].tolist();record['ongoing_route_clock_s']=before[2].tolist()
        atomic_json((self.directory/f'step_{self.model.num_timesteps}').with_suffix('.json'),record)


def scale_audit(p):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from audit_initial_actions import statistics,context
    path=ROOT/'wheelleg_warp/results/twentyninth_round_capped_support_20261003/state_bank.npz'
    initial=json.loads((ROOT/'wheelleg_warp/results/twentyninth_round_capped_support_20261003/initial_action_config.json').read_text())
    assert digest(path)==initial['state_bank_sha256']
    with np.load(path,allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    report={};normalized=np.c_[np.clip(bank['obs']/np.sqrt(1+1e-8),-10,10),np.zeros(len(bank['obs']))].astype(np.float32)
    for arm,a in ARMS.items():
        def factory():
            e=gym.Env();e.observation_space=gym.spaces.Box(-np.inf,np.inf,(39,),dtype=np.float32);e.action_space=gym.spaces.Box(-1,1,(a['dimensions'],),dtype=np.float32);return e
        vec=DummyVecEnv([factory])
        try:
            model=agent(p,arm,1609,vec)
            with torch.no_grad():mean=model.policy.get_distribution(torch.as_tensor(normalized,device=model.device)).distribution.mean.cpu().numpy()
            c=context(bank,a['native'],mean,np.arange(len(mean)),bank['gaussian_z'])
            if a['dimensions']==2:c['noise']=c['noise'][:,:2];c['matrix']=c['matrix'][:,:,:2]
            std=np.exp(np.asarray(spec()['initial_log_std'][a['initial']]))
            report[arm]=dict(dimensions=a['dimensions'],log_std=np.log(std).tolist(),statistics=statistics(c,std))
        finally:vec.close()
    atomic_json(OUT/'initial_effort_audit.json',dict(state_bank_sha256=digest(path),reference_states=888,gaussian_samples=256,arms=report,
        scope='Static reference-state linear mapping, slew and projection diagnostic inherited from original calibration; not new training-state distributions, physics rollouts or equality of covariance/reachable sets. Route feature zero at common diagnostic states. No recalibration by performance.'))


def run_one(arm,seed,engineering=False):
    d=spec();p=protocol(BASE);torch.set_num_threads(1)
    n=10 if engineering else 100;target=1000 if engineering else 200000;interval=500 if engineering else 20000
    if not engineering:
        contract=json.loads((OUT/'trainer_contract.json').read_text());assert contract['proposal_sha256']==digest(OUT/'proposal.json')
    directory=OUT/('engineering' if engineering else 'runs')/arm/str(seed);directory.mkdir(parents=True,exist_ok=False)
    raw,route,env=training_env(p,arm,seed,n);model=agent(p,arm,seed,env)
    initial=weight_digest(model);world=hashlib.sha256()
    for value in [raw.venv.q0.numpy(),raw.venv.param.numpy(),raw.venv.k['gains'].numpy()]:world.update(value.tobytes())
    peer=directory.parent.parent/('M3-zero' if arm=='M3-route' else 'M3-route')/str(seed)/'initialization.json'
    if arm in ('M3-zero','M3-route') and peer.exists():
        other=json.loads(peer.read_text());assert other['weight_sha256']==initial and other['world_sha256']==world.hexdigest()
    atomic_json(directory/'initialization.json',dict(arm=arm,seed=seed,weight_sha256=initial,world_sha256=world.hexdigest(),dimensions=ARMS[arm]['dimensions'],observations=39,engineering_only=engineering))
    cb=RouteLedger(raw,route,directory,interval,ARMS[arm]['dimensions']==2)
    try:
        started=time.perf_counter();model.learn(total_timesteps=target,callback=cb);cb.save();elapsed=time.perf_counter()-started
        assert model.num_timesteps==target and weight_digest(model)!=initial
        assert [r['policy_steps'] for r in cb.saved]==list(range(interval,target+1,interval))
        assert model.observation_space.shape==(39,) and model.action_space.shape==(ARMS[arm]['dimensions'],)
        atomic_json(directory/'episodes.json',dict(episodes=cb.rows,curriculum_transitions=raw.transition_log))
        if not engineering:
            for panel in ['regular','controlled']:
                result=evaluate(d[panel],arm,model,directory/f'step_{target}.pkl');atomic_json(directory/(panel+'_final.json'),result)
        final=cb.saved[-1]
        atomic_json(directory/'verification.json',dict(verified=True,policy_steps=target,ppo_epochs=model._n_updates,adam_updates=final['adam_updates'],
            dimensions=ARMS[arm]['dimensions'],observations=39,source_sha256=d['source_sha256'],initial_weight_sha256=initial,
            checkpoints=[r['policy_steps'] for r in cb.saved],completed_episodes=len(cb.rows),forced_physical_or_route_resets_midrun=0,
            final_route_memory=final['ongoing_route_y_m'],engineering_only=engineering,weights_promoted=False,checkpoint_selection='fixed final target',old_gate_or_final_used=False))
        report=json.loads((directory/'verification.json').read_text());report['continuous_learn_and_checkpoint_wall_s']=elapsed
        atomic_json(directory/'verification.json',report)
        atomic_json(directory/'progress.json',dict(status='complete',sampled_steps=target,trained_steps=target))
        print('COMPLETED',arm,seed,target,'epochs',model._n_updates,'Adam',final['adam_updates'],flush=True)
    except BaseException as e:
        atomic_json(directory/'interruption.json',dict(sampled_steps=model.num_timesteps,confirmed_trained_steps=cb.trained,last_saved_steps=cb.saved[-1]['policy_steps'] if cb.saved else 0,error=str(e),silently_resumable=False));raise
    finally:env.close()


def engineering():
    d=spec();p=protocol(BASE);self_check();scale_audit(p)
    for arm in ARMS:run_one(arm,1609,True)
    records={arm:json.loads((OUT/'engineering'/arm/'1609/verification.json').read_text()) for arm in ARMS}
    assert all(r['verified'] and r['policy_steps']==1000 and r['ppo_epochs']==20 and r['adam_updates']==40 for r in records.values())
    atomic_json(OUT/'engineering_verification.json',dict(verified=True,records=records,total_training_steps=4000,source_sha256=d['source_sha256'],scope='Four CUDA pipelines/continuous checkpoints and leg-only absence, not performance.'))


def classical():
    d=spec();folder=OUT/'classical';folder.mkdir(exist_ok=False);records={}
    for label,c in d['classical'].items():
        for panel in ['regular','controlled']:
            result=evaluate(d[panel],'M3-route',classical=c);atomic_json(folder/(label+'_'+panel+'.json'),result)
            records[label+'_'+panel]={k:v for k,v in result.items() if k!='runs'};print('CLASSICAL',label,panel,records[label+'_'+panel],flush=True)
    # A 5pp gain is required only on controlled40, not ceiling-limited regular96.
    best=max(records[k+'_controlled']['summary']['success_count'] for k in d['classical'])
    feasible=40-best>=2 and all(records['B1_'+p]['summary']['complete'] for p in ['regular','controlled'])
    atomic_json(OUT/'classical_feasibility.json',dict(records=records,controlled_maximum_gain_pp=100*(40-best)/40,
        formal_gate_numerically_attainable=feasible,proposal_sha256=digest(OUT/'proposal.json'),scope='Finite development headroom only, not proof an RL controller can reach the gate.'))


def freeze():
    d=spec();e=json.loads((OUT/'engineering_verification.json').read_text());f=json.loads((OUT/'classical_feasibility.json').read_text())
    assert e['verified'] and e['source_sha256']==d['source_sha256'] and f['formal_gate_numerically_attainable']
    target=OUT/'trainer_contract.json';assert not target.exists()
    atomic_json(target,dict(proposal_sha256=digest(OUT/'proposal.json'),engineering_sha256=digest(OUT/'engineering_verification.json'),
        scale_sha256=digest(OUT/'initial_effort_audit.json'),classical_sha256=digest(OUT/'classical_feasibility.json'),source_sha256=d['source_sha256'],
        lifecycle='Continuous200k, post-update checkpoints, route memory unchanged by saving; no silent resume',selection='Final200k only'))
    print('FROZEN ROUTE PILOT',flush=True)


def queue():
    d=spec();assert (OUT/'trainer_contract.json').exists();completed=[]
    try:
        for seed in d['training_seeds']:
            for arm in ARMS:
                atomic_json(OUT/'queue_progress.json',dict(status='running',completed_runs=completed,pending_job=dict(arm=arm,seed=seed)))
                run_one(arm,seed);completed.append(dict(arm=arm,seed=seed))
        atomic_json(OUT/'queue_progress.json',dict(status='complete',completed_runs=completed,training_steps=2400000))
    except BaseException as e:
        atomic_json(OUT/'queue_interruption.json',dict(completed_runs=completed,error=str(e),silently_resumable=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','engineering','classical','freeze','queue']);args=parser.parse_args()
    globals()[args.command]()
