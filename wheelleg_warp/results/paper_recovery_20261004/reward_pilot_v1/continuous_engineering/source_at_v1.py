"""Continuous registered reward pilot, with post-update checkpoints and cut audit."""
from pathlib import Path
import argparse,json,random,hashlib,sys
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from train_height_comparison import protocol,CurriculumEnv,new_agent,equal,evaluate,summary
from training_contract import digest,checkpoint_hashes
from dashboard.live_env import atomic_json
from reward_potential import FailurePotential
from smoke_reward_training import Ledger,weight_digest,check_agent

BASE=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
STUDY=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reward_pilot_v1'
EXTRA=['reward_pilot_train.py','reward_potential.py','smoke_reward_training.py']

def source_hashes():
    return {str(Path('wheelleg_warp')/name):digest(ROOT/'wheelleg_warp'/name) for name in EXTRA}

def freeze():
    base=protocol(BASE);proposal=json.loads((STUDY/'proposal.json').read_text());feasible=json.loads((STUDY/'feasibility.json').read_text())
    assert feasible['admitted'] and feasible['proposal_sha256']==digest(STUDY/'proposal.json')
    engineering=STUDY/'continuous_engineering/verification.json'
    report=json.loads(engineering.read_text());assert report['verified'] and report['source_sha256']==source_hashes()
    target=STUDY/'trainer_contract.json';assert not target.exists()
    atomic_json(target,dict(proposal_sha256=digest(STUDY/'proposal.json'),original_protocol_sha256=digest(BASE/'protocol.json'),
        source_sha256={**base['source_sha256'],**source_hashes()},engineering_sha256=digest(engineering),
        checkpoints='on_rollout_start after previous PPO train; final after learn returns; sampled/trained counters separate',
        lifecycle='one continuous200k learn, no physical/Phi/RNG restart, nonresumable without separately registered recovery',
        actor_critic_info='original38, no Phi input',policy_network='standardPPO 2x64',checkpoint_promoted=False))
    print('FROZEN continuous trainer',digest(target),flush=True)

class Continuous(Ledger):
    def __init__(self,potential,raw,directory,interval):
        super().__init__(potential);self.raw=raw;self.directory=directory;self.interval=interval
        self.length=np.zeros(raw.num_envs,dtype=int);self.trained=0;self.saved=[]
    def _on_step(self):
        super()._on_step();self.length+=1;self.length[self.locals['dones']]=0
        if self.num_timesteps%5000==0:
            atomic_json(self.directory/'progress.json',dict(status='running',sampled_steps=self.num_timesteps,trained_steps=self.trained))
        return True
    def _on_rollout_start(self):
        if not self.num_timesteps:return
        self.trained=self.num_timesteps
        if self.trained%self.interval==0:self.save()
    def save(self):
        steps=self.model.num_timesteps;rollout=self.model.n_steps*self.model.n_envs
        epochs=steps//rollout*self.model.n_epochs;adam=epochs*(rollout//self.model.batch_size)
        assert steps%rollout==0 and self.model._n_updates==epochs
        statecheck=check_agent(self.model,adam)
        if self.potential:
            expected=self.weight*self.potential.potential
            assert np.all(abs(self.delta-expected)<=self.rounding+1e-10)
            phi=self.potential.potential.copy();boundary=expected.tolist()
        else:phi=np.zeros(self.raw.num_envs);boundary=[0.]*self.raw.num_envs
        native=self.raw.venv
        before=[native.data.qpos.numpy(),native.data.qvel.numpy(),native.state.numpy(),native.k['state'].numpy(),phi.copy()]
        rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
        prefix=self.directory/f'step_{steps}'
        assert not prefix.with_suffix('.json').exists()
        self.model.save(prefix);self.training_env.save(str(prefix)+'.pkl')
        after=[native.data.qpos.numpy(),native.data.qvel.numpy(),native.state.numpy(),native.k['state'].numpy(),self.potential.potential.copy() if self.potential else phi.copy()]
        equal(before,after);equal(rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate()))
        record=dict(policy_steps=steps,trained_steps=steps,ppo_epochs=epochs,adam_updates=adam,adam=statecheck,
            completed_episodes=len(self.rows),nonzero_shaping_transitions=self.nonzero_shaping_transitions,
            ongoing_episode_policy_steps=self.length.tolist(),ongoing_phi=phi.tolist(),discounted_nonterminal_phi_boundary=boundary,
            potential_discount_errors=self.delta.tolist() if self.potential else [0.]*self.raw.num_envs,
            rounding_bounds=self.rounding.tolist() if self.potential else [0.]*self.raw.num_envs,
            physical_Phi_and_RNG_unchanged_by_save=True,checkpoint=checkpoint_hashes(prefix),checkpoint_promoted=False)
        atomic_json(prefix.with_suffix('.json'),record);self.saved.append(record);self.trained=steps
        atomic_json(self.directory/'progress.json',dict(status='running',sampled_steps=steps,trained_steps=steps,latest_checkpoint=str(prefix)))
        print('TRAINED CHECKPOINT',steps,'epochs',epochs,'Adam',adam,flush=True)

def run(arm,seed,engineering=False):
    config=protocol(BASE);proposal=json.loads((STUDY/'proposal.json').read_text())
    assert seed in proposal['training_seeds'] and arm in proposal['arms']
    if engineering:
        directory=STUDY/'continuous_engineering';n=10;target=12000;milestones=[4000,8000];interval=2000
    else:
        contract=json.loads((STUDY/'trainer_contract.json').read_text())
        assert contract['proposal_sha256']==digest(STUDY/'proposal.json') and contract['original_protocol_sha256']==digest(BASE/'protocol.json')
        assert all(digest(ROOT/name)==value for name,value in contract['source_sha256'].items())
        directory=STUDY/'runs'/arm/str(seed);n=proposal['environments'];target=proposal['policy_steps_per_run'];milestones=proposal['curriculum_milestones'];interval=proposal['checkpoint_every']
    directory.mkdir(parents=True,exist_ok=False)
    raw=CurriculumEnv(config,'diff3',seed,n,milestones=milestones)
    potential=FailurePotential(raw,gamma=config['ppo']['gamma'],beta=10.) if arm=='potential' else None
    env=VecNormalize(VecCheckNan(potential or raw,raise_exception=True),**config['normalization'])
    agent=new_agent(config,'M3',seed,env);initial=weight_digest(agent)
    h=hashlib.sha256()
    for value in [raw.venv.q0.numpy(),raw.venv.param.numpy(),raw.venv.k['gains'].numpy()]:h.update(value.tobytes())
    initial_world=h.hexdigest()
    peer=STUDY/'runs'/('original' if arm=='potential' else 'potential')/str(seed)/'initialization.json'
    if not engineering and peer.is_file():
        other=json.loads(peer.read_text());assert other['initial_weight_sha256']==initial and other['initial_world_sha256']==initial_world
    atomic_json(directory/'initialization.json',dict(arm=arm,seed=seed,environments=n,target_steps=target,initial_weight_sha256=initial,
        initial_world_sha256=initial_world,source_sha256=source_hashes(),engineering_only=engineering))
    cb=Continuous(potential,raw,directory,interval)
    try:
        agent.learn(total_timesteps=target,callback=cb)
        assert agent.num_timesteps==target and weight_digest(agent)!=initial
        cb.save();assert [r['policy_steps'] for r in cb.saved]==list(range(interval,target+1,interval))
        assert np.isfinite(env.obs_rms.mean).all() and np.isfinite(env.obs_rms.var).all()
        if arm=='potential':assert cb.nonzero_shaping_transitions>0
        atomic_json(directory/'episodes.json',dict(episodes=cb.rows,curriculum_transitions=raw.transition_log))
        evaluation=None
        if not engineering:
            rows=evaluate(proposal['development'],'diff3',agent,str(directory/f'step_{target}.pkl'))
            evaluation=dict(summary=summary(rows),runs=rows);atomic_json(directory/'development_final.json',evaluation)
        atomic_json(directory/'verification.json',dict(verified=True,policy_steps=target,ppo_epochs=agent._n_updates,
            checkpoints=[r['policy_steps'] for r in cb.saved],completed_episodes=len(cb.rows),nonzero_shaping_transitions=cb.nonzero_shaping_transitions,
            maximum_completed_potential_discount_error=cb.max_discount_error,initial_weight_sha256=initial,initial_world_sha256=initial_world,
            physical_resets_during_learning=0,partial_episode_discarded_midrun=False,final_nonterminal_cut=cb.saved[-1],
            original38_actor_value_inputs=True,standard_ppo_loss=True,source_sha256=source_hashes(),engineering_only=engineering,
            checkpoint_selection='fixed final target only',independent_gate_or_old_final_used=False))
        atomic_json(directory/'progress.json',dict(status='complete',sampled_steps=target,trained_steps=target))
        print('PASS CONTINUOUS',arm,seed,target,'steps; complete episodes',len(cb.rows),flush=True)
    except BaseException as error:
        atomic_json(directory/'interruption.json',dict(status='incomplete',sampled_steps=agent.num_timesteps,trained_checkpoint_steps=cb.trained,
            error_type=type(error).__name__,error=str(error),resumable_as_exact_trajectory=False,source_sha256=source_hashes()))
        raise
    finally:env.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--engineering',action='store_true')
    parser.add_argument('--arm',choices=['original','potential'],default='potential');parser.add_argument('--seed',type=int,default=1609)
    args=parser.parse_args();freeze() if args.freeze else run(args.arm,args.seed,args.engineering)
