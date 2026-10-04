"""Paired engineering-only CUDA PPO/restore check; never promotes a checkpoint."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from train_height_comparison import protocol,CurriculumEnv,new_agent,Episodes,equal
from training_contract import digest,checkpoint_hashes
from dashboard.live_env import atomic_json
from reward_potential import FailurePotential

P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'

def weight_digest(agent):
    h=hashlib.sha256()
    for name,tensor in sorted(agent.policy.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def check_agent(agent,expected_adam):
    assert agent.device.type=='cuda' and all(v.device.type=='cuda' and torch.isfinite(v).all() for v in agent.policy.parameters())
    states=agent.policy.optimizer.state_dict()['state'];assert states
    moments=set();steps=set()
    for state in states.values():
        for name,value in state.items():
            if isinstance(value,torch.Tensor):
                assert torch.isfinite(value).all()
                if name=='step':steps.add(int(value.item()))
                else:moments.add(value.device.type)
    assert moments=={'cuda'} and steps=={expected_adam}
    return dict(moment_devices=sorted(moments),adam_steps=sorted(steps))

class Ledger(Episodes):
    def __init__(self,potential=None):
        super().__init__();self.potential=potential;self.nonzero_shaping_transitions=0;self.max_discount_error=0.;self.completed_potential_episodes=0
        if potential:
            n=potential.num_envs;self.weight=np.ones(n);self.delta=np.zeros(n);self.rounding=np.zeros(n);self.previous=np.zeros(n)
    def _on_step(self):
        super()._on_step()
        if self.potential:
            p=self.potential;np.testing.assert_array_equal(self.previous,p.last_before)
            self.nonzero_shaping_transitions+=int(np.count_nonzero(p.last_shaping))
            delivered=self.locals['rewards'].astype(float);baseline=p.last_original_reward.astype(float)
            self.delta+=self.weight*(delivered-baseline)
            self.rounding+=self.weight*abs(delivered-p.last_exact_reward)
            self.weight*=p.gamma;self.previous=p.last_after.copy()
            done=self.locals['dones'];assert np.all(p.last_after[done]==0)
            for w in np.flatnonzero(done):
                error=abs(float(self.delta[w]));assert error<=self.rounding[w]+1e-10
                self.max_discount_error=max(self.max_discount_error,error);self.completed_potential_episodes+=1
                self.weight[w]=1.;self.delta[w]=0.;self.rounding[w]=0.
        return True

def make_env(config,arm,start=0):
    raw=CurriculumEnv(config,'diff3',1609,10,start=start,milestones=[4000,8000])
    potential=FailurePotential(raw,gamma=config['ppo']['gamma'],beta=10.) if arm=='potential' else None
    return raw,potential,VecCheckNan(potential or raw,raise_exception=True)

def run(out):
    config=protocol(P);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'registration.json',dict(role='Engineering check only, no method-performance gate or checkpoint promotion',
        arms=['original','potential'],method='M3',seed=1609,environments=10,policy_steps_per_arm=12000,
        first_segment_steps=6000,resumed_segment_steps=6000,curriculum_milestones=[4000,8000],ppo=config['ppo'],
        normalization=config['normalization'],source_sha256={n:digest(ROOT/'wheelleg_warp'/n) for n in ['smoke_reward_training.py','reward_potential.py']},
        protocol_sha256=digest(P/'protocol.json'),old_gate_or_final_replayed=False,
        restore_scope='Weights/Adam/RMS exact; physical episodes and potential restart, RNG trajectory not continued',
        required='Matched initial weights/world setup, real CUDA updates, finite actor/value/Adam, counters, original episode metrics, potential exercised, exact save/load'))
    initial=None;initial_world=None;reports={}
    for arm in ['original','potential']:
        directory=out/arm;directory.mkdir();raw,potential,checked=make_env(config,arm)
        env=VecNormalize(checked,**config['normalization']);agent=new_agent(config,'M3',1609,env)
        fingerprint=weight_digest(agent)
        world=dict(q0=raw.venv.q0.numpy(),param=raw.venv.param.numpy(),gains=raw.venv.k['gains'].numpy())
        if initial is None:initial=fingerprint;initial_world=world
        else:assert fingerprint==initial;equal(world,initial_world)
        cb=Ledger(potential);restored=None
        try:
            agent.learn(total_timesteps=6000,callback=cb)
            assert agent.num_timesteps==6000 and agent._n_updates==120 and cb.rows and weight_digest(agent)!=initial
            first_adam=check_agent(agent,240)
            prefix=directory/'step_6000';agent.save(prefix);env.save(str(prefix)+'.pkl')
            new_raw,new_potential,new_checked=make_env(config,arm,start=6000)
            restored=VecNormalize.load(str(prefix)+'.pkl',new_checked)
            loaded=PPO.load(str(prefix)+'.zip',env=restored,device='cuda')
            equal(agent.policy.state_dict(),loaded.policy.state_dict());equal(agent.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict())
            equal(env.obs_rms.mean,restored.obs_rms.mean);equal(env.obs_rms.var,restored.obs_rms.var);equal(env.obs_rms.count,restored.obs_rms.count)
            assert loaded.num_timesteps==6000 and loaded._n_updates==120
            cb2=Ledger(new_potential);loaded.learn(total_timesteps=6000,reset_num_timesteps=False,callback=cb2)
            assert loaded.num_timesteps==12000 and loaded._n_updates==240 and np.any(new_raw.stages==3)
            final_adam=check_agent(loaded,480)
            assert np.isfinite(restored.obs_rms.mean).all() and np.isfinite(restored.obs_rms.var).all()
            assert new_raw.venv.data.qpos.device.is_cuda
            final=directory/'step_12000';loaded.save(final);restored.save(str(final)+'.pkl')
            count=cb.nonzero_shaping_transitions+cb2.nonzero_shaping_transitions
            if arm=='potential':assert count>0 and cb.completed_potential_episodes+cb2.completed_potential_episodes>0
            report=dict(verified=True,policy_steps=12000,ppo_epochs=240,adam_updates=480,initial_weight_sha256=fingerprint,
                policy_changed=True,actor_value_device=str(loaded.device),physics_device=str(new_raw.venv.data.qpos.device),
                first_adam=first_adam,final_adam=final_adam,weights_optimizer_RMS_restored_exactly=True,
                physical_state_potential_RNG_trajectory_restored=False,nonzero_shaping_transitions=count,
                potential_complete_episodes=cb.completed_potential_episodes+cb2.completed_potential_episodes,
                maximum_potential_discount_error=max(cb.max_discount_error,cb2.max_discount_error),episodes=cb.rows+cb2.rows,
                curriculum_transitions=raw.transition_log+new_raw.transition_log,stages_after_resume=new_raw.stages.tolist(),
                checkpoint_6000=checkpoint_hashes(prefix),checkpoint_12000=checkpoint_hashes(final),checkpoint_promoted=False)
            atomic_json(directory/'verification.json',report);reports[arm]=report
            print('PASS',arm,'12000steps/240epochs/480Adam; episodes',len(report['episodes']),'shaping transitions',count,flush=True)
        finally:
            env.close()
            if restored is not None:restored.close()
    atomic_json(out/'result.json',dict(verified=True,total_engineering_policy_steps=24000,matched_initial_weight_sha256=initial,
        initial_worlds_matched=True,arms={a:{k:v for k,v in r.items() if k not in ['episodes','curriculum_transitions']} for a,r in reports.items()},
        formal_training=False,checkpoint_promoted=False,scope='One engineering seed and ten worlds, segmented restart. No independent evaluation or learning-benefit conclusion.',
        source_sha256={n:digest(ROOT/'wheelleg_warp'/n) for n in ['smoke_reward_training.py','reward_potential.py']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
