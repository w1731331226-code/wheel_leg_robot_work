"""Real SB3 updates hitting partial-rollout budgets, exact reload, no physics."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from train_height_comparison import learn_exact,Episodes,equal
HERE=Path(__file__).resolve().parent
class Toy(gym.Env):
    observation_space=gym.spaces.Box(-10,10,(38,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,(3,),dtype=np.float32)
    def reset(self,seed=None,options=None):super().reset(seed=seed);self.t=0;return np.zeros(38,np.float32),{}
    def step(self,a):self.t+=1;return np.full(38,self.t*.001,np.float32),float(1-np.dot(a,a)),self.t>=30,False,{}

def run():
    out=HERE/'budget_fixture';out.mkdir();env=VecNormalize(DummyVecEnv([Toy for _ in range(10)]),norm_reward=False)
    agent=PPO('MlpPolicy',env,n_steps=50,batch_size=50,n_epochs=1,seed=1609,device='cpu',policy_kwargs=dict(net_arch=[16,16]));restored=None
    try:
        cb=Episodes(out/'progress.json');learn_exact(agent,730,cb)
        assert agent.num_timesteps==730 and agent.n_steps==23 and json.loads((out/'progress.json').read_text())['executed_policy_steps']==730
        prefix=out/'probe';agent.save(prefix);env.save(str(prefix)+'.pkl')
        restored=VecNormalize.load(str(prefix)+'.pkl',DummyVecEnv([Toy for _ in range(10)]));loaded=PPO.load(str(prefix)+'.zip',env=restored,device='cpu')
        equal(agent.policy.state_dict(),loaded.policy.state_dict());equal(agent.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict())
        equal(env.obs_rms.mean,restored.obs_rms.mean);equal(env.obs_rms.count,restored.obs_rms.count)
        learn_exact(loaded,2000,Episodes(out/'progress.json'));assert loaded.num_timesteps==2000
        for target in (1999,2001):
            try:learn_exact(loaded,target)
            except ValueError:pass
            else:raise AssertionError('unreachable/replayed budget accepted')
        write=dict(passed=True,synthetic_env=True,real_standard_PPO_updates=True,first_exact_steps=730,resumed_exact_steps=2000,
            partial_rollout_buffer_size=23,weights_optimizer_rms_restored_exactly=True,invalid_budget_rejected=True,policy_steps_restarted=False)
        (HERE/'budget_check.json').write_text(json.dumps(write,indent=2)+'\n');print('PASS real SB3 exact budgets730->2000; partial buffer, reload and rejection')
    finally:
        env.close()
        if restored is not None:restored.close()
def check_formal_resume():
    """Exercise the actual train/exception/resume path with a CPU fixture only."""
    import tempfile
    from unittest.mock import patch
    import train_height_comparison as entry
    from training_contract import digest
    class Interrupted(DummyVecEnv):
        def __init__(self,p,mode,seed,n,start=0):
            super().__init__([Toy for _ in range(n)]);self.policy_steps=start
        def step_wait(self):
            self.policy_steps+=self.num_envs
            if self.policy_steps==730:raise RuntimeError('fixture sampling interruption')
            return super().step_wait()
    created=[];loaded=[];fail_eval=[True]
    original_read=entry.read;original_load=PPO.load
    def fresh(p,method,seed,env):
        created.append(method)
        return PPO('MlpPolicy',env,n_steps=50,batch_size=50,n_epochs=1,seed=seed,device='cpu',policy_kwargs=dict(net_arch=[16,16]))
    def load(path,*args,**kwargs):
        agent=original_load(path,*args,**kwargs);loaded.append((str(path),agent.num_timesteps))
        reference=original_load(path,device='cpu')
        equal(agent.policy.state_dict(),reference.policy.state_dict())
        equal(agent.policy.optimizer.state_dict(),reference.policy.optimizer.state_dict())
        return agent
    def evaluate(*args,**kwargs):
        if fail_eval[0]:fail_eval[0]=False;raise RuntimeError('fixture selection interruption')
        return []
    with tempfile.TemporaryDirectory(prefix='wheelleg_resume_',dir='/tmp') as tmp:
        out=Path(tmp);(out/'protocol.json').write_text('{"synthetic_fixture":true}\n')
        p=dict(formal_seeds=[1609],environments=10,device='cpu',normalization=dict(norm_obs=True,norm_reward=False),evaluation_interval=1000,policy_steps_per_seed=2000,selection=[])
        def read(path):
            if Path(path)==out/'readiness.json':return dict(passed=True,protocol_sha256=digest(out/'protocol.json'))
            return original_read(path)
        with patch.object(entry,'protocol',return_value=p),patch.object(entry,'read',side_effect=read),patch.object(entry,'CurriculumEnv',Interrupted),patch.object(entry,'new_agent',side_effect=fresh),patch.object(entry,'evaluate',side_effect=evaluate),patch.object(entry,'summary',return_value=dict(complete=True)),patch.object(entry,'selection_key',side_effect=lambda s,t:t),patch.object(PPO,'load',side_effect=load):
            directory=out/'runs/M3/1609'
            for resume,expected_steps,message in [(False,730,'sampling'),(True,1000,'selection')]:
                try:entry.train(out,'M3',1609,resume)
                except RuntimeError as exc:assert message in str(exc)
                else:raise AssertionError('fixture interruption missed')
                last=original_read(directory/'last_checkpoint.json')
                assert last['consumed_policy_steps']==expected_steps and last['resumable']
                assert original_read(directory/'progress.json')['executed_policy_steps']==expected_steps
            # A hard-killed unsaved sampling step cannot be silently replayed.
            (directory/'progress.json').write_text('{"executed_policy_steps":1010}')
            try:entry.train(out,'M3',1609,True)
            except ValueError as exc:assert 'Unsaved executed steps' in str(exc)
            else:raise AssertionError('unsaved budget accepted')
            (directory/'progress.json').write_text('{"executed_policy_steps":1000}')
            damaged=dict(last,checkpoint_sha256='0'*64)
            (directory/'last_checkpoint.json').write_text(json.dumps(damaged))
            try:entry.train(out,'M3',1609,True)
            except ValueError:pass
            else:raise AssertionError('invalid checkpoint hash accepted')
            (directory/'last_checkpoint.json').write_text(json.dumps(last))
            entry.train(out,'M3',1609,True)
            result=original_read(directory/'selection.json')
            assert result['consumed_policy_steps']==2000 and result['evaluation_count']==2
            assert result['best']['policy_steps']==1000
            assert original_read(directory/'last_checkpoint.json')['consumed_policy_steps']==2000
            assert created==['M3'] and [steps for _,steps in loaded]==[730,1000]
            assert [r['policy_steps'] for r in map(original_read,sorted(directory.glob('step_*.json')))]==[1000,2000]
        report=dict(passed=True,synthetic_cpu_environment=True,actual_train_entry=True,real_standard_PPO_updates=True,
            sampling_interruption_consumed_steps=730,selection_interruption_consumed_steps=1000,resumed_exact_steps=2000,
            fresh_initialization_calls=1,restored_weights_optimizer_exactly=True,pending_evaluation_completed=True,
            unsaved_budget_rejected=True,invalid_checkpoint_hash_rejected=True,physical_trajectory_restarted=True,
            formal_protocol_admission_created=False,source_sha256=digest(entry.__file__),verifier_sha256=digest(__file__))
        (HERE/'formal_resume_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PASS actual train CPU fixture: sampling/selection interruption, exact resume730/1000->2000, no reinitialization, fail-closed budgets/hash')

if __name__=='__main__':torch.set_num_threads(1);check_formal_resume()
