"""One exact formal-size rollout/update on CUDA; weights are discarded."""
from pathlib import Path
import sys,json,tempfile
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
from train_height_comparison import protocol,CurriculumEnv,new_agent,equal
from training_contract import digest
HERE=Path(__file__).resolve().parent

def run():
    p=protocol(HERE/'protocol_gpu_v3');raw=CurriculumEnv(p,'diff3',1609,p['environments'])
    env=VecNormalize(VecCheckNan(raw,raise_exception=True),**p['normalization'])
    try:
        agent=new_agent(p,'M3',1609,env);before={k:v.clone() for k,v in agent.policy.state_dict().items()}
        agent.learn(total_timesteps=5000)
        assert agent.num_timesteps==5000 and agent.n_steps==50 and agent.n_envs==100 and agent._n_updates==10
        assert agent.device.type=='cuda' and all(v.device.type=='cuda' for v in agent.policy.parameters())
        assert any(not torch.equal(before[k],v) for k,v in agent.policy.state_dict().items())
        states=agent.policy.optimizer.state.values()
        assert all(state['exp_avg'].device.type==state['exp_avg_sq'].device.type=='cuda' for state in states)
        assert all(int(state['step'].item())==200 for state in agent.policy.optimizer.state.values())
        with tempfile.TemporaryDirectory(prefix='wheelleg_batch_',dir='/tmp') as tmp:
            prefix=Path(tmp)/'probe';agent.save(prefix);env.save(str(prefix)+'.pkl')
            loaded=PPO.load(str(prefix)+'.zip',device='cuda')
            equal(agent.policy.state_dict(),loaded.policy.state_dict());equal(agent.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict())
        result=dict(passed=True,environments=100,policy_steps=5000,rollout_steps_per_world=50,
            batch_size=250,training_epochs=10,optimizer_minibatch_updates=200,ppo_device='cuda',
            physics_device=str(raw.venv.data.qpos.device),weights_optimizer_restored_exactly=True,
            formal_training=False,probe_weights_promoted=False,protocol_sha256=digest(HERE/'protocol_gpu_v3/protocol.json'),verifier_sha256=digest(__file__))
        (HERE/'batch_check.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PASS CUDA exact formal batch:100worlds x50steps=5000, batch250,10epochs/200Adam updates; exact restore; probe discarded')
    finally:env.close()

if __name__=='__main__':torch.set_num_threads(1);run()
