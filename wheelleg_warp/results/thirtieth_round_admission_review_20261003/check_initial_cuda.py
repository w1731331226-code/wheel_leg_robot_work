"""Check CUDA fresh-policy means against the frozen CPU mechanical audit."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from train_height_comparison import protocol,new_agent,METHODS
from audit_initial_actions import policy_mean,context,rms
from training_contract import digest
HERE=Path(__file__).resolve().parent

def run():
    p=protocol(HERE/'protocol_gpu_v3')
    config=json.loads((ROOT/p['initial_action_config']).read_text())
    bank_path=HERE.parent/'twentyninth_round_capped_support_20261003/state_bank.npz'
    assert digest(bank_path)==config['state_bank_sha256']
    with np.load(bank_path,allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    normalized=np.clip(bank['obs']/np.sqrt(1+1e-8),-10,10).astype(np.float32)
    indices=np.arange(len(normalized));rows=[]
    for method,mode in METHODS.items():
        dim=3 if mode=='diff3' else 6
        def factory():
            e=gym.Env();e.observation_space=gym.spaces.Box(-np.inf,np.inf,(38,),dtype=np.float32)
            e.action_space=gym.spaces.Box(-1,1,(dim,),dtype=np.float32);return e
        env=DummyVecEnv([factory for _ in range(100)])
        try:
            for seed in p['training_seeds']:
                cpu=policy_mean(bank,mode,seed);agent=new_agent(p,method,seed,env)
                assert agent.device.type=='cuda'
                with torch.no_grad():gpu=agent.policy.get_distribution(torch.as_tensor(normalized,device=agent.device)).distribution.mean.cpu().numpy()
                error=float(np.max(abs(cpu-gpu)));assert error<1e-6
                std=agent.policy.log_std.exp().detach().cpu().numpy()
                np.testing.assert_allclose(std,np.exp(p['initial_log_std'][method]),rtol=1e-6)
                a=rms(context(bank,mode,cpu,indices,bank['gaussian_z']),std)
                b=rms(context(bank,mode,gpu,indices,bank['gaussian_z']),std)
                relative=float(np.max(abs(a-b)/a));assert relative<1e-5
                rows.append(dict(method=method,seed=seed,mean_peak_difference=error,mechanical_rms_relative_difference=relative))
        finally:env.close()
    result=dict(passed=True,states=len(normalized),rows=rows,policy_device='cuda',calibration_refitted=False,
        conditional_initial_audit_only=True,full_distribution_equivalence_claimed=False,
        protocol_sha256=digest(HERE/'protocol_gpu_v3/protocol.json'),verifier_sha256=digest(__file__))
    (HERE/'initial_cuda_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS CUDA initialization:888states x3methods x3seeds; frozen sigma, mean and conditional mechanical RMS preserved')

if __name__=='__main__':torch.set_num_threads(1);run()
