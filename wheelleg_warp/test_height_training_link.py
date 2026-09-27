"""多高度独立PPO链路的两次短更新；权重仅用于工程预检。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecCheckNan,VecNormalize
from benchmark_parallel import TimedPPO
from native.environment import NativeEnv
from native.terrain import bank_height_v3


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    config_path=ROOT/'wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21/training_config.json'
    config=json.loads(config_path.read_text());ppo=dict(config['ppo']);ppo['n_steps']=50
    torch.set_num_threads(1)
    raw=NativeEnv(n=128,stage=3,seed=1130000,bank_factory=bank_height_v3,height_conditioned=True)
    heights=raw.stand_heights.copy()
    assert min(heights)==.16 and max(heights)==.38 and len(np.unique(heights))>100
    env=VecNormalize(VecCheckNan(raw,raise_exception=True),**config['normalization'])
    model=TimedPPO('MlpPolicy',env,seed=1609,device='cpu',**ppo);model.timings=[]
    initial={name:value.clone() for name,value in model.policy.state_dict().items()}
    try:
        model.learn(total_timesteps=128*50*2)
        assert model.num_timesteps==12800 and len(model.timings)==2
        assert all(torch.isfinite(value).all() for value in model.policy.state_dict().values())
        assert any(not torch.equal(initial[name],value) for name,value in model.policy.state_dict().items())
        assert env.obs_rms.var[11]>1.e-4
        losses={key:float(value) for key,value in model.logger.name_to_value.items()
                if key.startswith('train/') and np.isscalar(value)}
        assert losses and all(np.isfinite(value) for value in losses.values())
        checkpoint=output/'pilot_policy';model.save(checkpoint);env.save(str(checkpoint)+'.pkl')
        restored=TimedPPO.load(str(checkpoint)+'.zip',device='cpu')
        assert restored.num_timesteps==model.num_timesteps
        source_names=('wheelleg_warp/test_height_training_link.py','wheelleg_warp/native/environment.py',
                      'wheelleg_warp/native/controller.py','wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py')
        hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
        result=dict(passed=True,role='engineering_preflight_only_not_a_candidate',steps=model.num_timesteps,updates=len(model.timings),
            environment_count=128,policy_n_steps=50,height_min_m=float(min(heights)),height_max_m=float(max(heights)),
            unique_initial_heights=int(len(np.unique(heights))),
            target_observation_mean=float(env.obs_rms.mean[11]),target_observation_variance=float(env.obs_rms.var[11]),
            endpoint_counts=dict(minimum=int(np.count_nonzero(heights==.16)),maximum=int(np.count_nonzero(heights==.38))),
            timings=model.timings,losses=losses,ppo=ppo,normalization=config['normalization'],source_sha256=hashes,
            training_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
        (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('PASS',result['steps'],result['updates'],output)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.output)
