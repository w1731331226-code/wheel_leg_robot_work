"""Two-update terrain-v1 admission; results never enter formal training."""
import hashlib,json,os,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecCheckNan,VecNormalize
from benchmark_parallel import TimedPPO
from dashboard.live_env import atomic_json as write
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from pretrain_yaw import summarize
from terrain_eval import evaluate_terrain
from train_terrain import protocol


if __name__=='__main__':
    source=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
    p=protocol(source);torch.set_num_threads(1);raw=TerrainEnv(1024,stage=1,seed=170000)
    env=VecNormalize.load(str(source/'bootstrap/policy.pkl'),VecCheckNan(raw,raise_exception=True));env.training=True;env.norm_reward=False
    model=TimedPPO.load(str(source/'bootstrap/policy.zip'),env=env,device='cpu');model.timings=[];start=model.num_timesteps
    initial={k:v.clone() for k,v in model.policy.state_dict().items()};started=time.perf_counter()
    try:
        model.learn(total_timesteps=102400,reset_num_timesteps=False)
        assert model.num_timesteps==start+102400 and len(model.timings)==2
        assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
        assert any(not torch.equal(initial[k],v) for k,v in model.policy.state_dict().items())
        prefix=output/'policy';model.save(prefix);env.save(str(prefix)+'.pkl')
        dev=[TerrainScenario(**x) for x in p['development_cases'][:8]];legacy=[TerrainScenario(**x) for x in p['legacy_regression_cases'][:4]]
        tr=evaluate_terrain(model,str(prefix)+'.pkl',dev);lr=evaluate_terrain(model,str(prefix)+'.pkl',legacy)
        result=dict(passed=True,policy_steps=model.num_timesteps,additional_steps=102400,updates=len(model.timings),seconds=time.perf_counter()-started,
            terrain=summarize(tr,[s.terrain_seed for s in dev]),legacy=summarize(lr,[s.terrain_seed for s in legacy]),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),formal_weights_used=False)
        write(output/'completed.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:env.close()
