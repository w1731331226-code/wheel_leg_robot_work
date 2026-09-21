"""Two-update terrain-v3 GPU admission; probe weights are never promoted."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import torch
from stable_baselines3.common.vec_env import VecCheckNan,VecNormalize
from benchmark_parallel import TimedPPO
from dashboard.live_env import atomic_json
from native.environment import NativeEnv
from native.terrain import TerrainScenario,bank_v3
from pretrain_yaw import summarize
from terrain_eval import evaluate_terrain


def cases(items):return [TerrainScenario(**x) for x in items]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'wheelleg_warp/results/terrain_v2_1024_20260922/round_002/step_11300000'
    protocol=json.loads((ROOT/'wheelleg_warp/results/terrain_v3_protocol_20260922/protocol.json').read_text())
    raw=NativeEnv(1024,stage=3,seed=390000,bank_factory=bank_v3);env=VecNormalize.load(str(source)+'.pkl',VecCheckNan(raw,raise_exception=True));env.training=True;env.norm_reward=False
    model=TimedPPO.load(str(source)+'.zip',env=env,device='cpu');model.timings=[];model.set_random_seed(930001);start=model.num_timesteps;started=time.perf_counter()
    try:
        model.learn(total_timesteps=102400,reset_num_timesteps=False)
        assert model.num_timesteps==start+102400 and len(model.timings)==2
        assert all(torch.isfinite(value).all() for value in model.policy.state_dict().values())
        checkpoint=output/'probe';model.save(checkpoint);env.save(str(checkpoint)+'.pkl')
        development=cases(protocol['development_cases']);legacy=cases(protocol['legacy_regression_cases'])
        terrain_rows=evaluate_terrain(model,str(checkpoint)+'.pkl',development);legacy_rows=evaluate_terrain(model,str(checkpoint)+'.pkl',legacy)
        result=dict(passed=True,additional_steps=102400,updates=len(model.timings),seconds=time.perf_counter()-started,probe_weights_promoted=False,
            terrain=summarize(terrain_rows,[x.terrain_seed for x in development]),legacy=summarize(legacy_rows,[x.terrain_seed for x in legacy]),
            finite_weights=True,source_checkpoint=str(source),checkpoint=str(checkpoint))
        atomic_json(output/'verified.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:
        atomic_json(output/'failed.json',dict(error=repr(exc),policy_steps=model.num_timesteps));raise
    finally:env.close()
