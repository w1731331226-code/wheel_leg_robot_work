"""固定预算的height-v1公开开发能力探针；不晋升权重。"""
from dataclasses import asdict,replace
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
from native.terrain import HeightTerrainScenario,bank_height_v3,sample_height_terrain_v3


def training_cases():
    cases=[sample_height_terrain_v3(1130000+i,stage=3,split='train') for i in range(128)]
    for i in range(32):
        rng=np.random.default_rng(np.random.SeedSequence([1130000+i,170091]))
        height=float(rng.uniform(.016,.02));left=i%2==0
        cases[i]=replace(cases[i],terrain='legacy',height_l=height if left else 0.,height_r=0. if left else height,
                         grade_deg=0.,roughness_m=0.,step_height_m=0.,transition_run_m=0.,lateral_margin_m=0.)
    assert len(cases)==128 and len({s.stand_height_m for s in cases})>100
    return cases


def development(name):
    path=ROOT/f'wheelleg_warp/results/height_scope_recheck_20260928/{name}/verification.json'
    return path,[HeightTerrainScenario(**row['scenario']) for row in json.loads(path.read_text())['rows']]


def evaluate(model,normalization,cases):
    raw=NativeEnv(n=len(cases),scenario=cases,bank_factory=bank_height_v3,height_conditioned=True)
    env=VecNormalize.load(str(normalization),raw);env.training=False;env.norm_reward=False
    stats=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count)
    try:
        obs=env.reset();found=[None]*len(cases)
        for _ in range(700):
            action=model.predict(obs,deterministic=True)[0]
            obs,_,done,infos=env.step(action)
            for i in np.flatnonzero(done):
                if found[i] is None:
                    info=infos[i]
                    found[i]=dict(scenario=asdict(cases[i]),success=bool(info['success']),reason=info['reason'],
                        peak_deg=info['peak_deg'],relative_peak_deg=info['relative_peak_deg'],
                        height_rmse_m=info['height_rmse_m'],velocity_rmse=info['velocity_rmse'],
                        stop_distance_m=info['stop_distance_m'],tail_speed_m_s=info['tail_speed_m_s'],
                        terrain_evidence_passed=info['terrain_evidence_passed'])
            if all(row is not None for row in found):break
        assert all(row is not None for row in found)
        np.testing.assert_array_equal(stats[0],env.obs_rms.mean)
        np.testing.assert_array_equal(stats[1],env.obs_rms.var)
        assert stats[2]==env.obs_rms.count
        return found
    finally:env.close()


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    config_path=ROOT/'wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21/training_config.json'
    config=json.loads(config_path.read_text());ppo=dict(config['ppo']);ppo['n_steps']=50
    dev_path,dev=development('stratified_run1');hard_path,hard=development('boundary_run1')
    cases=training_cases();assert len({s.stand_height_m for s in cases})>100
    assert not ({s.terrain_seed for s in cases}&{s.terrain_seed for s in dev if s.terrain_seed})
    names=('wheelleg_warp/train_height_capability.py','wheelleg_warp/native/environment.py',
           'wheelleg_warp/native/controller.py','wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py')
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    protocol=dict(role='public_development_capability_pilot_not_formal_training',worlds=128,seed=1130000,
        ppo_seed=1609,total_policy_steps=102400,checkpoints=(0,51200,102400),
        training_high_bump_worlds=32,height_range_m=(.16,.38),residual_scale=1.,
        ppo=ppo,normalization=config['normalization'],training_cases=[asdict(s) for s in cases],
        development=dict(stratified=str(dev_path),boundary=str(hard_path)),
        source_sha256=hashes,training_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        development_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in (('stratified',dev_path),('boundary',hard_path))},
        selection='no promotion; repeat/reorder only if boundary >=24/48 and stratified >=118/120',
        final_holdout_opened=False)
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    torch.set_num_threads(1)
    raw=NativeEnv(n=128,scenario=cases,bank_factory=bank_height_v3,height_conditioned=True)
    env=VecNormalize(VecCheckNan(raw,raise_exception=True),**config['normalization'])
    model=TimedPPO('MlpPolicy',env,seed=1609,device='cpu',**ppo);model.timings=[]
    records=[]
    try:
        for target in (0,51200,102400):
            losses={}
            if target:
                model.learn(total_timesteps=51200,reset_num_timesteps=False)
                assert model.num_timesteps==target
                assert all(torch.isfinite(value).all() for value in model.policy.state_dict().values())
                losses={key:float(value) for key,value in model.logger.name_to_value.items()
                        if key.startswith('train/') and np.isscalar(value)}
                assert losses and all(np.isfinite(value) for value in losses.values())
            checkpoint=output/f'policy_{target}';model.save(checkpoint);env.save(str(checkpoint)+'.pkl')
            dev_rows=evaluate(model,str(checkpoint)+'.pkl',dev)
            hard_rows=evaluate(model,str(checkpoint)+'.pkl',hard)
            by_height={str(h):sum(row['success'] for row in hard_rows if row['scenario']['stand_height_m']==h)
                       for h in (.16,.20,.25,.30,.35,.38)}
            record=dict(policy_steps=target,stratified_success=sum(row['success'] for row in dev_rows),
                boundary_success=sum(row['success'] for row in hard_rows),boundary_by_height=by_height,
                target_observation_variance=float(env.obs_rms.var[11]),timings=model.timings.copy(),losses=losses)
            (output/f'evaluation_{target}.json').write_text(json.dumps(dict(summary=record,stratified=dev_rows,boundary=hard_rows),
                ensure_ascii=False,indent=2)+'\n')
            records.append(record)
            (output/'status.json').write_text(json.dumps(dict(status='running' if target<102400 else 'completed',
                records=records,no_weight_promotion=True),ensure_ascii=False,indent=2)+'\n')
            print('EVAL',target,record['stratified_success'],record['boundary_success'],by_height,flush=True)
        assert len(model.timings)==16 and model.num_timesteps==102400
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.output)
