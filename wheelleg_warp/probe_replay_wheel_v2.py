"""Pre-registered wheel-difference actions from one serialized public prefix."""
from collections import defaultdict
from pathlib import Path
import argparse, hashlib, json, sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from probe_replay_prefix_v2 import PANEL,SEED,model_hashes,restore
from training_contract import TASK_CONTRACT_VERSION,source_hashes
from dashboard.live_env import atomic_json as write

PREFIX=ROOT/'wheelleg_warp/results/replay_prefix_probe_v2_20260928'
ORDER=('baseline','wheel_minus_one','wheel_plus_one','wheel_plus_one','wheel_minus_one','baseline')*3


def action_wheel(original,mode,step):
    if step>30 or mode=='baseline':return original
    return -1. if mode=='wheel_minus_one' else 1.


def rollout(env,norm,model,index,mode):
    records=[];changed=0
    for step in range(1,351):
        action=model.predict(norm.normalize_obs(env.obs.numpy()),deterministic=True)[0]
        original=float(action[index,2]);action[index,2]=action_wheel(original,mode,step)
        changed+=int(action[index,2]!=original)
        env.targets.assign(action);wp.capture_launch(env.graph)
        state=env.state.numpy();wheel=state[index,24:26]
        records.append(dict(step=step,time_s=float(state[index,0]*.0005),
            original_wheel_action=original,used_wheel_action=float(action[index,2]),
            contact_mask=int(state[index,14]),wheel_progress_m=wheel.tolist(),
            peak_deg=(state[index,21:24]*180/np.pi).tolist(),done=int(env.done.numpy()[index])))
        if env.done.numpy()[index]:break
    if not records[-1]['done']:raise RuntimeError('Selected world did not terminate')
    first=lambda bit:next((r['time_s'] for r in records if r['contact_mask']&bit),None)
    return dict(mode=mode,changed_policy_steps=changed,policy_steps=len(records),
        reason=records[-1]['done'],success=bool(env.state.numpy()[index,19]),
        first_left_contact_policy_s=first(4),first_right_contact_policy_s=first(8),
        first_5deg_policy_s=next((r['time_s'] for r in records if max(r['peak_deg'])>5),None),
        peak_deg=max(records[-1]['peak_deg']),records=records)


def run(output):
    old=json.loads((PREFIX/'protocol.json').read_text())
    assert old['task_contract_version']==TASK_CONTRACT_VERSION and old['worlds']==160 and old['seed']==SEED
    for path,digest in old['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    manifest=json.loads((PREFIX/'prefix_manifest.json').read_text())
    snapshot_digest=hashlib.sha256((PREFIX/'prefix_state.npz').read_bytes()).hexdigest()
    assert snapshot_digest==manifest['snapshot_sha256']
    frozen=json.loads(PANEL.read_text())
    assert hashlib.sha256(PANEL.read_bytes()).hexdigest()==old['panel_sha256']
    checkpoint=frozen['checkpoints']['terrain_v3']['path']
    for suffix,key in (('.zip','checkpoint_sha256'),('.pkl','normalization_sha256')):
        assert hashlib.sha256(Path(checkpoint+suffix).read_bytes()).hexdigest()==frozen['checkpoints']['terrain_v3'][key]
    cases=[TerrainScenario(**row['scenario']) for row in frozen['panels']]
    index=next(i for i,case in enumerate(cases) if case.terrain_seed==SEED)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(task_contract_version=TASK_CONTRACT_VERSION,
        prefix_snapshot_sha256=snapshot_digest,source_sha256=source_hashes(__file__),
        checkpoint=frozen['checkpoints']['terrain_v3'],seed=SEED,worlds=160,
        order=ORDER,intervention_policy_steps=30,action_channel='M3 differential wheel torque',
        action_targets={'wheel_minus_one':-1.,'wheel_plus_one':1.},
        exploratory_signal='candidate >=5/6 success while baseline <=1/6; never sufficient for promotion',
        training=False,holdout_evaluated=False,privileged_prefix=True))
    env=TerrainEnv(160,scenario=cases)
    norm=VecNormalize.load(checkpoint+'.pkl',env);norm.training=False;norm.norm_reward=False
    torch.set_num_threads(1);model=PPO.load(checkpoint+'.zip',device='cpu')
    stats=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
    grouped=defaultdict(list)
    try:
        assert model_hashes(env)==manifest['static_model_array_sha256']
        assert [env.data.nworld,env.data.naconmax,env.data.njmax]==manifest['data_capacity']
        with np.load(PREFIX/'prefix_state.npz',allow_pickle=False) as saved:
            for number,mode in enumerate(ORDER,1):
                restore(env,saved)
                result=rollout(env,norm,model,index,mode)
                write(output/f'run_{number:02d}.json',result);grouped[mode].append(result)
                print(number,mode,'changed',result['changed_policy_steps'],
                    'success',result['success'],'reason',result['reason'],'peak',round(result['peak_deg'],3),flush=True)
        np.testing.assert_array_equal(norm.obs_rms.mean,stats[0]);np.testing.assert_array_equal(norm.obs_rms.var,stats[1])
        assert norm.obs_rms.count==stats[2]
        summary={mode:dict(success=[r['success'] for r in runs],
            reason=[r['reason'] for r in runs],peak_deg=[r['peak_deg'] for r in runs],
            changed_policy_steps=[r['changed_policy_steps'] for r in runs]) for mode,runs in grouped.items()}
        baseline_success=sum(summary['baseline']['success'])
        signal={mode:sum(row['success'])>=5 and baseline_success<=1
                for mode,row in summary.items() if mode!='baseline'}
        write(output/'summary.json',dict(groups=summary,exploratory_signal=signal,
            snapshot_restored_bitwise_each_run=True,training=False,promoted=False,
            note='Six correlated GPU replays per mode from one public prefix; not six independent scenario seeds or a full-panel regression.'))
    finally:norm.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        assert action_wheel(.2,'baseline',1)==.2
        assert action_wheel(.2,'wheel_minus_one',1)==-1.
        assert action_wheel(.2,'wheel_plus_one',1)==1.
        assert action_wheel(.2,'wheel_plus_one',31)==.2
        print('wheel gate check passed')
    elif a.output is None:p.error('--output required unless --check')
    else:run(a.output)
