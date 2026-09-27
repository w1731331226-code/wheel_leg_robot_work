"""Pre-registered post-contact wheel correction after a fixed pre-contact pulse."""
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
from probe_replay_wheel_v2 import PREFIX
from training_contract import TASK_CONTRACT_VERSION,source_hashes
from dashboard.live_env import atomic_json as write

ORDER=('minus_only','minus_then_zero','minus_then_plus','minus_then_plus','minus_then_zero','minus_only')*3


def wheel_target(original,mode,step):
    if step<=30:return -1.
    if step<=40:
        if mode=='minus_then_zero':return 0.
        if mode=='minus_then_plus':return 1.
    return original


def yaw_degrees(q):
    w,x,y,z=map(float,q[3:7])
    return float(np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))


def rollout(env,norm,model,index,mode):
    records=[];changed=0
    for step in range(1,351):
        action=model.predict(norm.normalize_obs(env.obs.numpy()),deterministic=True)[0]
        original=float(action[index,2]);action[index,2]=wheel_target(original,mode,step)
        changed+=int(action[index,2]!=original)
        env.targets.assign(action);wp.capture_launch(env.graph)
        state=env.state.numpy();q=env.data.qpos.numpy()[index]
        records.append(dict(step=step,time_s=float(state[index,0]*.0005),
            original_action=original,used_action=float(action[index,2]),
            contact_mask=int(state[index,14]),wheel_progress_m=state[index,24:26].tolist(),
            yaw_deg=yaw_degrees(q),peak_deg=(state[index,21:24]*180/np.pi).tolist(),
            done=int(env.done.numpy()[index])))
        if env.done.numpy()[index]:break
    if not records[-1]['done']:raise RuntimeError('Selected world did not terminate')
    first=lambda bit:next((r['time_s'] for r in records if r['contact_mask']&bit),None)
    return dict(mode=mode,changed_policy_steps=changed,policy_steps=len(records),
        reason=records[-1]['done'],success=bool(env.state.numpy()[index,19]),
        first_left_contact_policy_s=first(4),first_right_contact_policy_s=first(8),
        first_5deg_policy_s=next((r['time_s'] for r in records if max(r['peak_deg'])>5),None),
        peak_deg=max(records[-1]['peak_deg']),records=records)


def run(output):
    parent=json.loads((PREFIX/'protocol.json').read_text())
    assert parent['task_contract_version']==TASK_CONTRACT_VERSION and parent['seed']==SEED and parent['worlds']==160
    for path,digest in parent['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    manifest=json.loads((PREFIX/'prefix_manifest.json').read_text())
    snapshot_digest=hashlib.sha256((PREFIX/'prefix_state.npz').read_bytes()).hexdigest()
    assert snapshot_digest==manifest['snapshot_sha256']
    frozen=json.loads(PANEL.read_text());assert hashlib.sha256(PANEL.read_bytes()).hexdigest()==parent['panel_sha256']
    checkpoint=frozen['checkpoints']['terrain_v3']['path']
    for suffix,key in (('.zip','checkpoint_sha256'),('.pkl','normalization_sha256')):
        assert hashlib.sha256(Path(checkpoint+suffix).read_bytes()).hexdigest()==frozen['checkpoints']['terrain_v3'][key]
    cases=[TerrainScenario(**row['scenario']) for row in frozen['panels']]
    index=next(i for i,case in enumerate(cases) if case.terrain_seed==SEED)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(task_contract_version=TASK_CONTRACT_VERSION,
        prefix_snapshot_sha256=snapshot_digest,source_sha256=source_hashes(__file__),
        parent_probe_sha256=hashlib.sha256((ROOT/'wheelleg_warp/probe_replay_wheel_v2.py').read_bytes()).hexdigest(),
        checkpoint=frozen['checkpoints']['terrain_v3'],seed=SEED,worlds=160,order=ORDER,
        first_30_policy_steps=-1.,next_10_policy_steps={'minus_only':'policy','minus_then_zero':0.,'minus_then_plus':1.},
        exploratory_signal='>=5/6 successes for a phased mode and <=1/6 for minus_only; never sufficient for promotion',
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
        summary={mode:dict(success=[r['success'] for r in runs],reason=[r['reason'] for r in runs],
            peak_deg=[r['peak_deg'] for r in runs],changed_policy_steps=[r['changed_policy_steps'] for r in runs])
            for mode,runs in grouped.items()}
        signal={mode:sum(row['success'])>=5 and sum(summary['minus_only']['success'])<=1
                for mode,row in summary.items() if mode!='minus_only'}
        write(output/'summary.json',dict(groups=summary,exploratory_signal=signal,
            snapshot_restored_bitwise_each_run=True,training=False,promoted=False,
            note='Six correlated GPU replays per mode from one public prefix; fixed post-contact timing is privileged and not a deployable 32D policy.'))
    finally:norm.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        assert wheel_target(.2,'minus_only',1)==-1.
        assert wheel_target(.2,'minus_only',31)==.2
        assert wheel_target(.2,'minus_then_zero',31)==0.
        assert wheel_target(.2,'minus_then_plus',40)==1.
        assert wheel_target(.2,'minus_then_plus',41)==.2
        assert yaw_degrees([0.,0.,0.,1.,0.,0.,0.])==0.
        print('phase gate check passed')
    elif a.output is None:p.error('--output required unless --check')
    else:run(a.output)
