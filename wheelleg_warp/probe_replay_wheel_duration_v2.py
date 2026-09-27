"""One frozen duration screen for the bounded wheel-difference pulse."""
from collections import defaultdict
from pathlib import Path
import argparse,hashlib,json,sys

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
from probe_replay_wheel_phase_v2 import yaw_degrees
from training_contract import TASK_CONTRACT_VERSION,source_hashes
from dashboard.live_env import atomic_json as write

ORDER=(30,28,26,26,28,30)*3


def rollout(env,norm,model,index,duration):
    records=[];changed=0
    for step in range(1,351):
        action=model.predict(norm.normalize_obs(env.obs.numpy()),deterministic=True)[0]
        original=float(action[index,2])
        if step<=duration:action[index,2]=-1.;changed+=int(action[index,2]!=original)
        env.targets.assign(action);wp.capture_launch(env.graph)
        state=env.state.numpy();q=env.data.qpos.numpy()[index]
        records.append(dict(step=step,time_s=float(state[index,0]*.0005),
            original_action=original,used_action=float(action[index,2]),
            contact_mask=int(state[index,14]),yaw_deg=yaw_degrees(q),
            peak_deg=(state[index,21:24]*180/np.pi).tolist(),done=int(env.done.numpy()[index])))
        if env.done.numpy()[index]:break
    if not records[-1]['done']:raise RuntimeError('Selected world did not terminate')
    first=lambda bit:next((r['time_s'] for r in records if r['contact_mask']&bit),None)
    return dict(pulse_policy_steps=duration,changed_policy_steps=changed,reason=records[-1]['done'],
        success=bool(env.state.numpy()[index,19]),first_left_contact_policy_s=first(4),
        first_right_contact_policy_s=first(8),
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
        phase_probe_sha256=hashlib.sha256((ROOT/'wheelleg_warp/probe_replay_wheel_phase_v2.py').read_bytes()).hexdigest(),
        checkpoint=frozen['checkpoints']['terrain_v3'],seed=SEED,worlds=160,order=ORDER,
        action_channel='M3 differential wheel torque',action_target=-1.,
        exploratory_signal='duration 26 or 28 >=5/6 successes while duration 30 <=1/6; never sufficient for promotion',
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
            for number,duration in enumerate(ORDER,1):
                restore(env,saved)
                result=rollout(env,norm,model,index,duration)
                write(output/f'run_{number:02d}.json',result);grouped[duration].append(result)
                print(number,'duration',duration,'success',result['success'],
                      'reason',result['reason'],'peak',round(result['peak_deg'],3),flush=True)
        np.testing.assert_array_equal(norm.obs_rms.mean,stats[0]);np.testing.assert_array_equal(norm.obs_rms.var,stats[1])
        assert norm.obs_rms.count==stats[2]
        summary={str(duration):dict(success=[r['success'] for r in runs],reason=[r['reason'] for r in runs],
            peak_deg=[r['peak_deg'] for r in runs],changed_policy_steps=[r['changed_policy_steps'] for r in runs])
            for duration,runs in grouped.items()}
        signal={key:sum(row['success'])>=5 and sum(summary['30']['success'])<=1
                for key,row in summary.items() if key!='30'}
        write(output/'summary.json',dict(groups=summary,exploratory_signal=signal,
            snapshot_restored_bitwise_each_run=True,training=False,promoted=False,
            note='Final fixed-pulse duration screen on one public prefix; six correlated GPU replays per duration.'))
    finally:norm.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        assert len(ORDER)==18 and sorted(set(ORDER))==[26,28,30]
        assert all(ORDER.count(x)==6 for x in (26,28,30))
        print('duration order check passed')
    elif a.output is None:p.error('--output required unless --check')
    else:run(a.output)
