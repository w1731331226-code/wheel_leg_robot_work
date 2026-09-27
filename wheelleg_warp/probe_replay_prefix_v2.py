"""Serialize a full 160-world Warp prefix and compare bounded M3 actions."""
from dataclasses import fields, is_dataclass
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
from training_contract import TASK_CONTRACT_VERSION,source_hashes
from dashboard.live_env import atomic_json as write

PANEL=ROOT/'wheelleg_warp/results/contract_v2_baseline_checked_20260923/protocol.json'
SEED=700302
MODES=('baseline','baseline','support_plus_one','support_plus_one')
EXTRA=('state','param','command','active','done','reward','residual','obs','history','targets',
       'stopped_q','stopped_v','stopped_w','contact_flags','mask','q0','wheel_offsets','ids')


def array_fields(prefix,obj):
    for field in fields(obj):
        value=getattr(obj,field.name);name=prefix+field.name
        if isinstance(value,wp.array):yield name,value
        elif is_dataclass(value):yield from array_fields(name+'__',value)


def arrays(env):
    result=dict(array_fields('data__',env.data))
    result.update({'env__'+key:getattr(env,key) for key in EXTRA})
    result.update({'k__'+key:value for key,value in env.k.items()})
    return result


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def model_hashes(env):
    return {name:digest(array.numpy()) for name,array in array_fields('model__',env.model)}


def restore(env,saved):
    current=arrays(env)
    if set(current)!=set(saved.files):raise ValueError('Warp state field list changed')
    for name,array in current.items():
        value=saved[name]
        if value.shape!=array.numpy().shape:raise ValueError('Warp state shape changed: '+name)
        array.assign(value)
    bad=[name for name,array in current.items() if digest(array.numpy())!=digest(saved[name])]
    if bad:raise ValueError('Warp state did not restore exactly: '+str(bad[:5]))


def rollout(env,norm,model,index,mode):
    records=[];changed=0
    for step in range(1,351):
        state=env.state.numpy();action=model.predict(norm.normalize_obs(env.obs.numpy()),deterministic=True)[0]
        original=float(action[index,0])
        if mode=='support_plus_one' and step<=30:
            action[index,0]=1.;changed+=int(action[index,0]!=original)
        env.targets.assign(action);wp.capture_launch(env.graph)
        state=env.state.numpy();q=env.data.qpos.numpy()[index]
        records.append(dict(step=step,time_s=float(state[index,0]*.0005),
            action0_original=original,action0_used=float(action[index,0]),
            wheel_progress_m=state[index,24:26].tolist(),contact_mask=int(state[index,14]),
            peak_deg=(state[index,21:24]*180/np.pi).tolist(),
            base_xyz_m=q[:3].tolist(),done=int(env.done.numpy()[index])))
        if env.done.numpy()[index]:break
    if not records[-1]['done']:raise RuntimeError('Selected world did not terminate after prefix replay')
    return dict(mode=mode,steps=len(records),changed_policy_steps=changed,
        reason=records[-1]['done'],success=bool(env.state.numpy()[index,19]),
        first_target_contact_policy_s=next((r['time_s'] for r in records if r['contact_mask']&12),None),
        first_5deg_policy_s=next((r['time_s'] for r in records if max(r['peak_deg'])>5),None),
        max_peak_deg=max(records[-1]['peak_deg']),records=records)


def run(output):
    frozen=json.loads(PANEL.read_text())
    assert frozen['task_contract_version']==TASK_CONTRACT_VERSION and len(frozen['panels'])==160
    checkpoint=frozen['checkpoints']['terrain_v3']['path']
    for suffix,key in (('.zip','checkpoint_sha256'),('.pkl','normalization_sha256')):
        assert hashlib.sha256(Path(checkpoint+suffix).read_bytes()).hexdigest()==frozen['checkpoints']['terrain_v3'][key]
    cases=[TerrainScenario(**row['scenario']) for row in frozen['panels']]
    index=next(i for i,case in enumerate(cases) if case.terrain_seed==SEED)
    assert len({case.terrain_seed for case in cases})==160
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(task_contract_version=TASK_CONTRACT_VERSION,
        panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),source_sha256=source_hashes(__file__),
        checkpoint=frozen['checkpoints']['terrain_v3'],seed=SEED,worlds=160,modes=MODES,
        prefix_progress_window_m=[-.62,-.58],intervention_policy_steps=30,
        training=False,holdout_evaluated=False,privileged_prefix=True,
        note='All Warp Data and environment arrays serialized at one public pre-contact policy boundary; short M3 override is an upper-bound probe.'))
    torch.set_num_threads(1);source=TerrainEnv(160,scenario=cases)
    source_norm=VecNormalize.load(checkpoint+'.pkl',source)
    source_norm.training=False;source_norm.norm_reward=False
    model=PPO.load(checkpoint+'.zip',device='cpu')
    try:
        source.reset()
        for step in range(1,501):
            action=model.predict(source_norm.normalize_obs(source.obs.numpy()),deterministic=True)[0]
            source.targets.assign(action);wp.capture_launch(source.graph)
            state=source.state.numpy();progress=float(min(state[index,24:26]))
            if -.62<=progress<=-.58 and int(state[index,14])&12==0:break
        else:raise RuntimeError('Target did not reach the frozen pre-contact window')
        fields_to_save={name:array.numpy().copy() for name,array in arrays(source).items()}
        with (output/'prefix_state.npz').open('xb') as stream:np.savez_compressed(stream,**fields_to_save)
        static=model_hashes(source)
        write(output/'prefix_manifest.json',dict(policy_steps=step,time_s=float(state[index,0]*.0005),
            target_wheel_progress_m=state[index,24:26].tolist(),contact_mask=int(state[index,14]),
            saved_arrays=len(fields_to_save),static_model_array_sha256=static,
            data_capacity=[source.data.nworld,source.data.naconmax,source.data.njmax],
            snapshot_sha256=hashlib.sha256((output/'prefix_state.npz').read_bytes()).hexdigest()))
    finally:source_norm.close()
    replay=TerrainEnv(160,scenario=cases)
    replay_norm=VecNormalize.load(checkpoint+'.pkl',replay)
    replay_norm.training=False;replay_norm.norm_reward=False
    results=[]
    try:
        assert model_hashes(replay)==static
        manifest=json.loads((output/'prefix_manifest.json').read_text())
        assert [replay.data.nworld,replay.data.naconmax,replay.data.njmax]==manifest['data_capacity']
        initial_stats=(replay_norm.obs_rms.mean.copy(),replay_norm.obs_rms.var.copy(),replay_norm.obs_rms.count)
        with np.load(output/'prefix_state.npz',allow_pickle=False) as saved:
            for number,mode in enumerate(MODES,1):
                restore(replay,saved)
                result=rollout(replay,replay_norm,model,index,mode)
                write(output/f'run_{number:02d}.json',result);results.append(result)
                print('run',number,mode,'changed',result['changed_policy_steps'],
                    'reason',result['reason'],'success',result['success'],'peak',result['max_peak_deg'],flush=True)
        np.testing.assert_array_equal(replay_norm.obs_rms.mean,initial_stats[0])
        np.testing.assert_array_equal(replay_norm.obs_rms.var,initial_stats[1])
        assert replay_norm.obs_rms.count==initial_stats[2]
        write(output/'summary.json',dict(runs=[{k:v for k,v in r.items() if k!='records'} for r in results],
            snapshot_restored_bitwise=True,model_arrays_equal=True,training=False,
            note='Repeated baseline trajectories quantify GPU variation; candidate is not a deployable policy or a paired capability result.'))
    finally:replay_norm.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
