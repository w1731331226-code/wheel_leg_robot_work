"""Record selected public failures at 2kHz while retaining the full 160-world batch."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from native.terrain import TerrainScenario
from terrain_eval import validate_terrain_rows
from trace_failure_chain import RecordedEnv,COL,analyze
from training_contract import TASK_CONTRACT_VERSION,source_hashes
from dashboard.live_env import atomic_json as write

PANEL=ROOT/'wheelleg_warp/results/contract_v2_baseline_checked_20260923/protocol.json'
SEEDS=(330083,700100,700101,700102,700103)


def run(output):
    source=json.loads(PANEL.read_text());assert source['task_contract_version']==TASK_CONTRACT_VERSION
    for name,digest in source['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for item in source['checkpoints'].values():
        assert hashlib.sha256(Path(item['path']+'.zip').read_bytes()).hexdigest()==item['checkpoint_sha256']
        assert hashlib.sha256(Path(item['path']+'.pkl').read_bytes()).hexdigest()==item['normalization_sha256']
    cases=[TerrainScenario(**entry['scenario']) for entry in source['panels']]
    which={s:next(i for i,case in enumerate(cases) if case.terrain_seed==s) for s in SEEDS}
    assert len(cases)==160 and len(set(which.values()))==len(SEEDS)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(task_contract_version=TASK_CONTRACT_VERSION,source_panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),
        source_sha256={**source_hashes(__file__),str(Path('wheelleg_warp/trace_failure_chain.py')):hashlib.sha256((ROOT/'wheelleg_warp/trace_failure_chain.py').read_bytes()).hexdigest()},
        checkpoints=source['checkpoints'],selected_seeds=SEEDS,worlds=160,training=False,holdout_evaluated=False,
        note='Read-only telemetry. Legacy bump contact time is bracketed at 50Hz from cumulative contact bits; terrain contacts are resolved at 2kHz.'))
    torch.set_num_threads(1);results=[]
    for label in ('terrain_v3','original_gpu'):
        checkpoint=source['checkpoints'][label]['path'];raw=RecordedEnv(len(cases),scenario=cases)
        env=VecNormalize.load(checkpoint+'.pkl',raw);env.training=False;env.norm_reward=False
        model=PPO.load(checkpoint+'.zip',device='cpu')
        obs=env.reset();pending=set(which.values());all_pending=set(range(len(cases)));all_info=[None]*len(cases)
        chunks={i:[] for i in pending};actions={i:[] for i in pending}
        observations={i:[] for i in pending};raw_observations={i:[] for i in pending}
        progress={i:[] for i in pending};contact_bits={i:[] for i in pending};times={i:[] for i in pending}
        try:
            steps=0
            while all_pending:
                steps+=1
                if steps>1000:raise RuntimeError('A public development world did not terminate within 20 s')
                action=model.predict(obs,deterministic=True)[0]
                current_raw_obs=raw.obs.numpy() if pending else None
                for i in pending:
                    actions[i].append(action[i].copy());observations[i].append(obs[i].copy())
                    raw_observations[i].append(current_raw_obs[i].copy())
                obs,_,done,infos=env.step(action)
                physical=raw.trace.numpy() if pending else None
                states=raw.state.numpy() if pending else None
                for i in list(all_pending):
                    if done[i]:
                        all_info[i]={key:value for key,value in infos[i].items() if key!='terminal_observation'}
                        all_pending.remove(i)
                for i in list(pending):
                    chunks[i].append(physical[physical[:,i,0]>0,i,:].copy())
                    if done[i]:
                        info=all_info[i]
                        validate_terrain_rows([info]);wheel=info['wheel_progress_m']
                        touched=info['touched_contact_mask']|info['touched_terrain_contact_mask'];time=info['duration_s']
                    else:
                        wheel=states[i,24:26].tolist();touched=int(states[i,14]);time=float(states[i,0]*.0005)
                    progress[i].append(wheel);contact_bits[i].append(touched);times[i].append(time)
                    if not done[i]:continue
                    trace=np.concatenate(chunks[i]);assert len(trace)==info['physical_steps']
                    analysis=analyze(trace)
                    np.testing.assert_allclose(analysis['reward'],info['episode']['r'],atol=1e-7)
                    seed=cases[i].terrain_seed;file=output/f'{label}_{seed}.npz'
                    with file.open('xb') as stream:
                        np.savez_compressed(stream,trace=trace,columns=np.asarray(COL),policy_time=np.asarray(times[i]),
                            policy_action=np.asarray(actions[i]),policy_observation=np.asarray(observations[i]),
                            raw_observation=np.asarray(raw_observations[i]),
                            wheel_progress=np.asarray(progress[i]),contact_mask=np.asarray(contact_bits[i],dtype=np.int32))
                    row=dict(policy=label,seed=seed,group=source['panels'][i]['group'],checkpoint=checkpoint,
                        scenario=source['panels'][i]['scenario'],info=info,trace_file=file.name,
                        first_required_bump_contact_policy_s=next((float(t) for t,m in zip(times[i],contact_bits[i])
                            if m&int(raw.required_contact_masks[i])==int(raw.required_contact_masks[i])),None)
                            if raw.required_contact_masks[i] else None,
                        first_any_bump_contact_policy_s=next((float(t) for t,m in zip(times[i],contact_bits[i])
                            if m&int(raw.required_contact_masks[i])),None) if raw.required_contact_masks[i] else None,
                        **analysis)
                    results.append(row);pending.remove(i)
                    print(label,seed,'success',info['success'],'reason',info['reason'],'first_breach',row['first_attitude_failure_s'],flush=True)
            validate_terrain_rows(all_info)
            previous=json.loads((ROOT/f"wheelleg_warp/results/contract_v2_baseline_checked_20260923/run_{1 if label=='terrain_v3' else 2:02d}_{label}.json").read_text())['runs']
            comparison=dict(policy=label,success=sum(info['success'] for info in all_info),
                complete=sum(info['reason']=='completed' for info in all_info),
                gained=[cases[i].terrain_seed for i in range(len(cases)) if all_info[i]['success'] and not previous[i]['success']],
                lost=[cases[i].terrain_seed for i in range(len(cases)) if previous[i]['success'] and not all_info[i]['success']])
            write(output/f'all_worlds_{label}.json',comparison)
        finally:env.close()
    write(output/'summary.json',dict(runs=results,selected_seeds=SEEDS,worlds=160,training=False,
        note='Case outcomes are a new instrumented batch run, not asserted identical to earlier uninstrumented repeats.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
