"""Exploratory pre-contact action swap, fixed policies and task contract v2."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from baseline_contract_v2 import verify
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from terrain_eval import validate_terrain_rows
from training_contract import source_hashes,TASK_CONTRACT_VERSION
from dashboard.live_env import atomic_json as write

PANEL=ROOT/'wheelleg_warp/results/contract_v2_baseline_checked_20260923/protocol.json'
SEEDS=(330083,700102)
ORDER=('terrain_only','original_until_contact','original_full_selected',
       'original_full_selected','original_until_contact','terrain_only')


def yaw_deg(q):
    w,x,y,z=map(float,q[3:7]);return float(np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))


def run(output):
    frozen=json.loads(PANEL.read_text());verify(frozen)
    cases=[TerrainScenario(**row['scenario']) for row in frozen['panels']]
    selected={seed:next(i for i,c in enumerate(cases) if c.terrain_seed==seed) for seed in SEEDS}
    assert len(cases)==160 and len(set(selected.values()))==2
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(task_contract_version=TASK_CONTRACT_VERSION,
        panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),source_sha256=source_hashes(__file__),
        checkpoint_hashes=frozen['checkpoints'],selected_seeds=SEEDS,order=ORDER,worlds=len(cases),
        training=False,holdout_evaluated=False,
        intervention='For two selected worlds only, compare terrain-v3 throughout, original-GPU throughout, and original-GPU until either required obstacle contact is observed at 50Hz then terrain-v3 from the next policy step. All other worlds always use terrain-v3.'))
    torch.set_num_threads(1);runs=[]
    for repeat,mode in enumerate(ORDER,1):
        raw=TerrainEnv(len(cases),scenario=cases)
        terrain_prefix=frozen['checkpoints']['terrain_v3']['path'];original_prefix=frozen['checkpoints']['original_gpu']['path']
        terrain_norm=VecNormalize.load(terrain_prefix+'.pkl',raw);original_norm=VecNormalize.load(original_prefix+'.pkl',raw)
        terrain_norm.training=original_norm.training=False;terrain_norm.norm_reward=original_norm.norm_reward=False
        terrain_model=PPO.load(terrain_prefix+'.zip',device='cpu');original_model=PPO.load(original_prefix+'.zip',device='cpu')
        observations=terrain_norm.reset();all_pending=set(range(len(cases)));selected_pending=set(selected.values())
        rows=[None]*len(cases);trace={i:[] for i in selected_pending};contacted={i:False for i in selected_pending}
        initial=[(terrain_norm.obs_rms.mean.copy(),terrain_norm.obs_rms.var.copy(),terrain_norm.obs_rms.count),
                 (original_norm.obs_rms.mean.copy(),original_norm.obs_rms.var.copy(),original_norm.obs_rms.count)]
        try:
            step=0
            while all_pending:
                step+=1
                if step>1000:raise RuntimeError('A public development world did not terminate within 20 s')
                source_action=terrain_model.predict(observations,deterministic=True)[0]
                raw_obs=raw.obs.numpy()
                original_input=original_norm.normalize_obs(raw_obs)
                original_action=original_model.predict(original_input,deterministic=True)[0]
                action=source_action.copy();before=raw.state.numpy()
                for i in selected_pending:
                    mask=raw.required_contact_masks[i] or raw.required_terrain_contact_masks[i]
                    if int(before[i,14])&mask:contacted[i]=True
                    if mode=='original_full_selected' or (mode=='original_until_contact' and not contacted[i]):
                        action[i]=original_action[i]
                observations,_,done,infos=terrain_norm.step(action)
                after=raw.state.numpy() if selected_pending else None
                controls=raw.k['state'].numpy() if selected_pending else None
                diagnostics=raw.diag.numpy() if selected_pending else None
                q=raw.data.qpos.numpy() if selected_pending else None
                for i in list(all_pending):
                    if done[i]:
                        rows[i]={k:v for k,v in infos[i].items() if k!='terminal_observation'}
                        all_pending.remove(i)
                for i in list(selected_pending):
                    info=rows[i]
                    mask_value=(info['touched_contact_mask']|info['touched_terrain_contact_mask']) if info else int(after[i,14])
                    wheel=info['wheel_progress_m'] if info else after[i,24:26].tolist()
                    time=info['duration_s'] if info else float(after[i,0]*.0005)
                    trace[i].append(dict(time=time,original_action=original_action[i].tolist(),terrain_action=source_action[i].tolist(),
                        used_action=action[i].tolist(),contact_mask=mask_value,wheel_progress=wheel,
                        filtered_action=controls[i,16:19].tolist() if not info else None,
                        lambda_last_substep=float(diagnostics[i,12]),
                        yaw_deg=yaw_deg(q[i]) if not info else None,
                        used_original=bool(mode=='original_full_selected' or (mode=='original_until_contact' and not contacted[i]))))
                    if info:selected_pending.remove(i)
            validate_terrain_rows(rows)
            for norm,stats in zip((terrain_norm,original_norm),initial):
                np.testing.assert_array_equal(norm.obs_rms.mean,stats[0]);np.testing.assert_array_equal(norm.obs_rms.var,stats[1]);assert norm.obs_rms.count==stats[2]
            selected_rows={str(seed):dict(info=rows[i],trace=trace[i]) for seed,i in selected.items()}
            result=dict(run=repeat,mode=mode,all_success=sum(row['success'] for row in rows),
                all_complete=sum(row['reason']=='completed' for row in rows),
                selected=selected_rows,training=False,holdout_evaluated=False)
            write(output/f'run_{repeat:02d}_{mode}.json',result);runs.append(result)
            print('run',repeat,mode,[(seed,rows[i]['success']) for seed,i in selected.items()],flush=True)
        finally:terrain_norm.close()
    report=dict(task_contract_version=TASK_CONTRACT_VERSION,runs=[dict(run=r['run'],mode=r['mode'],
        all_success=r['all_success'],selected_success={seed:data['info']['success'] for seed,data in r['selected'].items()}) for r in runs],
        training=False,weights_promoted=False,holdout_evaluated=False,
        note='Exploratory within-panel action intervention; GPU contacts can change with instrumentation, and two chosen cases do not establish general success gains.')
    write(output/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output)
