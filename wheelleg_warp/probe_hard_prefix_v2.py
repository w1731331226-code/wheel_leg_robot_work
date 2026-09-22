"""Test original-GPU pre-contact M3 actions on all public hard steps, no learning."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
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
ORDER=('terrain_only','original_until_contact','original_full_hard',
       'original_full_hard','original_until_contact','terrain_only')


def run(output):
    source=json.loads(PANEL.read_text());verify(source);cases=[TerrainScenario(**row['scenario']) for row in source['panels']]
    hard={i for i,row in enumerate(source['panels']) if row['group']=='difficult/step'}
    assert len(cases)==160 and len(hard)==48
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(task_contract_version=TASK_CONTRACT_VERSION,
        source_panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),source_sha256=source_hashes(__file__),
        checkpoints=source['checkpoints'],worlds=160,hard_seeds=[cases[i].terrain_seed for i in sorted(hard)],order=ORDER,
        training=False,holdout_evaluated=False,privileged_contact_switch=True,
        note='All non-hard worlds use terrain-v3 actions. Donor modes use original-GPU actions on hard worlds either until first contact or throughout; switching uses simulator contact bits unavailable to the 32D policy.'))
    torch.set_num_threads(1);results=[]
    for repeat,mode in enumerate(ORDER,1):
        raw=TerrainEnv(len(cases),scenario=cases)
        terrain_prefix=source['checkpoints']['terrain_v3']['path'];original_prefix=source['checkpoints']['original_gpu']['path']
        terrain_norm=VecNormalize.load(terrain_prefix+'.pkl',raw);original_norm=VecNormalize.load(original_prefix+'.pkl',raw)
        terrain_norm.training=original_norm.training=False;terrain_norm.norm_reward=original_norm.norm_reward=False
        terrain_model=PPO.load(terrain_prefix+'.zip',device='cpu');original_model=PPO.load(original_prefix+'.zip',device='cpu')
        weights=[{name:value.clone() for name,value in model.policy.state_dict().items()}
                 for model in (terrain_model,original_model)]
        observations=terrain_norm.reset();pending=set(range(len(cases)));rows=[None]*len(cases)
        first_any={};first_both={};used_donor=Counter();seen={i:False for i in hard}
        initial=[(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count) for norm in (terrain_norm,original_norm)]
        try:
            step=0
            while pending:
                step+=1
                if step>1000:raise RuntimeError('A public development world did not terminate within 20 s')
                terrain_action=terrain_model.predict(observations,deterministic=True)[0]
                original_action=original_model.predict(original_norm.normalize_obs(raw.obs.numpy()),deterministic=True)[0]
                action=terrain_action.copy();before=raw.state.numpy()
                for i in sorted(hard&pending):
                    if int(before[i,14])&raw.required_terrain_contact_masks[i]:seen[i]=True
                    if mode=='original_full_hard' or (mode=='original_until_contact' and not seen[i]):
                        action[i]=original_action[i];used_donor[i]+=1
                observations,_,done,infos=terrain_norm.step(action)
                state=raw.state.numpy()
                for i in list(pending):
                    if done[i]:
                        rows[i]={k:v for k,v in infos[i].items() if k!='terminal_observation'}
                        pending.remove(i)
                for i in hard:
                    if i in first_both:continue
                    info=rows[i];mask=(info['touched_terrain_contact_mask'] if info else int(state[i,14]))&12
                    time=info['duration_s'] if info else float(state[i,0]*.0005)
                    wheels=info['wheel_progress_m'] if info else state[i,24:26].tolist()
                    if mask and i not in first_any:
                        first_any[i]=dict(time_s=time,wheel_gap_m=abs(wheels[0]-wheels[1]))
                    if mask==12:first_both[i]=time
            validate_terrain_rows(rows)
            verify(source)
            for model,initial_weights in zip((terrain_model,original_model),weights):
                assert all(torch.equal(value,model.policy.state_dict()[name]) for name,value in initial_weights.items())
            for norm,stats in zip((terrain_norm,original_norm),initial):
                np.testing.assert_array_equal(norm.obs_rms.mean,stats[0]);np.testing.assert_array_equal(norm.obs_rms.var,stats[1]);assert norm.obs_rms.count==stats[2]
            observations_rows=[dict(seed=cases[i].terrain_seed,group=source['panels'][i]['group'],**row) for i,row in enumerate(rows)]
            result=dict(run=repeat,mode=mode,all_success=sum(row['success'] for row in rows),
                hard_success=sum(rows[i]['success'] for i in hard),hard_complete=sum(rows[i]['reason']=='completed' for i in hard),
                outcomes=observations_rows,hard_contact={str(cases[i].terrain_seed):dict(first_any=first_any.get(i),
                    first_both_s=first_both.get(i),used_donor_policy_steps=used_donor[i]) for i in sorted(hard)})
            write(output/f'run_{repeat:02d}_{mode}.json',result);results.append(result)
            print('run',repeat,mode,'hard',result['hard_success'],'complete',result['hard_complete'],'all',result['all_success'],flush=True)
        finally:terrain_norm.close()
    groups=defaultdict(list)
    for result in results:groups[result['mode']].append(result)
    summary={mode:dict(hard_success=[r['hard_success'] for r in items],hard_complete=[r['hard_complete'] for r in items],
        all_success=[r['all_success'] for r in items]) for mode,items in groups.items()}
    write(output/'summary.json',dict(task_contract_version=TASK_CONTRACT_VERSION,groups=summary,run_order=ORDER,
        training=False,holdout_evaluated=False,promoted=False,formal_training_ready=False,
        required_hard_success=45,required_hard_complete=46,
        note='A simulator-contact-based intervention is not an deployable 32D observation policy. Judge full runs, repeatability and original regression separately.'))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output)
