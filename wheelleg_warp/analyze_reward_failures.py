"""Offline original-gate failure reconstruction and bootstrap-history association."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reward_pilot_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def flags(r):
    relative=r['scenario']['relative_attitude'];att=r['relative_peak_deg'] if relative else r['peak_deg']
    return dict(completion=r['reason']!='completed',physical=not r['physical_safety_passed'],design=not r['design_joint_passed'],
        legacy_contact=(r['touched_contact_mask']&r['required_contact_mask'])!=r['required_contact_mask'],
        terrain_contact=not r['terrain_passed'],terrain_exit=not r['terrain_exit_passed'],
        attitude=max(att)>5,yaw=r['peak_deg'][2]>5,world_tilt=relative and max(r['peak_deg'][:2])>10,
        velocity=r['velocity_rmse']>.2*abs(r['scenario']['speed']),stop=r['stop_distance_m']>.6,
        tail=r['tail_speed_m_s']>.03,height_rms=r['height_rmse_m']>.02,
        height_final=abs(r['final_mean_fk_leg_m']-r['target_leg_m'])>.02)

def run():
    review=json.loads((S/'pair_review.json').read_text());assert review['completed_runs']==6
    for name,value in review['artifact_sha256'].items():assert sha(S/name)==value
    results={};pairs={};sources={};allrows={}
    for seed in [1609,1610,1611]:
        for arm in ['original','potential']:
            p=S/'runs'/arm/str(seed);data=json.loads((p/'development_final.json').read_text());rows=data['runs'];key=f'{arm}/{seed}'
            decoded=[flags(r) for r in rows]
            assert all(r['success']==(not any(f.values())) for r,f in zip(rows,decoded))
            counts={name:sum(f[name] for f in decoded) for name in decoded[0]}
            terrain_only=sum(not r['success'] and f['terrain_contact'] and not any(value for name,value in f.items() if name!='terrain_contact') for r,f in zip(rows,decoded))
            groups={label:dict(total=len(selected),success=sum(r['success'] for r in selected)) for label,selected in [
                ('115..160mm',[r for r in rows if r['scenario']['stand_height_m']<=.16]),
                ('160..300mm',[r for r in rows if .16<r['scenario']['stand_height_m']<=.3]),
                ('300..380mm',[r for r in rows if r['scenario']['stand_height_m']>.3]) ]}
            conditional=[]
            for step in range(20000,200001,20000):
                file=p/f'step_{step}.json';record=json.loads(file.read_text());b=record['last_rollout_bootstrap']
                phi=np.asarray(record['ongoing_phi']);value=np.asarray(b['values']);done=np.asarray(b['dones'],bool)
                assert len(phi)==len(value)==len(done)==100 and b['pre_update_epoch_counter']==record['ppo_epochs']-10
                conditional.append(dict(step=step,pre_update_epoch=b['pre_update_epoch_counter'],
                    groups={label:dict(n=int(mask.sum()),mean_value=float(value[mask].mean()) if mask.any() else None,
                        std_value=float(value[mask].std()) if mask.any() else None) for label,mask in [
                        ('phi0_ongoing',(phi==0)&~done),('phi_minus10_ongoing',(phi==-10)&~done),('last_transition_done',done)]}))
                sources[str(file.relative_to(S))]=sha(file)
            training=json.loads((p/'episodes.json').read_text())['episodes']
            assert all('r' in r['episode'] and isinstance(r['success'],bool) for r in training)
            results[key]=dict(development_total=96,success=data['summary']['success_count'],overlapping_failure_counts=counts,
                terrain_contact_only_failures=terrain_only,exploratory_height_groups=groups,bootstrap_value_by_phi=conditional,
                training_episodes=len(training),training_success=sum(r['success'] for r in training),
                mean_original_training_return=float(np.mean([r['episode']['r'] for r in training])))
            allrows[key]={r['seed']:(r,f) for r,f in zip(rows,decoded)}
            for name in ['development_final.json','episodes.json']:sources[str((p/name).relative_to(S))]=sha(p/name)
        a=allrows[f'original/{seed}'];b=allrows[f'potential/{seed}'];assert a.keys()==b.keys()
        lost=[];gained=[]
        for case in a:
            left,lf=a[case];right,rf=b[case]
            if left['success'] and not right['success']:lost.append(dict(case=case,failures=[k for k,v in rf.items() if v],terrain=right['scenario']['terrain'],height=right['scenario']['stand_height_m']))
            if not left['success'] and right['success']:gained.append(dict(case=case,removed_failures=[k for k,v in lf.items() if v],terrain=right['scenario']['terrain'],height=right['scenario']['stand_height_m']))
        assert len(gained)-len(lost)==review['pairs'][str(seed)]['success_difference']
        pairs[str(seed)]=dict(lost_count=len(lost),gained_count=len(gained),lost=lost,gained=gained)
    output=dict(verified=True,complete_success_reconstructed_cases=576,results=results,pairs=pairs,
        scope='Failure flags overlap and are not exclusive causes. Height grouping and Phi-conditioned bootstrap are exploratory. Saved values are predictions, not MonteCarlo targets/calibration errors; state/policy distributions differ between arms.',
        constraints='Phi predicate covers physical/design/attitude only, not legacy/terrain contacts, stopping/tail or all height/task requirements. Contact bits keep original broad geometry-contact semantics, not loaded-support certification.',
        next='Use retained model/normalization and a controlled observation-history/value test to examine representability; no new long training or attribution of all loss to credit timing without intervention.',
        input_sha256=sources,source_sha256=sha(Path(__file__)))
    (S/'failure_structure.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    for key,r in results.items():print(key,'success',r['success'],'failures',r['overlapping_failure_counts'],'terrain-only',r['terrain_contact_only_failures'])
    for seed,pair in pairs.items():print('PAIR',seed,'lost/gained',pair['lost_count'],pair['gained_count'])

if __name__=='__main__':run()
