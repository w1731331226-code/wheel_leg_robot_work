"""Register independent fixed-model channel audit; no training or evaluation."""
from pathlib import Path
from dataclasses import replace
import json
from train_height_comparison import protocol,entry
from native.terrain import sample_height_terrain_115
from training_contract import digest
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
S=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reward_pilot_v1'
OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/channel_validation_v1'

def run():
    base=protocol(BASE);review=json.loads((S/'pair_review.json').read_text());assert review['completed_runs']==6 and review['continuation_gate'] is False
    OUT.mkdir(parents=True,exist_ok=False)
    models=[]
    for seed in [1609,1610,1611]:
        for arm in ['original','potential']:
            p=S/'runs'/arm/str(seed)/'step_200000';r=json.loads(p.with_suffix('.json').read_text())
            assert digest(p.with_suffix('.zip'))==r['checkpoint']['checkpoint_sha256']
            assert digest(p.with_suffix('.pkl'))==r['checkpoint']['normalization_sha256']
            models.append(dict(seed=seed,arm=arm,prefix=str(p),checkpoint=r['checkpoint']))
    regular=[entry(i,sample_height_terrain_115(i,3,'development')) for i in range(4400000,4400096)]
    pressure={k:[] for k in ['mass','friction','drive','delay','obstacle_height','combination']}
    for j,kind in enumerate(pressure):
        for k in range(8):
            i=4420000+j*100+k;s=sample_height_terrain_115(i,3,'development')
            if kind=='mass':s=replace(s,mass=7.6+.4*k/7)
            elif kind=='friction':s=replace(s,mu_l=.4,mu_r=1.2)
            elif kind=='drive':s=replace(s,drive_difference=(-1 if k%2 else 1)*.05)
            elif kind=='delay':s=replace(s,delay_ms=15. if k%2 else 20.)
            elif kind=='obstacle_height':s=replace(s,terrain='legacy',height_l=.025 if k%2 else .03,height_r=0.,grade_deg=0.,roughness_m=0.,step_height_m=0.,relative_attitude=False)
            else:s=replace(s,mass=8.,mu_l=.4,mu_r=1.2,drive_difference=.05,delay_ms=20.)
            pressure[kind].append(entry(i,s))
    old=base['regression'];assert len(old)==28
    masks={'all':[1,1,1],'legs_only':[1,1,0],'wheel_only':[0,0,1],'zero':[0,0,0]}
    fixed=json.loads((S/'proposal.json').read_text())['fixed_classical']
    atomic_json(OUT/'protocol.json',dict(version='channel-confirmation-v1',role='New independent fixed-controller diagnostic, not reopening failed reward-benefit gates',
        claim='Estimate closed-loop task effect of removing learned wheel residual in the six fixed200k controllers, retaining physical constraints and original success definition',
        primary_estimand='Per model/scenario success(legs_only)-success(all), report all six model effects and3 seed clusters on fresh regular96',
        secondary='Wheel-only,zero and fixed strong B0/B1; yaw/physical/design/contact and failure transitions, no metric switching',
        models=models,masks=masks,regular=regular,pressure=pressure,legacy_regression=old,fixed_classical=fixed,
        evaluation_episodes_budget=(6*4+2)*(96+48+28),policy_training_steps=0,
        matching='Same controller checkpoint, same parameter scenario,0/1 request mask; policy adapts to changed action history in closed loop. FreshGPU arithmetic not bitwise paired trajectories.',
        rules='All fixed models/scenarios included; no weights/gains/mask-amplitude/model selection. Keep every failed case. Pressure,regular,old regression reported separately; old regression is reused/nonindependent.',
        independence='Fresh440/442 namespaces generated before any score, disjoint from pilot development/old64/old3000; inference conditional on fixed models and finite cases, not new training-seed significance.',
        failure_policy='Stop and preserve on runtime/source/budget failure; no silent retry, reconstruction or altered cases. Diagnostic outcomes never replace old or pilot admission result.',
        publication_scope='Empirical execution/channel/task-observation audit candidate; not a new PPO/PBRS algorithm, proof of global safety, all-PBRS harm or learned-policy superiority.',
        old_protocol_sha256=digest(BASE/'protocol.json'),pilot_review_sha256=digest(S/'pair_review.json'),source_sha256=digest(Path(__file__))))
    print('REGISTERED6models x4masks plus2classics; regular96/pressure48/old28;4472episodes, no replay',flush=True)

if __name__=='__main__':run()
