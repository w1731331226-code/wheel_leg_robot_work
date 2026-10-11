"""Describe actual training exposure and accepted reset requests; no training or simulation."""
import json
from collections import Counter
import numpy as np
from train_fixed_force_study import OUT
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def flat(scene):return scene['terrain']=='legacy' and scene['height_l']==scene['height_r']==0


def run():
    assert flat(dict(terrain='legacy',height_l=0,height_r=0))
    assert not flat(dict(terrain='legacy',height_l=.01,height_r=0))
    assert not flat(dict(terrain='ramp',height_l=0,height_r=0))
    destination=OUT/'training_scope_audit.json';assert not destination.exists()
    p=json.loads((OUT/'proposal.json').read_text());review=json.loads((OUT/'training_review.json').read_text())
    admission=json.loads((OUT/'model_evaluation_admission.json').read_text());assert review['verified'] and admission['training_review_sha256']==sha(OUT/'training_review.json')
    diag=OUT.parent/'frozen_wheel_interference_diagnostic_v1';received=json.loads((diag/'review.json').read_text());assert received['verified']
    banks={};runs={};hashes={};total=0;first_packet=None
    for seed in p['seeds']:
        for stage,bank in p['training_banks'][str(seed)].items():
            assert len(bank)==100
            banks[f'{seed}/{stage}']=dict(worlds=100,flat_complete_scenarios=sum(flat(c['scenario']) for c in bank),
                terrain_counts=dict(Counter(c['scenario']['terrain'] for c in bank)),
                positive_speed=sum(c['scenario']['speed']>0 for c in bank),negative_speed=sum(c['scenario']['speed']<0 for c in bank))
        for arm in p['arms']:
            label=f'{arm}_{seed}';directory=OUT/'training'/label;file=directory/'episodes.json'
            assert sha(file)==review['input_sha256'][str(file.relative_to(ROOT))];hashes[str(file.relative_to(ROOT))]=sha(file)
            episodes=json.loads(file.read_text())['episodes'];stages={}
            for row in episodes:
                expected=p['training_banks'][str(seed)][str(row['curriculum_stage'])][row['world_index']]
                assert row['seed']==expected['seed'] and row['scenario']==expected['scenario']
            for stage in (1,2,3):
                part=[r for r in episodes if r['curriculum_stage']==stage]
                stages[str(stage)]=dict(completed_episodes=len(part),success=sum(r['success'] for r in part),
                    physical_steps=sum(r['physical_steps'] for r in part),unique_world_indices=len({r['world_index'] for r in part}))
            metadata=[]
            for step in p['checkpoints']:
                file=directory/f'step_{step}.json';r=next(r for r in review['records'] if r['run']==label and r['step']==step)
                assert sha(file)==r['metadata_sha256'];hashes[str(file.relative_to(ROOT))]=sha(file)
                metrics=json.loads(file.read_text())['training_metrics'];assert all(np.isfinite(v) for v in metrics.values())
                metadata.append(dict(step=step,metrics=metrics))
            initial_file=diag/(label+'_original')/'actor_world_0.npz';request_file=diag/(label+'_original')/'unmasked_requests.npz'
            for f in (initial_file,request_file):assert sha(f)==received['raw_sha256'][str(f.relative_to(ROOT))];hashes[str(f.relative_to(ROOT))]=sha(f)
            with np.load(initial_file) as z:packet=z['trace'][0,:481].copy()
            with np.load(request_file) as z:request=z['trace'][0].copy()
            if first_packet is None:first_packet=packet
            else:np.testing.assert_array_equal(packet,first_packet)
            frames=packet[:390].reshape(10,39)
            assert not frames[:,[0,2,3,5,7,38]].any()
            for left,right in [(12,14),(13,15),(16,18),(17,19),(20,21),(22,23),(24,25),(26,27),(28,29),(32,33),(34,35),(36,37)]:np.testing.assert_array_equal(frames[:,left],frames[:,right])
            assert not packet[390:471].any()
            half=(request[::2]-request[1::2])/2
            runs[label]=dict(stages=stages,completed_episodes=len(episodes),completed_flat_scenario_episodes=sum(flat(r['scenario']) for r in episodes),
                checkpoints_descriptive_not_selected=metadata,initial_unmasked6=request.tolist(),initial_half_difference_F_H_wheel=half.tolist(),
                nonzero_differential_reset_request=bool(np.any(half!=0)))
            total+=len(episodes)
    assert total==review['training_episodes']==2876
    atomic_json(destination,dict(round=344,verified=True,registered_world_entries=900,unique_training_seed_banks=3,completed_training_episodes=total,banks=banks,runs=runs,input_sha256=hashes,
        proposal_sha256=sha(OUT/'proposal.json'),training_review_sha256=sha(OUT/'training_review.json'),diagnostic_review_sha256=sha(diag/'review.json'),auditor_sha256=sha(__file__),
        new_physics_steps=0,new_learning_samples=0,formal5_admitted=False,
        findings=['No wholly flat legacy scenario in these900registered stage/world entries;flatapproach/standingsegments stillexist,so no missing-state exposure claim.',
                  'Identical reset publichistories haveequalpairedmeasurements/zero lateral-yaw features but all3D3 means give nonzero differential requests. Actionantisymmetry doesnot impose policyequivariance orzero nominal response.',
                  'Training completedepisode success andcheckpointEV/KL/std are descriptive of stochastic repeatedbanks;not convergence,cause ofgeneralization failure orpermission toselectcheckpoint.'],
        decision='Keepclosedforce candidate/rewardtuning/broadmasktraining closed.345review whether ONE bounded public-input symmetry/nominal-response audit can distinguish actionprior versus mean-policy bias;mustverify full481 semantics/raw-before-normalization androbot/command symmetry assumptions before anyoperator ornewlearning.',
        limits='No global robotreflection law or fullpolicy symmetry test established. No newcontroller/normalizer/reward/curriculum change. Novelty andeffect mustbe separately defended against known symmetryRL andrelativeCRRL.'))
    print('PASS344900bank entries/2876actual episodes/60metadata/6resetrequests;0physics/learning',flush=True)


if __name__=='__main__':run()
