"""Bind existing development/legacy coverage and source compatibility; no rollout."""
import json
from collections import Counter
from pathlib import Path

from reference_role_probe import OUT,rec
from score_reference_learning import load_rows

BASE=OUT.parent
MAIN=BASE/'reference_budget_learning_v1'
LEGACY_PROTOCOL=rec.ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json'
CPU_LEGACY=rec.ROOT/'wheelleg_ppo/tools/results/fixes_2026-09-17/final/baseline.json'


def identity(case):return json.dumps(case['scenario'],sort_keys=True)


def run():
    current=json.loads((OUT/'proposal.json').read_text());main=json.loads((MAIN/'proposal.json').read_text())
    review=json.loads((OUT/'round183_pair_review.json').read_text());assert review['verified'] and review['floor_only_engineering_gate_passed']
    covered={c['seed']:c for c in current['cases']}
    all_control={c['seed']:c for c in main['controlled']}
    assert len(covered)==20 and len(all_control)==40
    assert all(identity(c)==identity(all_control[n]) for n,c in covered.items())
    missing=[c for c in main['controlled'] if c['seed'] not in covered]
    assert len(missing)==20 and len(main['regular'])==96
    protocol=json.loads(LEGACY_PROTOCOL.read_text());legacy=protocol['regression'];cpu=json.loads(CPU_LEGACY.read_text())
    assert len(legacy)==len(cpu['runs'])==28 and cpu['total']==cpu['success_count']==28
    assert [c['seed'] for c in legacy]==[r['name'] for r in cpu['runs']]
    for c,r in zip(legacy,cpu['runs']):
        assert all(c['scenario'][k]==v for k,v in r['scenario'].items()) and c['scenario']['stand_height_m']==.3
    inputs={};references={}
    for label,config in current['classical'].items():
        assert config==main['references'][label]
        for panel in ('regular','controlled'):
            file=MAIN/'classical'/f'{label}_{panel}.json';data=load_rows(file,main[panel])
            inputs[str(file.relative_to(rec.ROOT))]=rec.sha(file)
            references[label+'/'+panel]=dict(path=str(file.relative_to(rec.ROOT)),sha256=rec.sha(file),
                evaluations=len(data['runs']),success=data['summary']['success_count'],reuse='Pending full original-source/config/evaluator/no-op qualification; not automatically accepted by scenario match alone.')
    compatible=[c for c in main['regular'] if c['scenario']['terrain']=='legacy' and c['scenario']['delay_ms']==0]
    result=dict(verified=True,round=184,source_sha256=rec.sha(__file__),current_proposal_sha256=rec.sha(OUT/'proposal.json'),main_proposal_sha256=rec.sha(MAIN/'proposal.json'),
        role_review_sha256=rec.sha(OUT/'round183_pair_review.json'),legacy_protocol_sha256=rec.sha(LEGACY_PROTOCOL),cpu_legacy_sha256=rec.sha(CPU_LEGACY),input_sha256=inputs,
        current_covered_cases=list(covered),regular=main['regular'],remaining_controlled=missing,original_regression=legacy,
        counts=dict(covered_controlled=20,remaining_controlled=20,regular=96,legacy=28,total_unique_registered_IDs=164),
        regular_terrain_counts=dict(Counter(c['scenario']['terrain'] for c in main['regular'])),regular_max_delay_ms=max(c['scenario']['delay_ms'] for c in main['regular']),
        current_noise_table_helper_compatible_regular_cases=len(compatible),source_gap='Existing noise_table asserts legacy and delay0 even when std0. Clean-only broad adapter must preserve original delayed packet and dynamically verify horizon/capacity, not remove delay/terrain or silently crop bank.',
        references=references,
        prospective_scope=dict(status='For185 deep review only, no new execution budget registered',
            candidate_missing_development_evaluations=(96+20)*2,
            current_original_and_candidate_legacy_evaluations=28*2*2,
            proposed_new_total=344,
            reuse='40covered floor_only clean rows + source-qualified original regular/controlled272. CPU28 remains separate historical calibration, not current-GPU paired comparator. Need contemporaneous original/candidate legacy28 for bothfixedlaws.',
            implementation='Floor_only only, original gyroalpha.025, no new reference projection/noise condition. Small homogeneous solver batches, clean table adapter with constructor horizon and sensor/packet delay identity. Preserve full/gyro/role logging and interruptions.',
            gates='Original full task and5deg axes/no lost classical successes/physical-design all pass. Original28 allsuccess plus historic velocity<=old*1.05+.005 androll/pitch<=old+.1; distinguish old-CPU calibration from same-current-Nom effect.',
            exclusions='No original final3000/old gate64 use, no gain/floor/alpha sweeps, no PPO or default baseline replacement. These164 are prior development/regression checks, not new independent generalization.'),
        remaining_learning_space='Known floor_only failures persist at0.115/0.38; broader compatibility unknown. Conventional role correction is not a new method. Strong Nom qualification must precede matched new method and3qualification/5formal trained seeds,ablations,fresh independent ID/combination/OOD andPPO end-to-end/new manuscript.',
        new_evaluations=0,training_updates=0,next='185 deep direction review/cleanup: admit or revise finite broader qualification only after this source gap addressed. Full paper goal remains incomplete.')
    rec.atomic_json(OUT/'round184_qualification_scope.json',result)
    print('PASS coverage20/40,regular96,legacy28 identities;',len(compatible),'regular cases compatible with current helper;',
        '344new proposed for185 review,not admitted/run',flush=True)


if __name__=='__main__':run()
