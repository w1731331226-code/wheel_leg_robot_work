"""Test a necessary-condition hypothesis using archived evidence, no physics."""
import json

from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

BASE = ROOT / 'wheelleg_warp/results/paper_recovery_20261004'
OUT = BASE / 'task_mode_witness_v1'


def universal_clipping_hypothesis(rows):
    failures = [r for r in rows if not r['success']]
    assert failures
    return all(r['sampled_reference_clip_fraction'] > 0 for r in failures)


def run():
    yes = dict(success=False, sampled_reference_clip_fraction=.5)
    no = dict(success=False, sampled_reference_clip_fraction=0.)
    assert universal_clipping_hypothesis([yes])
    assert not universal_clipping_hypothesis([yes, no])
    ap = OUT / 'physical_mode_analysis.json'
    np = BASE / 'round159_shared_task_opportunity.json'
    a, n = json.loads(ap.read_text()), json.loads(np.read_text())
    assert a['verified'] and n['verified']
    assert a['source_sha256'] == sha(ROOT / 'wheelleg_warp/analyze_task_mode_witness.py')
    assert n['source_sha256'] == sha(ROOT / 'wheelleg_warp/audit_shared_task_opportunity.py')
    for name, value in a['input_sha256'].items():
        assert sha(OUT / name) == value
    for name, value in n['dependency_sha256'].items():
        assert sha(ROOT / name) == value
    controller = ROOT / 'wheelleg_warp/native/controller.py'
    text = controller.read_text()
    assert 'wp.clamp(D(.30)*roll+D(.12)*state[w,5],D(-.035),D(.035))' in text
    assert 'offset=wp.clamp(offset,-room,room)' in text
    panels = []
    for label in ('B0', 'B1-route'):
        for ref in n['nominal_reference_contract']['references']:
            h = ref['commanded_height_m']
            rows = [r for r in a['results'] if r['controller'] == label and r['height_m'] == h]
            assert len(rows) == 8 and all(r['terrain'] == 'legacy' for r in rows)
            room = ref['symmetric_roll_offset_room_m']
            if room >= .035:
                assert all(r['sampled_reference_clip_fraction'] == 0 for r in rows)
            panels.append(dict(controller=label, height_m=h, symmetric_room_m=room,
                failures=[r['case'] for r in rows if not r['success']],
                unclipped_failures=[r['case'] for r in rows if not r['success'] and r['sampled_reference_clip_fraction'] == 0],
                clipped_successes=[r['case'] for r in rows if r['success'] and r['sampled_reference_clip_fraction'] > 0]))
    classic = [r for r in a['results'] if r['controller'] in ('B0', 'B1-route') and r['terrain'] == 'legacy']
    assert len(classic) == 80 and not universal_clipping_hypothesis(classic)
    atomic_json(OUT / 'reference_clamp_hypothesis_audit.json', dict(verified=True,
        hypothesis='Every controlled classical failure requires roll-reference room clipping.',
        verdict='Rejected: four 0.16m failures per classical controller occur without this clipping.',
        panels=panels, unique_unclipped_failure_cases=4, paired_unclipped_failure_records=8,
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ap, np, controller)},
        source_sha256=sha(__file__), new_evaluations=0, training_updates=0,
        limits='Not a causal effect estimate. Endpoint clipping may still matter locally; clipped successes do not prove zero effect. Sample fractions are not time durations. Does not certify full passage or change original scores.',
        admission='No common-height repair or PPO admitted from this universal hypothesis. Round170 must choose a justified next experiment; missing passage evidence and new-method gates remain open.'))
    print('PASS: 80 archived controlled records; four unique unclipped failure cases; no new physics/training')


if __name__ == '__main__':
    run()
