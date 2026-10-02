"""Verify factor isolation, exact plan replay, and first design-boundary events."""
from pathlib import Path
import json
import numpy as np
from diagnostic import check, sha, ROOT, OUT, SOURCE


def run():
    check()
    r = json.loads((OUT / 'verification.json').read_text())
    registration = r['registration']
    scenes = registration['scenarios']
    trace = np.load(OUT / 'trace.npz', allow_pickle=False)
    t, u = trace['trace'], trace['extra']
    changes = ({}, dict(mass=7.5), dict(mu_l=.6, mu_r=1.), dict(drive_difference=.03))
    labels = ('nominal', 'mass_only', 'friction_only', 'drive_only')
    assert registration['parameter_groups'] == [g for g in labels for _ in range(6)]
    for group, change in enumerate(changes):
        for case in range(6):
            assert scenes[6*group+case] == dict(scenes[case], **change)
    assert all(s['delay_ms'] == 0 for s in scenes)
    starts = {b['world']: round(b['start_s'] / .0005) for b in r['start_boundaries']}
    plans = [np.load(SOURCE / f'world{w}_best.npz', allow_pickle=False)['schedule'] for w in range(2)]
    for w, selected in enumerate(registration['plan_by_world']):
        if selected < 0:
            assert not u[:, w].any()
            continue
        start = starts[w]
        assert start % 10 == 0
        assert not u[:start//10, w].any()
        active_blocks = int(np.ceil(r['episodes'][w]['physical_steps'] / 10))
        request = u[start//10:active_blocks, w]
        expected = plans[selected][np.minimum(np.arange(len(request)), len(plans[selected])-1)]
        np.testing.assert_array_equal(request, expected)
    events = []
    for w, margin in enumerate(r['active_design_margin_rad']):
        valid = t[:, w, 4] > 0
        excess = abs(t[:, w, :4]).max(axis=1) - 1.4
        bad = np.flatnonzero(valid & (excess > 0))
        assert bool(len(bad)) == bool(margin < 0)
        if not len(bad):
            continue
        k = int(bad[0])
        q = t[k, w, :4]
        nominal = w % 6
        nominal_k = starts[nominal] + k - starts[w]
        assert t[nominal_k, nominal, 4] > 0
        q0 = t[nominal_k, nominal, :4]
        mean_delta = ((q[:2]+q[2:]) - (q0[:2]+q0[2:])) / 2
        difference_delta = ((q[:2]-q[2:]) - (q0[:2]-q0[2:])) / 2
        # Identity checks the common/differential decomposition, not causality
        # or a controller/error bound outside these recorded trajectories.
        np.testing.assert_allclose(q[:2]-q0[:2], mean_delta+difference_delta, rtol=0, atol=1e-15)
        np.testing.assert_allclose(q[2:]-q0[2:], mean_delta-difference_delta, rtol=0, atol=1e-15)
        row = r['episodes'][w]
        events.append(dict(world=w, group=labels[w//6], speed=scenes[w]['speed'],
            first_joint=('alphaL','betaL','alphaR','betaR')[int(abs(q).argmax())],
            first_time_s=(k+1)*.0005, after_arrival_s=(k+1)*.0005-row['arrival_s'],
            first_joint_angles_rad=q.tolist(), same_plan_phase_nominal_joint_angles_rad=q0.tolist(),
            common_joint_delta_rad=mean_delta.tolist(), differential_joint_delta_rad=difference_delta.tolist(),
            peak_design_violation_rad=-margin, stop_distance_m=row['stop_distance_m'], tail_speed_m_s=row['tail_speed_m_s']))
    for name, digest in r['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    output = dict(role='single_factor_and_first_violation_recalculation',
        groups=r['groups'], factor_registration_checked=True, exact_frozen_plan_replay=True,
        complete_physical_evidence_steps=sum(e['physical_steps'] for e in r['episodes']),
        first_violations=events, learning=False, full_admission=False,
        limitations='One fixed value per factor, no interaction attribution or uncertainty bound. Joint differences aligned by known plan phase, not equal whole-plant state. Physical 1.5rad and design 1.4rad remain distinct. First violated design gate does not mean physical failure.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'verification.json',OUT/'trace.npz',OUT/'single_factor_cases.json')},
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target = OUT / 'analysis.json'
    if target.exists():
        assert json.loads(target.read_text()) == output
    else:
        target.write_text(json.dumps(output, indent=2)+'\n')
    print('CHECKED isolated factors, eight exact plan replays, six first violations; full gates 18/24.')


if __name__ == '__main__':
    run()
