"""Freeze reference-derived operational yaw stimuli, not new task trajectories."""
import json
import numpy as np
import run_phase_support_qualification as reference
from analyze_complete_contact import moments, unit

ROOT, sha, write = reference.ROOT, reference.sha, reference.write
OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/equal_exposure_recovery_v1'
IDS = [6301001, 6301003, 6301005, 6301007, 6301033, 6301035, 6301037, 6301039]


def run():
    unit(); reference.verify()
    assert not (OUT/'profiles.npz').exists()
    completion = json.loads((reference.OUT/'completion.json').read_text())
    inputs = {}; rows = []; full_profiles = []; normal_profiles = []; rest_profiles = []
    for seed in IDS:
        selected = []
        for job in completion['records']:
            if job['panel'] != 'controlled' or job['law'] != 'B0': continue
            file = reference.OUT/job['path']; assert sha(file) == job['sha256']
            selected += [(file, row) for row in json.loads(file.read_text())['runs'] if row['seed'] == seed]
        assert len(selected) == 1
        file, row = selected[0]; geometry_file = file.parent/'geometry.json'
        g = json.loads(geometry_file.read_text()); trace_file = file.parent/row['complete_trace']['path']
        assert sha(trace_file) == row['complete_trace']['sha256']
        for f in (file, geometry_file, trace_file): inputs[str(f.relative_to(ROOT))] = sha(f)
        with np.load(trace_file, allow_pickle=False) as z:
            trace, contact, meta, steps = [z[n] for n in ('trace', 'contacts', 'meta', 'contact_steps')]
        steps = steps.astype(int); wheel = np.asarray(g['wheel_geom_ids']); bodies = np.asarray(g['geom_body_ids'])
        a, b = contact[:, 2].astype(int), contact[:, 3].astype(int)
        wa, wb = np.isin(a, wheel), np.isin(b, wheel)
        external = (wa & (bodies[b] == 0)) | (wb & (bodies[a] == 0))
        full, normal, rest, _ = moments(contact, meta[steps-1, 3:6], np.where(wb, 1., -1.))
        series = [np.bincount(steps[external]-1, weights=x[external], minlength=len(trace))
                  for x in (full, normal, rest)]
        target_name = 'bump_L' if row['scenario']['height_l'] else 'bump_R'
        target_geom = g['geom_names'].index(target_name)
        target = external & ((a == target_geom) | (b == target_geom))
        load = np.bincount(steps[target]-1, weights=np.maximum(contact[target, 23], 0), minlength=len(trace))
        start = np.flatnonzero(load > 1e-6)[0]
        assert start+200 <= len(trace)
        profile = [x[start:start+200] for x in series]
        np.testing.assert_allclose(profile[0], profile[1]+profile[2], atol=1e-10, rtol=1e-10)
        assert all(np.isfinite(x).all() for x in profile)
        full_profiles.append(profile[0]); normal_profiles.append(profile[1]); rest_profiles.append(profile[2])
        rows.append(dict(profile_index=len(rows), source_case=seed, source_height_m=row['target_leg_m'],
            source_pre_start_s=float(trace[start, 2]), samples=200, dt_s=.0005,
            full_peak_abs_Nm=float(abs(profile[0]).max()), full_signed_impulse_Nms=float(profile[0].sum()*.0005),
            normal_signed_impulse_Nms=float(profile[1].sum()*.0005),
            remainder_signed_impulse_Nms=float(profile[2].sum()*.0005)))
    file = OUT/'profiles.npz'
    np.savez_compressed(file, yaw_moment_Nm=np.array(full_profiles), normal_only_Nm=np.array(normal_profiles),
                        remainder_Nm=np.array(rest_profiles), source_case_ids=np.array(IDS))
    write(OUT/'profile_manifest.json', dict(verified=True, source_sha256=sha(__file__),
        profile_file_sha256=sha(file), input_sha256=inputs, profiles=rows, scale=1.,
        selector='All8 prereview remaining controlled IDs; first200 solver-pre samples from first positive target load, B0 phase reference. No learner result or score chooses profile/sign/scale.',
        stimulus='Full solved external wheel-static yaw moment about whole robot COM, including normal, friction and contact torque; later replay as world-Z pure body torque.',
        limits='Operational reference-derived stimulus; includes reference closed-loop traction. Not a pure natural exogenous disturbance, contact-force replay, terrain-dynamics equivalence or independent test bank.',
        new_task_evaluations=0, training_updates=0))
    print('FROZEN8x200 reference samples,0 task evaluations; peaks', [r['full_peak_abs_Nm'] for r in rows], flush=True)


if __name__ == '__main__':
    run()
