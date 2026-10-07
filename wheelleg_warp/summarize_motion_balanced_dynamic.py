"""Correct derived monitor fields from immutable raw trajectories; no rerun."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wheelleg_ppo/tools'))
import numpy as np
import mujoco
from native.terrain import HeightTerrainScenario, model
import wheelleg_sim as sim
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/motion_balanced_dynamic_v1'


def run():
    assert not (OUT / 'delivery.json').exists()
    p = json.loads((OUT / 'proposal.json').read_text())
    c = json.loads((OUT / 'completion.json').read_text())
    k = json.loads((OUT / 'source_contract.json').read_text())
    assert c['verified'] and c['primary_physical_steps'] == 80000
    assert c['calibration_physical_steps'] == 800 and len(c['records']) == 40
    assert c['source_contract_sha256'] == sha(OUT / 'source_contract.json')
    assert all(sha(ROOT / n) == h for n, h in k['source_sha256'].items())
    # Native monitor: 31/32 are two actual chain lengths; 34 is loop error.
    # Original runner mistakenly used constant field30 for length and31 for loop.
    records, raw, recovered_sha = [], {}, {}
    models = [model(HeightTerrainScenario(**r['scenario'])) for r in k['cases']]
    initial_q, initial_v, initial_sensor = [], [], []
    for m, r in zip(models, p['moving_references']):
        with np.load(ROOT / r['path']) as z:
            qi, vi = z['q'].astype(np.float32), z['v'].astype(np.float32)
        d = mujoco.MjData(m); d.qpos[:], d.qvel[:] = qi, vi
        mujoco.mj_forward(m, d)
        initial_q.append(qi); initial_v.append(vi); initial_sensor.append(d.sensordata.astype(np.float32))
    initial_q, initial_v, initial_sensor = map(np.array, (initial_q, initial_v, initial_sensor))
    for artifact in c['completed']:
        path = OUT / artifact['path']
        assert sha(path) == artifact['sha256']
        with np.load(path, allow_pickle=False) as z:
            x = {n: z[n].copy() for n in z.files}
        assert x['post_q'].shape == (2000, 10, 17) and x['post_v'].shape == (2000, 10, 16)
        assert all(np.isfinite(v).all() for v in x.values())
        np.testing.assert_array_equal(x['physical_state'][-1, :, 37], 2000.)
        raw[artifact['backend'], artifact['arm']] = x
        if artifact['backend'] == 'CPU':
            # CPU Warp .numpy() exposes a view: the runner assigned post state before copying.
            # Recover each prefix from initial public state and previous preserved post state.
            x['pre_q'] = np.concatenate([initial_q[None], x['post_q'][:-1].astype(np.float32)])
            x['pre_v'] = np.concatenate([initial_v[None], x['post_v'][:-1].astype(np.float32)])
            x['pre_sensor'] = np.concatenate([initial_sensor[None], x['post_sensor'][:-1].astype(np.float32)])
            first_memory = raw['GPU', artifact['arm']]['memory_before'][0].copy()
            gyro = int(models[0].sensor_adr[models[0].sensor('body_gyro').id])
            first_memory[:, 5:7] = initial_sensor[:, gyro:gyro+2]
            first_memory[:, 8] = initial_sensor[:, gyro+2]
            x['memory_before'] = np.concatenate([first_memory[None], x['memory_after'][:-1]])
            recovered = OUT / f"recovered_CPU_{artifact['arm']}_prefix.npz"
            np.savez_compressed(recovered, **{n: x[n] for n in ('pre_q', 'pre_v', 'pre_sensor', 'memory_before')})
            recovered_sha[recovered.name] = sha(recovered)
        for i, r in enumerate(p['moving_references']):
            state = x['physical_state'][-1, i]
            length = float(min(state[31], state[32]))
            pitch = np.arcsin(np.clip(2*(x['post_q'][:, i, 3]*x['post_q'][:, i, 5] -
                                         x['post_q'][:, i, 6]*x['post_q'][:, i, 4]), -1., 1.))
            dofs = [models[i].jnt_dofadr[models[i].joint(n).id] for n in
                    ('alphaL', 'betaL', 'alphaR', 'betaR', 'wheel1', 'wheel2')]
            speeds = x['pre_v'][:, i, dofs]
            bounds = np.array([[sim.hw.torque_limit(float('inf'), float(v), j<4, 0., .0005)[0]
                                for j, v in enumerate(row)] for row in speeds])
            actual_excess = float(np.maximum(abs(x['actual_torque'][:, i])-bounds, 0.).max())
            command_excess = float(np.maximum(abs(x['ctrl'][:, i])-bounds, 0.).max())
            records.append(dict(backend=artifact['backend'], arm=artifact['arm'], world=i,
                height=r['height'], speed=r['speed'], actual_leg_min=length,
                loop_error_max=float(state[34]), eight_joint_margin=float(state[33]),
                active_joint_margin=float(state[38]), actual_torque_excess=actual_excess,
                command_torque_excess=command_excess, physical_samples=int(state[37]),
                physical_design_pass=bool(length>=.1147044660616607 and state[33]>=0 and
                    state[38]>=0 and max(actual_excess, command_excess)<=1e-6),
                body_vx_error_peak=float(abs(x['post_v'][:, i, 0]-r['speed']).max()),
                body_vx_error_rmse=float(np.sqrt(np.mean((x['post_v'][:, i, 0]-r['speed'])**2))),
                wheel_rate_error_peak=float(abs(x['post_v'][:, i, dofs[-2:]] -
                    x['pre_v'][0, i, dofs[-2:]]).max()),
                body_z_drift_peak=float(abs(x['post_q'][:, i, 2]-x['pre_q'][0, i, 2]).max()),
                body_pitch_peak_rad=float(abs(pitch).max()),
                commanded_flow_displacement_error_peak=float(abs(x['post_q'][:, i, 0] -
                    x['pre_q'][0, i, 0]-r['speed']*x['time'][:, i]).max()),
                projection_error_peak=float(x['diag'][:, i, 14].max())))
    for backend in ('CPU', 'GPU'):
        old, new = raw[backend, 'old'], raw[backend, 'candidate']
        for field in ('pre_q', 'pre_v', 'pre_warm', 'memory_before'):
            np.testing.assert_array_equal(old[field][0], new[field][0])
    replay_sha = {}
    for r in c['replay']:
        path = OUT / f"replay_{r['arm']}_{r['world']}.npz"
        with np.load(path) as z: errors = z['errors']
        assert errors.shape == (40, 3)
        assert r['passed'] == bool(errors[:, 0].max()<=5e-5 and errors[:, 1].max()<=.02 and errors[:, 2].all())
        replay_sha[path.name] = sha(path)
    prior = json.loads((OUT / 'preintegration_precision_failure.json').read_text())
    assert prior['primary_physical_steps'] == 0
    prior_fd = sum(line.startswith('reduction ') for line in (OUT / 'preintegration_attempt0.log').read_text().splitlines())
    assert prior_fd == 10
    atomic_json(OUT / 'delivery.json', dict(
        verified=True, summarizer_sha256=sha(__file__), completion_sha256=sha(OUT / 'completion.json'),
        records=records, physical_design_passed=sum(r['physical_design_pass'] for r in records),
        total_trajectories=40, replay_passed=sum(r['passed'] for r in c['replay']),
        replay_array_sha256=replay_sha, primary_steps=80000, replay_steps=800,
        baseline_constructor_transitionFD_total_this_round=prior_fd+c['baseline_constructor_transitionFD_calls'],
        recovered_CPU_prefix_sha256=recovered_sha, recovery_initial_CPU_forward_calls=10,
        additional_queries=0, additional_integration=0, new_training_samples=0,
        summary_correction='Use delivery.records and recovered_CPU_prefix arrays. Runner wrongly used monitor30/31 instead of31/32/34. CPU Warp numpy views also overwrote preq/v/sensor/memory_before and pre_v used in torque monitor; recover from initial F32 public state/fresh CPU sensors and previous post state, memory from shared GPU initializer with actual CPU initial gyro then previous memory_after. Recompute all torque bounds using recovered pre_v. Original raw/metadata retained; no controller or trajectory rerun.',
        production_admitted=False,
        limits='Fixed command nominal equilibrium initial states only; no disturbance/start/stop/continuous bridge/full164/newmethod benefit admission. Independent review pending.'))
    print('PASS271 corrected raw delivery physical/design', sum(r['physical_design_pass'] for r in records),
          '/40; common-input replay', sum(r['passed'] for r in c['replay']), '/20', flush=True)
    for backend in ('GPU', 'CPU'):
        for arm in ('old', 'candidate'):
            rows = [r for r in records if r['backend']==backend and r['arm']==arm]
            print(backend, arm, 'vxpeak', max(r['body_vx_error_peak'] for r in rows),
                  'minimum_actual_leg', min(r['actual_leg_min'] for r in rows), flush=True)


if __name__ == '__main__':
    run()
