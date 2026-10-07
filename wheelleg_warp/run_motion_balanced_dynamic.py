"""One registered fixed-command batch; retains complete trajectories and failures."""
import json
import time
from pathlib import Path
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
import light_phase_reference as light
import motion_balanced_control as candidate
import motion_balanced_nominal as proto
from native.controller import D
from native.environment import collect_physical
from native.terrain import HeightTerrainScenario, model
from train_height_comparison import raw_env
from query_nominal_compatibility import initialize
from test_nom_yaw_filter_probe import control_args
from probe_height_115_local_states import contact_snapshot
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = proto.OUT.parent / 'motion_balanced_dynamic_v1'


def clone(value):
    return wp.array(value.numpy(), dtype=value.dtype, device='cpu') if isinstance(value, wp.array) else value


def cpu_contacts(m, d, world):
    rows = []
    for i, contact in enumerate(d.contact):
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        rows.append(np.r_[world, contact.geom, contact.dist, contact.pos,
                          contact.frame, f[:3], contact.friction, contact.dim])
    return np.array(rows).reshape(-1, 25)


def run():
    assert not any((OUT / n).exists() for n in ('source_contract.json', 'completion.json', 'failure.json'))
    started = time.monotonic()
    p = json.loads((OUT / 'proposal.json').read_text())
    assert p['primary_physical_steps_maximum'] == 80000 and p['calibration_replay_steps'] == 800
    assert all(sha(ROOT / n) == h for n, h in p['source_sha256'].items())
    q, v, cases = [], [], []
    for i, r in enumerate(p['moving_references']):
        path = ROOT / r['path']
        assert sha(path) == r['sha256']
        with np.load(path, allow_pickle=False) as z:
            q.append(z['q']); v.append(z['v'])
        cases.append(dict(seed=2710000+i, scenario=HeightTerrainScenario(
            speed=r['speed'], mass=7., height_l=0., height_r=0., center=1.8,
            offset=0., mu_l=.8, mu_r=.8, drive_difference=0., delay_ms=0.,
            solver_iterations=100, stand_height_m=r['height']).__dict__))
    q, v = np.array(q, np.float32), np.array(v, np.float32)
    models = [model(HeightTerrainScenario(**r['scenario'])) for r in cases]
    fd_calls = []
    fd = mujoco.mjd_transitionFD
    def counted_fd(*args, **kwargs):
        fd_calls.append(float(args[2]))
        return fd(*args, **kwargs)
    mujoco.mjd_transitionFD = counted_fd
    try:
        raw = light.instrument(raw_env, cases, 'virtual6')
    finally:
        mujoco.mjd_transitionFD = fd
    steps, replay_steps, records, completed = 0, 0, [], []
    trace = {}
    try:
        assert len(fd_calls) == 10
        for i, m in enumerate(models):
            assert m.opt.timestep == .0005 and m.opt.iterations == 100 and int(m.opt.integrator) == 3
            for name in ('body_mass', 'body_pos', 'geom_pos', 'geom_size', 'geom_friction',
                         'jnt_range', 'actuator_gainprm'):
                actual = getattr(raw.model, name).numpy()[i]
                np.testing.assert_array_equal(actual, np.asarray(getattr(m, name), dtype=actual.dtype))
        admission = json.loads((proto.OUT / 'source_admission.json').read_text())
        values = [(np.array(r['feed']), r['theta']) for r in admission['records']]
        command = np.array([r['speed'] for r in p['moving_references']])
        source_names = ['wheelleg_warp/' + n for n in (
            'run_motion_balanced_dynamic.py', 'query_nominal_compatibility.py',
            'test_nom_yaw_filter_probe.py', 'probe_height_115_local_states.py')]
        sources = {**p['source_sha256'], **{n: sha(ROOT / n) for n in source_names}}
        atomic_json(OUT / 'source_contract.json', dict(
            verified=True, proposal_sha256=sha(OUT / 'proposal.json'), source_sha256=sources,
            cases=cases, baseline_transitionFD_calls=fd_calls,
            bank_topology=raw._light_phase_topology,
            geometry_offsets=raw.physical_args[8].numpy().tolist(),
            sensor_convention='Initial fresh forward; each step uses MuJoCo step sensor convention on both backends. CPU queries cast q/v/sensor F32, CPU plant retains MuJoCo F64.',
            contact_convention='Contact/force evaluation is preintegration state; stored after step with both pre and post state. No additional forward mutates warmstart.',
            scope='Actual fixed-command controllers and plants; no command_step/after/auto-reset or shared-reference correction update. Actual phase and physical monitor reused.'))
        for backend in ('GPU', 'CPU'):
            for arm, kernel in (('old', light.phase.roles.experimental.control_physical_nominal),
                                ('candidate', candidate.control_physical_nominal)):
                raw.reset()
                raw.data.qpos.assign(q); raw.data.qvel.assign(v)
                raw.data.qacc_warmstart.zero_(); raw.data.time.zero_()
                raw.data.qfrc_applied.zero_(); raw.data.xfrc_applied.zero_()
                raw.data.ctrl.zero_(); raw.targets.zero_(); raw.nominal_correction.zero_()
                raw.command.assign(command)
                mjw.forward(raw.model, raw.data)
                raw.data.qacc_warmstart.zero_()
                wp.launch(initialize, 10, [raw.data.qpos, raw.data.qvel, raw.ids, raw.k['state'], D(0.)])
                # Initial steady gyro memory must match each backend's actual sensors.
                args = control_args(raw)
                envstate = raw.state
                ref, role, phase = raw._light_phase_buffers
                base = raw.k['reference']
                physical = list(raw.physical_args)
                data = None
                if backend == 'CPU':
                    args = [clone(x) for x in args]
                    envstate, base = clone(envstate), clone(base)
                    ref, role, phase = clone(ref), clone(role), clone(phase)
                    physical = [clone(x) for x in physical]
                    physical[0], physical[3], physical[4], physical[5], physical[9] = args[0], args[14], args[5], args[7], envstate
                    data = [mujoco.MjData(m) for m in models]
                    for m, d, qi, vi in zip(models, data, q, v):
                        d.qpos[:] = qi; d.qvel[:] = vi
                        mujoco.mj_forward(m, d); d.qacc_warmstart[:] = 0.
                    args[2].assign(np.array([d.sensordata for d in data], np.float32))
                gyro = int(raw.ids.numpy()[10])
                mem = args[6].numpy()
                mem[:, 5:7] = args[2].numpy()[:, gyro:gyro+2]
                mem[:, 8] = args[2].numpy()[:, gyro+2]
                args[6].assign(mem)
                device = args[0].device
                private = wp.zeros((10, 21), dtype=D, device=device)
                contact_buffer = wp.zeros((1, raw.data.naconmax, 25), dtype=D)
                contact_rows, trace = [], {n: [] for n in (
                    'pre_q', 'pre_v', 'pre_sensor', 'pre_warm', 'post_q', 'post_v', 'post_sensor',
                    'post_warm', 'time', 'ctrl', 'actual_torque', 'memory_before', 'memory_after',
                    'diag', 'reference', 'role', 'phase', 'physical_state')}
                for step in range(2000):
                    state = envstate.numpy(); state[:, 0] = 4000 + step; state[:, 1] = -1
                    envstate.assign(state)
                    wp.launch(light.phase.roles.prepare, 10, [0, 0, base, args[0], args[2],
                              args[7], args[6], envstate, args[5], ref, role], device=device)
                    wp.launch(light.phase.select_anchor, 10, [0, 1, args[4], args[5],
                              ref, role, phase], device=device)
                    args[12] = ref
                    if arm == 'candidate':
                        private.assign(np.array([proto.private_reference(
                            row, r['height'], r['speed'], *value) for row, r, value in
                            zip(ref.numpy(), p['moving_references'], values)]))
                        args[12] = private
                    preq, prev, sensor = args[0].numpy(), args[1].numpy(), args[2].numpy()
                    warm = raw.data.qacc_warmstart.numpy() if backend == 'GPU' else np.array([d.qacc_warmstart for d in data])
                    before = args[6].numpy()
                    wp.launch(kernel, 10, args, block_dim=32, device=device)
                    wp.launch(light.phase.roles.finish, 10, [0, args[15], role], device=device)
                    ctrl = args[14].numpy()
                    if backend == 'GPU':
                        physical[1].assign(prev)
                        mjw.step(raw.model, raw.data); steps += 10
                        postq, postv, postsensor = raw.data.qpos.numpy(), raw.data.qvel.numpy(), raw.data.sensordata.numpy()
                        postwarm, clock = raw.data.qacc_warmstart.numpy(), raw.data.time.numpy()
                        torque = raw.data.actuator_force.numpy()
                        c = raw.data.contact
                        wp.launch(contact_snapshot, raw.data.naconmax, [
                            0, raw.data.nacon, c.worldid, c.geom, c.pos, c.dist, c.frame, c.friction,
                            c.dim, c.efc_address, raw.data.efc.force, raw.data.njmax,
                            raw.model.opt.cone, contact_buffer])
                        contacts = contact_buffer.numpy()[0]
                        contacts = contacts[contacts[:, 0] >= 0]
                    else:
                        for m, d, u in zip(models, data, ctrl):
                            d.ctrl[:] = u; mujoco.mj_step(m, d); steps += 1
                        postq, postv, postsensor = [np.array([getattr(d, n) for d in data])
                                                    for n in ('qpos', 'qvel', 'sensordata')]
                        postwarm = np.array([d.qacc_warmstart for d in data])
                        clock = np.array([d.time for d in data])
                        torque = np.array([d.actuator_force for d in data])
                        contacts = np.concatenate([cpu_contacts(m, d, i) for i, (m, d) in enumerate(zip(models, data))])
                        args[0].assign(postq.astype(np.float32))
                        args[1].assign(postv.astype(np.float32))
                        args[2].assign(postsensor.astype(np.float32))
                        physical[1].assign(prev); physical[2].assign(torque.astype(np.float32))
                    wp.launch(collect_physical, 10, physical, device=device)
                    current = dict(pre_q=preq, pre_v=prev, pre_sensor=sensor, pre_warm=warm,
                        post_q=postq, post_v=postv, post_sensor=postsensor, post_warm=postwarm,
                        time=clock, ctrl=ctrl, actual_torque=torque, memory_before=before,
                        memory_after=args[6].numpy(), diag=args[15].numpy(),
                        reference=args[12].numpy(), role=role.numpy()[0], phase=phase.numpy()[0],
                        physical_state=envstate.numpy())
                    for name, value in current.items():
                        trace[name].append(value.copy())
                    contact_rows.append(np.c_[np.full(len(contacts), step), contacts])
                    assert np.isfinite(postq).all() and np.isfinite(postv).all()
                    assert not raw.data.overflow.numpy().any()
                    if step % 500 == 499:
                        print(backend, arm, step+1, '/2000; primary steps', steps, flush=True)
                arrays = {n: np.array(x) for n, x in trace.items()}
                arrays['contacts'] = np.concatenate(contact_rows)
                path = OUT / f'{backend}_{arm}.npz'
                np.savez_compressed(path, **arrays)
                completed.append(dict(backend=backend, arm=arm, path=path.name,
                                      sha256=sha(path), worlds=10, steps_per_world=2000))
                for i, r in enumerate(p['moving_references']):
                    final = arrays['physical_state'][-1, i]
                    records.append(dict(backend=backend, arm=arm, world=i,
                        height=r['height'], speed=r['speed'], actual_leg_min=float(final[30]),
                        loop_error_max=float(final[31]), eight_joint_margin=float(final[33]),
                        active_joint_margin=float(final[38]), actual_torque_excess=float(final[35]),
                        command_torque_excess=float(final[36]),
                        physical_design_pass=bool(final[30]>=.1147044660616607 and final[33]>=0 and
                            final[38]>=0 and max(final[35],final[36])<=1e-6),
                        body_vx_error_peak=float(abs(arrays['post_v'][:, i, 0]-r['speed']).max()),
                        end_vx_error=float(arrays['post_v'][-1, i, 0]-r['speed'])))
        replay = []
        for arm in ('old', 'candidate'):
            with np.load(OUT / f'GPU_{arm}.npz', allow_pickle=False) as z:
                rows = z['contacts']
                for i, m in enumerate(models):
                    d = mujoco.MjData(m)
                    d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = z['pre_q'][0, i], z['pre_v'][0, i], z['pre_warm'][0, i]
                    errors = []
                    for step in range(40):
                        d.ctrl[:] = z['ctrl'][step, i]
                        mujoco.mj_step(m, d); replay_steps += 1
                        cpu_pairs = {tuple(sorted(c.geom)) for c in d.contact}
                        contacts = rows[(rows[:, 0] == step) & (rows[:, 1] == i)]
                        gpu_pairs = {tuple(sorted(row[2:4].astype(int))) for row in contacts}
                        errors.append([abs(d.qpos-z['post_q'][step, i]).max(),
                                       abs(d.qvel-z['post_v'][step, i]).max(), cpu_pairs == gpu_pairs])
                    errors = np.array(errors)
                    np.savez_compressed(OUT / f'replay_{arm}_{i}.npz', errors=errors)
                    replay.append(dict(arm=arm, world=i, max_qpos_error=float(errors[:, 0].max()),
                        max_qvel_error=float(errors[:, 1].max()), contact_pairs_all_equal=bool(errors[:, 2].all()),
                        passed=bool(errors[:, 0].max()<=5e-5 and errors[:, 1].max()<=.02 and errors[:, 2].all())))
        assert steps == 80000 and replay_steps == 800
        assert all(sha(ROOT / n) == h for n, h in sources.items())
        atomic_json(OUT / 'completion.json', dict(verified=True, records=records, completed=completed,
            replay=replay, replay_passed=all(r['passed'] for r in replay), primary_physical_steps=steps,
            calibration_physical_steps=replay_steps, total_physical_steps=steps+replay_steps,
            source_contract_sha256=sha(OUT / 'source_contract.json'),
            baseline_constructor_transitionFD_calls=len(fd_calls), new_experimental_transitionFD_calls=0,
            optimization_calls=0, new_training_samples=0, total_seconds=time.monotonic()-started,
            production_admitted=False, limits='Fixed nodes/equilibrium initial states only. Complete task and perturbation qualification not performed; independent review required.'))
        print('DONE27140trajectories; calibration', sum(r['passed'] for r in replay), '/20', flush=True)
    except BaseException as error:
        if trace:
            np.savez_compressed(OUT / 'partial_failure.npz', **{n: np.array(x) for n, x in trace.items()})
        atomic_json(OUT / 'failure.json', dict(error=repr(error), primary_physical_steps=steps,
                   replay_steps=replay_steps, completed=completed, records=records, implicit_retry=False))
        raise
    finally:
        raw.close()


if __name__ == '__main__':
    run()
