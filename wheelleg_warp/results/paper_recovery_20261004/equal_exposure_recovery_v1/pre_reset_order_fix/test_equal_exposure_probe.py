"""Signed force API/static graph, pulse clock and frozen inference admission."""
import gc
import json
import tempfile
from pathlib import Path
import numpy as np
import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import support
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
import equal_exposure_probe as p
from route_state import RouteState
from test_nom_yaw_filter_probe import control_args
from smoke_reward_training import weight_digest


def api_check():
    wp.init(); wp.set_device('cuda:0'); records = []
    for quat in ('1 0 0 0', '.7071067811865476 0 .7071067811865476 0', '.7071067811865476 0 0 .7071067811865476'):
        xml = f'<mujoco><option gravity="0 0 0"/><worldbody><body pos=".3 -.2 1" quat="{quat}"><freejoint/><geom type="box" size=".1 .12 .08" pos=".2 .1 0" mass="2"/></body></worldbody></mujoco>'
        cpu = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(cpu); mujoco.mj_forward(cpu, data)
        model = mjw.put_model(cpu); gpu = mjw.put_data(cpu, data, nworld=1, nconmax=64, njmax=64)
        for force in ((0., 0., 0., 0., 0., 2.), (0., 0., 0., 0., 0., -2.), (2., -3., 4., 5., 6., -7.), (0.,)*6):
            ft = np.zeros((1, cpu.nbody, 6), np.float32); ft[0, 1] = force; gpu.xfrc_applied.assign(ft)
            result = wp.zeros((1, cpu.nv)); support.xfrc_accumulate(model, gpu, result)
            expected = np.zeros(cpu.nv)
            mujoco.mj_applyFT(cpu, data, np.array(force[:3]), np.array(force[3:]), data.xipos[1], 1, expected)
            np.testing.assert_allclose(result.numpy()[0], expected, atol=3e-6, rtol=3e-6)
            records.append(dict(quaternion=quat, world_force_torque=force, CPU_qfrc=expected.tolist(), GPU_qfrc=result.numpy()[0].tolist()))
    return records


def set_sample(raw, step, slot=0):
    state = raw.state.numpy(); state[:, 0] = step; raw.state.assign(state)
    tags, desired, submitted, monitor = raw._force_buffers
    wp.launch(p.set_wrench, raw.num_envs, [slot, raw._force_root, raw.state, raw.active,
        tags, desired, submitted, raw.data.xfrc_applied, monitor])


def static_checks(raw, cases):
    raw.reset(); values = p.profiles(); monitored = []
    protected = [raw.data.qpos, raw.data.qvel, raw.data.sensordata, raw.obs, raw.history,
                 raw.targets, raw.k['state'], raw.k['reference'], raw.nominal_correction]
    # Whole200 sample sequence plus both boundaries, without a task integration.
    for step in range(4999, 5201):
        raw.data.xfrc_applied.fill_(wp.spatial_vector(1., 2., 3., 4., 5., 6.))
        saved = [a.numpy().copy() for a in protected]
        set_sample(raw, step); log = raw._force_buffers[3].numpy()[0]; applied = raw.data.xfrc_applied.numpy()
        for a, b in zip(protected, saved): np.testing.assert_array_equal(a.numpy(), b)
        for w, case in enumerate(cases):
            index = -1 if case['profile'] == 'zero' else case['profile']
            p.check_samples(log[w:w+1], index, raw._force_root, values)
            assert not np.delete(applied[w], raw._force_root, axis=0).any()
        monitored.append(log)
    # Different activation masks cannot retain any applied wrench on an inactive world.
    active = raw.active.numpy(); active[0] = 0; raw.active.assign(active); set_sample(raw, 5000)
    assert not raw.data.xfrc_applied.numpy()[0].any() and not raw._force_buffers[3].numpy()[0, 0].any()
    raw.reset(); raw.command.fill_(.8)
    args = control_args(raw); arrays = [a for a in args if isinstance(a, wp.array)]
    saved = [a.numpy().copy() for a in arrays]
    p.phase.prepare_step(raw, 0, 'phase_support'); altered = list(args); altered[12] = raw._role_buffers[0]
    wp.launch(p.phase.roles.experimental.control_physical_nominal, raw.num_envs, altered, block_dim=32)
    before = [a.numpy().copy() for a in (raw.k['state'], raw.diag, raw.data.ctrl)]
    for a, b in zip(arrays, saved): a.assign(b)
    set_sample(raw, 0); p.phase.prepare_step(raw, 0, 'phase_support')
    wp.launch(p.phase.roles.experimental.control_physical_nominal, raw.num_envs, altered, block_dim=32)
    for a, b in zip(before, (raw.k['state'].numpy(), raw.diag.numpy(), raw.data.ctrl.numpy())):
        np.testing.assert_array_equal(a, b)
    raw.reset()
    assert not raw.data.xfrc_applied.numpy().any() and not raw._force_buffers[3].numpy().any()
    assert not raw.data.time.numpy().any() and not raw.state.numpy()[:, 0].any()
    return len(monitored)*len(cases)


def stream_fixture(case):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp); raw = p.instrument([case], 'virtual6', directory)
        try:
            raw.reset(); ref, role = raw._role_buffers; table, gyro, error, _ = raw._gyro_buffers
            trace, _, count, *_ = raw._complete_buffers
            for step, command in enumerate((0., -1., 0.), 1):
                state = raw.state.numpy(); state[0, 0] = step-1; raw.state.assign(state)
                raw.command.fill_(command); p.phase.prepare_step(raw, 0, 'phase_support')
                args = control_args(raw)
                wp.launch(p.phase.roles.noise.before_control, 1, [0, args[2], args[7], raw.state,
                    args[6], args[5], 0, table.shape[1], gyro, error])
                args[12] = ref
                wp.launch(p.phase.roles.experimental.control_physical_nominal, 1, args, block_dim=32)
                wp.launch(p.phase.roles.finish, 1, [0, raw.diag, role])
                set_sample(raw, step-1)
                wp.launch(p.phase.roles.noise.filtered_measurement, 1, [0, raw.k['state'], gyro])
                wp.launch(p.phase.roles.noise.fresh_measurement, 1, [0, raw.data.sensordata, raw.ids,
                    raw.state, raw.active, table, 0, gyro, error])
                values = np.zeros(trace.shape); values[0, 0, :4] = [1, step, (step-1)*.0005, step*.0005]
                values[0, 0, 43] = command; trace.assign(values); count.assign(np.array([[step, step]], float))
                state[0, 0] = step; raw.state.assign(state)
                if step == 3: raw.done.fill_(5)
                result = raw.step_wait()
                if step == 2:
                    partial = p.preserve_force(raw, directory)
                    assert partial['prefix_rows'] == [2] and not partial['complete_case_ids']
            assert raw._force_frozen == raw._phase_frozen == raw._role_frozen == raw._gyro_frozen == raw._complete_frozen == {0}
            assert result[3][0]['force_delivery']['samples'] == 0
            assert result[3][0]['force_delivery']['complete'] == (case['profile'] == 'zero')
            with np.load(directory/result[3][0]['force_trace']['path'], allow_pickle=False) as z:
                p.check_samples(z['trace'], -1 if case['profile'] == 'zero' else case['profile'], raw._force_root, p.profiles())
                np.testing.assert_array_equal(z['trace'][:, 1], [1, 2, 3])
            raw.data.xfrc_applied.fill_(wp.spatial_vector(1., 2., 3., 4., 5., 6.))
            raw.done.fill_(5); assert 'force_trace' not in raw.step_wait()[3][0]
            assert not raw.data.xfrc_applied.numpy().any()
            raw.reset(); assert not raw._force_frozen and not raw._force_chunks[0]
            assert not raw._force_buffers[3].numpy().any() and not raw.data.xfrc_applied.numpy().any()
        finally: raw.close()


def frozen_inference_check(proposal):
    cases = [c for c in proposal['cases'] if c['profile'] == 'zero']; records = []
    for spec in proposal['controllers']['models']:
        prefix = Path(spec['prefix'])
        assert p.rec.sha(prefix.with_suffix('.zip')) == spec['checkpoint']['checkpoint_sha256']
        assert p.rec.sha(prefix.with_suffix('.pkl')) == spec['checkpoint']['normalization_sha256']
        raw = p.instrument(cases, 'virtual6'); route = RouteState(raw, 'route')
        env = VecNormalize.load(str(prefix.with_suffix('.pkl')), VecCheckNan(route, raise_exception=True))
        env.training = False; env.norm_reward = False
        model = PPO.load(str(prefix.with_suffix('.zip')), device='cuda')
        try:
            obs = env.reset(); assert obs.shape == (5, 39) and model.observation_space.shape == (39,)
            assert model.action_space.shape == raw.action_space.shape == (6,)
            weights = (weight_digest(model), model.num_timesteps, model._n_updates)
            rms = (env.obs_rms.mean.copy(), env.obs_rms.var.copy(), env.obs_rms.count)
            before = (raw.obs.numpy().copy(), raw.history.numpy().copy())
            action = model.predict(obs, deterministic=True)[0]
            assert action.shape == (5, 6) and np.isfinite(action).all() and np.all(abs(action) <= 1)
            raw._force_buffers[0].assign(np.arange(5, dtype=np.int32)); set_sample(raw, 5000)
            np.testing.assert_array_equal(raw.obs.numpy(), before[0]); np.testing.assert_array_equal(raw.history.numpy(), before[1])
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], action)
            assert weights == (weight_digest(model), model.num_timesteps, model._n_updates)
            np.testing.assert_array_equal(rms[0], env.obs_rms.mean); np.testing.assert_array_equal(rms[1], env.obs_rms.var)
            assert rms[2] == env.obs_rms.count
            records.append(dict(label=spec['label'], observation_shape=[5, 39], action_shape=[5, 6],
                weight_digest=weights[0], num_timesteps=weights[1], n_updates=weights[2],
                inference_only=True, RMS_unchanged=True, raw_packet_not_directly_changed_by_force_metadata=True))
        finally: env.close()
    return records


def check():
    proposal = json.loads((p.OUT/'proposal.json').read_text()); cases = proposal['cases']
    api = api_check(); owners = []; samples = 0
    defaults = (wp.launch, mjw.step, p.phase.roles.noise.noise_table, p.rec.old.geometry)
    for mode in ('diff3', 'virtual6'):
        for group in p.phase.broad.batches(cases):
            selected = [c for _, c in group]; raw = p.instrument(selected, mode)
            try:
                assert defaults == (wp.launch, mjw.step, p.phase.roles.noise.noise_table, p.rec.old.geometry)
                assert raw.observation_space.shape == (38,) and raw.action_space.shape == ((3,) if mode == 'diff3' else (6,))
                assert raw._role_buffers[0].shape[1] == 16 and not raw._role_buffers[0].numpy()[:, 10:15].any()
                samples += static_checks(raw, selected)
                owners.append(dict(indices=[i for i, _ in group], force=raw._force_topology,
                    phase=raw._phase_topology, role=raw._role_topology, core=raw._complete_topology,
                    broad=raw._broad_contract))
            finally: raw.close()
            del raw; gc.collect()
            print('CHECKED', mode, [i for i, _ in group], flush=True)
    stream_fixture(cases[0]); stream_fixture(cases[1])
    learned = frozen_inference_check(proposal)
    assert samples == 90*202
    p.rec.atomic_json(p.OUT/'unit.json', dict(verified=True, API_world_COM_CPU_GPU_signed_rotated=api,
        static_registered_worlds=90, static_force_clock_samples=samples, batch_owners=owners,
        source_virtual6_not_inferred_from_diff3=True, raw38_route39_and_frozen6D_models=learned,
        zero_control_state_diag_command_exact=True, force_after_control_before_step=True,
        exact_profile_float32_quantization_all200_samples_and_boundaries=True,
        inactive_zero_reset_done_no_leak=True, five_stream_terminal_and_repeat_done_checked=True,
        incomplete_early_delivery_retained=True, force_prefix_last_buffer_saved=True, globals_restored=True,
        new_task_evaluations=0, training_updates=0,
        limits='CPU/GPU static applied-force Jacobian and control/clock fixtures, constructors and frozen inference; '
            'no recorded task trajectory integration. Does not certify zero controls performance, equal effective motion, '
            'terrain dynamics equivalence or a new trained method.'))
    print('PASS force API/90static worlds/18180clock samples/native6/39/RMS/five streams;0 task evaluations', flush=True)


if __name__ == '__main__':
    check()
