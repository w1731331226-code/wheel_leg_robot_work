"""Isolated reference-action adapter; source/capture qualification is required."""
import numpy as np
import warp as wp
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper
import native.environment as environment
import complete_contact_recorder as complete
import execution_input_collector as collector
import parking_withdrawal_probe as parking
import joint_reference_control as candidate
import joint_reference_prepare as reference
from joint_reference_mapping import select
from task_mode_recorder import D, signature

wp.set_module_options({'enable_backward': False})
DIMENSIONS = dict(M3=3, U6=6, leg_only=2, yaw_only=1)
COL = ['valid', 'step', 'nominal_height', 'tracking_mean', 'half_difference',
       'left_target', 'right_target', 'guard_anchor', 'seen_motion', 'withdraw']
COL += [f'{name}_{i}' for name in ('requested', 'effective') for i in range(6)]
COL += ['filtered_mean', 'filtered_difference']
COL += [f'filtered_motor_{i}' for i in range(6)]
COL += ['projection_error', 'current_command', 'length_lower', 'length_upper']
assert len(COL) == 34


def decode(actions, arm, worlds):
    if arm not in DIMENSIONS:
        raise ValueError('Unknown reference arm')
    a = np.asarray(actions, dtype=float)
    if a.shape != (worlds, DIMENSIONS[arm]) or not np.isfinite(a).all() or np.any(abs(a) > 1):
        raise ValueError('Finite declared-size reference actions in [-1,1] required')
    a = a.astype(np.float32)
    canonical = np.zeros((worlds, 6), np.float32)
    if arm == 'U6':
        canonical[:] = a
    elif arm in ('M3', 'leg_only'):
        canonical[:, :2] = a[:, :2]
        if arm == 'M3':
            canonical[:, 4] = a[:, 2]; canonical[:, 5] = -a[:, 2]
    else:
        canonical[:, 4] = a[:, 0]; canonical[:, 5] = -a[:, 0]
    motor = canonical.copy(); motor[:, :2] = 0
    return canonical, motor


class JointReferenceActions(VecEnvWrapper):
    def __init__(self, venv, raw, arm):
        if arm not in DIMENSIONS or venv.action_space.shape != (6,):
            raise ValueError('Reference wrapper needs a registered arm and virtual6 base')
        self.raw, self.arm = raw, arm
        super().__init__(venv, action_space=Box(-1., 1., (DIMENSIONS[arm],), dtype=np.float32))

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        canonical, motor = decode(actions, self.arm, self.num_envs)
        self.raw._joint_requested.assign(canonical)
        self.venv.step_async(motor)

    def step_wait(self):
        return self.venv.step_wait()


@wp.kernel
def convert(effective: wp.array2d[float], doubles: wp.array2d[D], motor: wp.array2d[float]):
    w = wp.tid()
    for j in range(6):
        doubles[w, j] = D(effective[w, j])
        motor[w, j] = effective[w, j]
        if j < 2:
            motor[w, j] = 0.


@wp.kernel
def record(slot: int, state: wp.array2d[D], command: wp.array[D], active: wp.array[int],
           base: wp.array2d[D], private: wp.array2d[D], requested: wp.array2d[float],
           effective: wp.array2d[float], seen: wp.array[int], memory: wp.array2d[D],
           diag: wp.array2d[D], trace: wp.array3d[D]):
    w = wp.tid()
    for j in range(34):
        trace[slot, w, j] = D(0)
    if active[w] == 0:
        return
    trace[slot, w, 0] = D(1); trace[slot, w, 1] = state[w, 0]+D(1)
    trace[slot, w, 2] = base[w, 2]
    trace[slot, w, 3] = (diag[w, 21]+diag[w, 22])/D(2)
    trace[slot, w, 4] = (diag[w, 21]-diag[w, 22])/D(2)
    trace[slot, w, 5] = diag[w, 21]; trace[slot, w, 6] = diag[w, 22]
    trace[slot, w, 7] = base[w, 15]; trace[slot, w, 8] = D(seen[w])
    trace[slot, w, 9] = D(int(seen[w] != 0 and command[w] == D(0)))
    for j in range(6):
        trace[slot, w, 10+j] = D(requested[w, j])
        trace[slot, w, 16+j] = D(effective[w, j])
        trace[slot, w, 24+j] = memory[w, 16+j]
    trace[slot, w, 22] = private[w, 16]; trace[slot, w, 23] = private[w, 17]
    trace[slot, w, 30] = diag[w, 14]; trace[slot, w, 31] = command[w]
    trace[slot, w, 32] = base[w, 3]; trace[slot, w, 33] = base[w, 6]


@wp.kernel
def clear(mask: wp.array[int], requested: wp.array2d[float], effective: wp.array2d[float],
          doubles: wp.array2d[D], motor: wp.array2d[float], filtered: wp.array2d[D],
          seen: wp.array[int], private: wp.array2d[D]):
    w = wp.tid()
    if mask[w] == 0:
        return
    for j in range(6):
        requested[w, j] = 0.
        effective[w, j] = 0.; doubles[w, j] = D(0); motor[w, j] = 0.
    for j in range(2):
        filtered[w, j] = D(0)
    for j in range(18):
        private[w, j] = D(0)
    seen[w] = 0


def check_log(data, role, phase, initial_seen=False):
    if not isinstance(initial_seen, (bool, np.bool_)):
        raise ValueError('Explicit Boolean initial motion latch required')
    assert data.ndim == 2 and data.shape[1] == len(COL) and len(data)
    assert role.shape == (len(data), 11) and phase.shape == (len(data), 6)
    assert all(np.isfinite(x).all() for x in (data, role, phase))
    np.testing.assert_array_equal(data[:, 0], 1)
    np.testing.assert_array_equal(data[:, 1], np.arange(1, len(data)+1))
    np.testing.assert_array_equal(data[:, :2], role[:, :2])
    np.testing.assert_array_equal(data[:, :2], phase[:, :2])
    np.testing.assert_array_equal(data[:, 2], role[:, 2])
    np.testing.assert_array_equal(role[:, 3], role[:, 2])
    np.testing.assert_array_equal(data[:, 5:7], role[:, 7:9])
    np.testing.assert_array_equal(data[:, 7], phase[:, 5])
    parking.phase.check_log(phase, 'phase_support')
    assert np.all(role[:, 10] >= 0) and np.all(role[:, 10] <= role[:, 9]+1e-9)
    np.testing.assert_array_equal(data[:, 8], np.maximum.accumulate((data[:, 31] != 0) | initial_seen))
    withdrawn = (data[:, 8] != 0) & (data[:, 31] == 0)
    np.testing.assert_array_equal(data[:, 9], withdrawn)
    np.testing.assert_array_equal(data[:, 16:22], np.where(withdrawn[:, None], 0, data[:, 10:16]))
    assert np.all(abs(data[:, 10:24]) <= 1)
    previous = np.vstack((np.zeros((1, 2)), data[:-1, 22:24]))
    np.testing.assert_allclose(data[:, 22:24], previous+np.clip(data[:, 16:18]-previous, -.01, .01), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(data[:, 24:26], 0)
    previous_motor = np.vstack((np.zeros((1, 6)), data[:-1, 24:30]))
    desired_motor = data[:, 16:22].copy(); desired_motor[:, :2] = 0
    np.testing.assert_allclose(data[:, 24:30], previous_motor+np.clip(desired_motor-previous_motor, -.01, .01), rtol=0, atol=1e-12)
    assert np.all(abs(data[:, 3]-data[:, 2]) <= .02+1e-12)
    assert np.all(abs(data[:, 4]) <= .035+1e-12)
    assert np.all(data[:, 5:7] >= data[:, 32:33]-1e-12)
    assert np.all(data[:, 5:7] <= data[:, 33:34]+1e-12)
    room = np.maximum(0, np.minimum(data[:, 2]-data[:, 32], data[:, 33]-data[:, 2]))
    d0 = np.clip(role[:, 6], -room, room)
    actions = np.column_stack((data[:, 22:24], np.zeros(len(data))))
    mean, difference, _ = select(data[:, 2], d0, actions)
    np.testing.assert_allclose(data[:, 5:7], np.column_stack((mean+difference, mean-difference)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(data[:, 3], data[:, 5:7].mean(axis=1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(data[:, 4], (data[:, 5]-data[:, 6])/2, rtol=0, atol=1e-12)


def instrument(factory, cases, mode, directory=None):
    if mode != 'virtual6' or not cases or len(cases) > 20 or len({c['scenario']['solver_iterations'] for c in cases}) != 1:
        raise ValueError('Diagnostic reference capture needs1…20 homogeneous-solver virtual6 worlds')
    n = len(cases); device = 'cuda:0'
    requested = wp.zeros((n, 6), dtype=float, device=device)
    effective = wp.zeros((n, 6), dtype=float, device=device)
    doubles = wp.zeros((n, 6), dtype=D, device=device)
    motor = wp.zeros((n, 6), dtype=float, device=device)
    filtered = wp.zeros((n, 2), dtype=D, device=device)
    seen = wp.zeros(n, dtype=int, device=device)
    base = wp.zeros((n, 16), dtype=D, device=device)
    private = wp.zeros((n, 18), dtype=D, device=device)
    role = wp.zeros((40, n, 11), dtype=D, device=device)
    phase = wp.zeros((40, n, 6), dtype=D, device=device)
    park = wp.zeros((40, n, 30), dtype=D, device=device)
    trace = wp.zeros((40, n, len(COL)), dtype=D, device=device)
    mask = wp.ones(n, dtype=int, device=device)
    launch, kernel = wp.launch, environment.control_physical_nominal
    watched, geometry = complete.old.control_physical_nominal, complete.old.geometry
    current = command = None; signatures = []
    def intercept(k, dim, inputs=None, **kwargs):
        nonlocal current, command
        if k is environment.command_step:
            current, command = inputs[0], inputs[2]
        if k is parking.phase.roles.experimental.control_physical_nominal:
            assert current is not None and command is inputs[4]
            slot = len(signatures) % 40
            launch(parking.phase.roles.prepare, dim, [slot, 0, inputs[12], inputs[0], inputs[2],
                inputs[7], inputs[6], current, inputs[5], base, role])
            launch(parking.phase.select_anchor, dim, [slot, 1, command, inputs[5], base, role, phase])
            launch(parking.prepare, dim, [slot, 1, command, inputs[5], current, inputs[6], requested, seen, effective, park])
            launch(convert, dim, [effective, doubles, motor])
            launch(reference.prepare, dim, [base, doubles, inputs[5], filtered, private])
            args = list(inputs); args[3] = motor; args[12] = private
            signatures.append(tuple(signature(x) for x in args))
            result = launch(candidate.control_physical_nominal, dim, args, **kwargs)
            launch(parking.phase.roles.finish, dim, [slot, inputs[15], role])
            launch(record, dim, [slot, current, command, inputs[5], base, private, requested,
                effective, seen, inputs[6], inputs[15], trace])
            return result
        return launch(k, dim, **kwargs) if inputs is None else launch(k, dim, inputs, **kwargs)
    wp.launch = intercept
    environment.control_physical_nominal = parking.phase.roles.experimental.control_physical_nominal
    complete.old.control_physical_nominal = parking.phase.roles.experimental.control_physical_nominal
    complete.old.geometry = parking.phase.broad.geometry
    try:
        raw = collector.instrument(lambda rows, m: complete.instrument(factory, rows, m, directory),
                                   cases, mode, expected_captures=2)
    finally:
        wp.launch, environment.control_physical_nominal = launch, kernel
        complete.old.control_physical_nominal, complete.old.geometry = watched, geometry
    assert len(signatures) == 80 and signatures[:40] == signatures[40:]
    assert all(s[14] == signature(raw.data.ctrl) for s in signatures)
    raw._joint_requested = requested
    raw._joint_control_kernel = candidate.control_physical_nominal
    raw._joint_buffers = (requested, effective, doubles, motor, filtered, seen, base, private, role, phase, park, trace, mask)
    raw._joint_topology = dict(version='joint-reference-v1', captured_controller_calls=80,
        control_steps_per_capture=40, matching_capture_signatures=True, final_ctrl_owner=True,
        nominal_height_separate_from_tracking_mean=True, parking_request_withdrawal_preserved=True,
        clean_gyro_only=True, gyro_filter_alpha=.025, diagnostic_world_limit=20,
        columns=COL, source_capture_qualified=False)
    chunks = [[] for _ in cases]; frozen = set()
    reset, wait = raw.reset, raw.step_wait
    def reset_all():
        result = reset(); mask.fill_(1)
        wp.launch(clear, n, [mask, requested, effective, doubles, motor, filtered, seen, private])
        trace.zero_(); role.zero_(); phase.zero_(); park.zero_()
        frozen.clear()
        for parts in chunks:
            parts.clear()
        return result
    def step_wait():
        result = wait(); frames = raw._complete_buffers[0].numpy()
        values, roles, phases = [b.numpy() for b in (trace, role, phase)]
        for w in range(n):
            if w in frozen:
                continue
            keep = frames[:, w, 0] == 1
            if keep.any():
                chunks[w].append((values[keep, w].copy(), roles[keep, w].copy(), phases[keep, w].copy()))
            if result[2][w]:
                data, r, ph = [np.concatenate([part[i] for part in chunks[w]]) for i in range(3)]
                assert len(data) == result[3][w]['physical_steps']
                if directory is not None:
                    file = directory / f'case_reference_{cases[w]["seed"]}.npz'
                    assert not file.exists()
                    role_columns = list(parking.phase.roles.COL)
                    role_columns[3] = 'nominal_tracking_reference'
                    phase_columns = list(parking.phase.COL); phase_columns[4] = 'nominal_height'
                    np.savez_compressed(file, trace=data, role=r, phase=ph, columns=np.array(COL),
                        role_columns=np.array(role_columns), phase_columns=np.array(phase_columns))
                    result[3][w]['joint_reference_trace'] = dict(path=file.name, sha256=complete.sha(file), rows=len(data))
                check_log(data, r, ph)
                result[3][w]['joint_reference_contract'] = raw._joint_topology
                frozen.add(w); chunks[w].clear()
        if result[2].any():
            mask.assign(result[2].astype(np.int32))
            wp.launch(clear, n, [mask, requested, effective, doubles, motor, filtered, seen, private])
        return result
    raw.reset, raw.step_wait = reset_all, step_wait
    raw._joint_chunks, raw._joint_frozen = chunks, frozen
    return raw
