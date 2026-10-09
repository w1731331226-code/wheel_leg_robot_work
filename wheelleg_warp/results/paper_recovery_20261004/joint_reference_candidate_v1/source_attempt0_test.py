"""Decoder/log and GPU-buffer reset checks; no robot environment or physics."""
from types import SimpleNamespace
import numpy as np
import warp as wp
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnv
import joint_reference_adapter as adapter
from joint_reference_mapping import select, LOW


class NoPhysics(VecEnv):
    def __init__(self):
        super().__init__(2, Box(-np.inf, np.inf, (481,), dtype=np.float32), Box(-1., 1., (6,), dtype=np.float32))
        self.last = None
    def reset(self):
        return np.zeros((2, 481), np.float32)
    def step_async(self, actions):
        self.last = actions.copy()
    def step_wait(self):
        return self.reset(), np.zeros(2), np.zeros(2, bool), [{}, {}]
    def close(self):
        pass
    def get_attr(self, name, indices=None):
        return [None, None]
    def set_attr(self, name, value, indices=None):
        raise AssertionError('No physical environment')
    def env_method(self, name, *args, **kwargs):
        raise AssertionError('No physical environment')
    def env_is_wrapped(self, cls, indices=None):
        return [False, False]


def run():
    rng = np.random.default_rng(30131)
    actions = rng.uniform(-1, 1, (128, 3))
    canonical, motor = adapter.decode(actions, 'M3', 128)
    full, full_motor = adapter.decode(canonical, 'U6', 128)
    np.testing.assert_array_equal(canonical, full); np.testing.assert_array_equal(motor, full_motor)
    np.testing.assert_array_equal(motor[:, :4], 0)
    np.testing.assert_array_equal(motor[:, 4], -motor[:, 5])
    leg, _ = adapter.decode(actions[:, :2], 'leg_only', 128)
    yaw, _ = adapter.decode(actions[:, 2:3], 'yaw_only', 128)
    np.testing.assert_array_equal(leg[:, 2:], 0); np.testing.assert_array_equal(yaw[:, :4], 0)
    for arm, value in [('M3', [[1+1e-9, 0, 0]]), ('U6', [[np.nan]*6]),
                       ('M3', [[0, 0]]), ('unknown', [[0, 0, 0]])]:
        try:
            adapter.decode(value, arm, 1)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid action accepted')
    venv = NoPhysics(); requested = []
    raw = SimpleNamespace(_joint_requested=SimpleNamespace(assign=lambda x: requested.append(x.copy())))
    wrapper = adapter.JointReferenceActions(venv, raw, 'M3')
    np.testing.assert_array_equal(wrapper.reset(), np.zeros((2, 481)))
    sample = np.array([[.2, -.3, .4], [-.2, .3, -.4]])
    wrapper.step_async(sample); wrapper.step_wait()
    expected, motors = adapter.decode(sample, 'M3', 2)
    np.testing.assert_array_equal(requested[-1], expected); np.testing.assert_array_equal(venv.last, motors)
    assert wrapper.observation_space.shape == (481,) and wrapper.action_space.shape == (3,)
    wrapper.close()
    # Synthetic complete episode: moving then parking withdrawal, with distinct nominal/selected means.
    n = 4; data = np.zeros((n, len(adapter.COL))); role = np.zeros((n, 11)); phase = np.zeros((n, 6))
    data[:, 0] = 1; data[:, 1] = np.arange(1, n+1); data[:, 2] = .38
    data[:, 10:16] = [.5, .8, 0, 0, .25, -.25]
    data[:, 31] = [.7, .7, 0, 0]; data[:, 8] = 1; data[2:, 9] = 1
    data[:, 16:22] = data[:, 10:16]; data[2:, 16:22] = 0
    filtered, motor_filter = np.zeros(2), np.zeros(6)
    for i in range(n):
        filtered += np.clip(data[i, 16:18]-filtered, -.01, .01)
        motor_request = data[i, 16:22].copy(); motor_request[:2] = 0
        motor_filter += np.clip(motor_request-motor_filter, -.01, .01)
        data[i, 22:24] = filtered; data[i, 24:30] = motor_filter
    data[:, 32] = LOW; data[:, 33] = .38; data[:, 7] = [.115, .115, .16, .16]
    m, d, _ = select(data[:, 2], np.zeros(n), np.column_stack((data[:, 22:24], np.zeros(n))))
    data[:, 3:7] = np.column_stack((m, d, m+d, m-d))
    role[:, :2] = phase[:, :2] = data[:, :2]
    role[:, 2:4] = .38; role[:, 6] = .03; role[:, 7:9] = data[:, 5:7]
    phase[:, 2] = data[:, 31]; phase[:, 3] = data[:, 31] == 0
    phase[:, 4] = .38; phase[:, 5] = data[:, 7]
    adapter.check_log(data, role, phase)
    assert np.any(data[:, 3] != data[:, 2])
    for field in (3, 5, 7, 16, 22, 24):
        bad = data.copy(); bad[0, field] += .001
        try:
            adapter.check_log(bad, role, phase)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'Log corruption accepted at field{field}')
    # Only GPU buffers, no model, controller call, graph or physics integration.
    mask = wp.array([0, 1, 0], dtype=int, device='cuda:0')
    request = wp.ones((3, 6), dtype=float, device='cuda:0')
    effective = wp.ones((3, 6), dtype=float, device='cuda:0')
    doubles = wp.ones((3, 6), dtype=adapter.D, device='cuda:0')
    motor = wp.ones((3, 6), dtype=float, device='cuda:0')
    filtered = wp.ones((3, 2), dtype=adapter.D, device='cuda:0')
    seen = wp.ones(3, dtype=int, device='cuda:0')
    private = wp.ones((3, 18), dtype=adapter.D, device='cuda:0')
    wp.launch(adapter.clear, 3, [mask, request, effective, doubles, motor, filtered, seen, private])
    for value in (request, effective, doubles, motor, filtered, private):
        result = value.numpy(); np.testing.assert_array_equal(result[1], 0)
        np.testing.assert_array_equal(result[[0, 2]], 1)
    np.testing.assert_array_equal(seen.numpy(), [1, 0, 1])
    # A failed factory must restore all process-wide hooks without a robot environment.
    before = (wp.launch, adapter.environment.control_physical_nominal,
              adapter.complete.old.control_physical_nominal, adapter.complete.old.geometry)
    def fail_factory(*args):
        raise RuntimeError('deliberate_no_robot_factory')
    try:
        adapter.instrument(fail_factory, [dict(seed=301001, scenario=dict(solver_iterations=100))], 'virtual6')
    except RuntimeError as error:
        assert str(error) == 'deliberate_no_robot_factory'
    else:
        raise AssertionError('Failed factory did not stop')
    after = (wp.launch, adapter.environment.control_physical_nominal,
             adapter.complete.old.control_physical_nominal, adapter.complete.old.geometry)
    assert before == after
    print('PASS301 128M3/U6 exact embeds, ablations/boundaries/481 wrapper, nominal-vs-selected/parking/filter log,6 corruption checks,GPU masked reset;0controller/physics', flush=True)


if __name__ == '__main__':
    run()
