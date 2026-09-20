"""50 Hz Gym interface, 2 kHz nominal control and residual mapping. No training here."""
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import gymnasium as gym
import mujoco
import numpy as np

import wheelleg_sim as sim
from state_estimation import leg_kinematics

XML = str(Path(__file__).resolve().parents[1] / 'xml/wheelleg.xml')
MODES = {'diff1': 1, 'diff2': 2, 'diff3': 3, 'virtual6': 6, 'torque6': 6}
SPLITS = {'train': 11, 'validation': 23, 'test_iid': 37, 'test_combination': 41, 'test_range': 53}
TORQUE_SCALE = np.array([40., 40., 40., 40., 4.5, 4.5])
STARTUP_RAMP_S = 1.0
STOP_OBSERVATION_S = 2.0


@dataclass(frozen=True)
class Scenario:
    speed: float = 1.0
    mass: float = 7.0
    height_l: float = 0.0
    height_r: float = 0.0
    center: float = 2.0
    offset: float = 0.0
    mu_l: float = 0.8
    mu_r: float = 0.8
    drive_difference: float = 0.0
    delay_ms: float = 0.0
    solver_iterations: int = 100

    def __post_init__(self):
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError('场景参数必须有限')
        if not (.5 <= abs(self.speed) <= 1 and 7 <= self.mass <= 8
                and 0 <= min(self.height_l, self.height_r) <= max(self.height_l, self.height_r) <= .03
                and .4 <= min(self.mu_l, self.mu_r) <= max(self.mu_l, self.mu_r) <= 1.2
                and 1.5 <= self.center <= 2.2 and abs(self.offset) <= .1
                and abs(self.drive_difference) <= .05 and 0 <= self.delay_ms <= 20
                and self.solver_iterations in (50, 100)):
            raise ValueError('场景超出冻结的开发/压力测试范围')
        if not math.isclose(self.delay_ms * 2, round(self.delay_ms * 2), abs_tol=1e-9):
            raise ValueError('延迟必须为0.5 ms整数倍')


def heldout_combination(s):
    return max(s.height_l, s.height_r) >= .016 and abs(s.mu_l - s.mu_r) >= .25


def sample_scenario(split, seed, stage=3):
    """Separate seed namespaces; reserve a joint region, not just new random seeds."""
    if split not in SPLITS or stage not in (0, 1, 2, 3) or (split == 'test_combination' and stage < 2):
        raise ValueError('未知划分或课程阶段')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), SPLITS[split]]))
    for _ in range(10000):
        heights = rng.uniform(0, .02, 2)
        if rng.random() < .5:
            heights[rng.integers(2)] = 0
        mu = rng.uniform(.6, 1., 2) if stage >= 2 else [.8, .8]
        s = Scenario(speed=float(rng.choice([-1, 1]) * rng.uniform(.5, 1)),
                     mass=float(rng.uniform(7, 7.5)) if stage >= 3 else 7.,
                     height_l=float(heights[0]) if stage else 0.,
                     height_r=float(heights[1]) if stage else 0.,
                     center=float(rng.uniform(1.5, 2.2)),
                     offset=float(rng.uniform(-.1, .1)) if stage else 0.,
                     mu_l=float(mu[0]), mu_r=float(mu[1]),
                     drive_difference=float(rng.uniform(-.03, .03)) if stage >= 2 else 0.,
                     delay_ms=float(rng.integers(21) * .5) if stage >= 3 else 0.)
        if heldout_combination(s) == (split == 'test_combination'):
            if split == 'test_range':
                values = asdict(s)
                factor = int(rng.integers(4))
                if factor == 0:
                    values['height_l'] = float(rng.choice([.025, .03]))
                elif factor == 1:
                    values['mass'] = float(rng.uniform(7.6, 8.))
                elif factor == 2:
                    values['mu_l'] = float(rng.choice([.4, 1.2]))
                else:
                    values['delay_ms'] = float(rng.choice([15, 20]))
                s = Scenario(**values)
            return s
    raise ValueError('课程阶段无法生成指定保留集')


def build_model(s):
    spec = mujoco.MjSpec.from_file(XML)
    sim.hw.configure_spec(spec)
    # 7.0 and the summed 6.999999999999999 budget denote the same nominal model.
    # Do not perturb compiled inertia by a rounding-only payload when replaying the baseline.
    if not math.isclose(s.mass, sim.hw.DESIGN_MASS, rel_tol=0, abs_tol=1e-12):
        spec.geom('chassis_lid').mass += s.mass - sim.hw.DESIGN_MASS
    direction = math.copysign(1, s.speed)
    for side, sign, height, mu in (('L', -1, s.height_l, s.mu_l), ('R', 1, s.height_r, s.mu_r)):
        wheel = spec.geom('wheel_collide_' + side)
        # Higher priority selects friction, but must NOT also change contact compliance.
        # Ground and added boxes share the same nominal solref/solimp; preserve their mixture.
        if mu != .8:
            ground = spec.geom('floor')
            weight = wheel.solmix / (wheel.solmix + ground.solmix)
            wheel.solref = weight * wheel.solref + (1 - weight) * ground.solref
            wheel.solimp = weight * wheel.solimp + (1 - weight) * ground.solimp
            wheel.priority = 1
        wheel.friction = (mu, .02, .001)
        if height:
            spec.worldbody.add_geom(name='bump_' + side, type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[direction * (s.center + sign * s.offset / 2), sign * sim.hw.TRACK_WIDTH / 2, height / 2],
                size=[.25, .035, height / 2], friction=[.8, .02, .001])
    model = spec.compile()
    model.opt.iterations = s.solver_iterations
    for side, sign in (('L', 1), ('R', -1)):
        model.actuator_gainprm[model.actuator('motor_wheel' + side).id, 0] *= 1 + sign * s.drive_difference
    return model


class Residual:
    """Normalized targets; identical mapping and output-space diagnostics across modes."""
    def __init__(self, mode='diff3', slew=20.):
        if mode not in MODES or not math.isfinite(slew) or slew <= 0:
            raise ValueError('无效残差模式或变化率')
        self.mode, self.slew = mode, slew
        self.target = np.zeros(MODES[mode])
        self.action = self.target.copy()
        self.base = np.zeros(6)
        self.requested = np.zeros(6)
        self.executed = np.zeros(6)
        self.low, self.high = -TORQUE_SCALE.copy(), TORQUE_SCALE.copy()
        self.lam = 1.
        self.event = None
        self.final_clipped = False

    def set_action(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.target.shape or not np.isfinite(action).all() or np.any(abs(action) > 1 + 1e-7):
            raise ValueError('动作必须是正确维数的[-1,1]有限向量')
        self.target[:] = np.clip(action, -1, 1)

    def map(self, model, data, action):
        force_scale = .1 * sim.hw.DESIGN_MASS * 9.81 / 2
        if self.mode == 'torque6':
            return action * [1., 1., 1., 1., .3, .3]
        if self.mode == 'virtual6':
            virtual = action.reshape(3, 2) * np.array([force_scale, 1., .3])[:, None]
        else:
            a = np.zeros(3)
            if self.mode == 'diff1':
                a[2] = action[0]
            elif self.mode == 'diff2':
                a[[0, 2]] = action
            else:
                a[:] = action
            virtual = (a * [force_scale, 1., .3])[:, None] * [1., -1.]
        out = np.zeros(model.nu)
        for i, side in enumerate(('L', 'R')):
            joints = [model.joint(name + side).id for name in ('alpha', 'beta')]
            q = data.qpos[model.jnt_qposadr[joints]]
            dq = data.qvel[model.jnt_dofadr[joints]]
            jac = leg_kinematics(q, dq)[3]
            if not np.isfinite(jac).all() or np.linalg.cond(jac) > 1e6:
                raise ValueError('无效VMC映射')
            ids = [model.actuator('motor_' + name + side).id for name in ('alpha', 'beta')]
            out[ids] = jac @ virtual[:2, i]
            out[model.actuator('motor_wheel' + side).id] = virtual[2, i]
        return out

    def apply(self, model, data, state, low, high, *, base_infeasible=False):
        self.base[:] = data.ctrl
        self.low[:], self.high[:] = low, high
        self.event = None
        self.action += np.clip(self.target - self.action, -self.slew * model.opt.timestep,
                               self.slew * model.opt.timestep)
        self.requested[:] = 0
        if base_infeasible:
            self.event = 'base_infeasible'
        elif state.jp != 'DRIVE':
            self.event = 'outside_ground_task'
        elif np.any(self.action):
            try:
                self.requested[:] = self.map(model, data, self.action)
            except (ValueError, np.linalg.LinAlgError):
                self.event = 'invalid_mapping'
        positive, negative = self.requested > 0, self.requested < 0
        self.lam = float(np.clip(min(1., np.min((high[positive] - self.base[positive]) / self.requested[positive], initial=1.),
                                    np.min((low[negative] - self.base[negative]) / self.requested[negative], initial=1.)), 0, 1))
        if self.event is not None:
            self.lam = 0.
        self.executed[:] = self.lam * self.requested
        data.ctrl[:] = self.base + self.executed


class WheelLegEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, mode='diff3', split='train', stage=3, scenario=None, max_seconds=None):
        if mode not in MODES or split not in SPLITS or stage not in (0, 1, 2, 3):
            raise ValueError('无效环境配置')
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError('无效时间上限')
        self.mode, self.split, self.stage = mode, split, stage
        self.fixed_scenario, self.max_seconds = scenario, max_seconds
        self.action_space = gym.spaces.Box(-1., 1., (MODES[mode],), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (32,), dtype=np.float32)
        self.model = self.data = None
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and set(options) != {'scenario'}:
            raise ValueError('reset只接受scenario选项')
        self.scenario = ((options or {}).get('scenario') or self.fixed_scenario
                         or sample_scenario(self.split, self.np_random.integers(2**31), self.stage))
        if not isinstance(self.scenario, Scenario):
            raise TypeError('scenario必须为Scenario')
        self.model = build_model(self.scenario)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.keyframe('stand').id)
        mujoco.mj_forward(self.model, self.data)
        self.state = sim.make_state(self.model, True, True)
        self.residual = Residual(self.mode)
        self.dt = float(self.model.opt.timestep)
        self.substeps = round(.02 / self.dt)
        assert math.isclose(self.substeps * self.dt, .02, abs_tol=1e-12)
        self.direction = math.copysign(1, self.scenario.speed)
        self.goal = self.scenario.center + abs(self.scenario.offset) / 2 + .25 + .5
        self.deadline = 1 + STARTUP_RAMP_S / 2 + 1.5 * self.goal / abs(self.scenario.speed)
        self.arrived = self.stop_origin = None
        self.peak = np.zeros(3)
        self.angle_sum = np.zeros(3)
        self.velocity_sum = self.velocity_time = self.stop_distance = self.tail_speed = 0.
        self.steps = self.saturated = self.mapping_failures = 0
        self.torque_saturated = 0
        self.base_infeasible_steps = self.final_clip_steps = 0
        self.peak_torque_rate = np.zeros(6)
        self.peak_actual_torque = np.zeros(6)
        self.last_total = self.last_residual = np.zeros(6)
        self.touched = set()
        self.first_contacts = {}
        self.contact_mu = {}
        self.last_contact = self.recovered = self.recovery_start = None
        self.wheels = {self.model.geom('wheel_collide_' + s).id for s in ('L', 'R')}
        self.bumps = {self.model.geom('bump_' + s).id for s, h in
                      (('L', self.scenario.height_l), ('R', self.scenario.height_r)) if h}
        self.delay_steps = round(self.scenario.delay_ms / 1000 / self.dt)
        obs = self._observation()
        self.history = deque([obs.copy() for _ in range(self.delay_steps + 1)], maxlen=self.delay_steps + 1)
        self.done = False
        self.reason = None
        return obs, {'scenario': asdict(self.scenario), 'split': self.split}

    def _command(self):
        if self.arrived is not None:
            return 0.
        return self.scenario.speed * min(1., max(0., (self.data.time - 1.) / STARTUP_RAMP_S))

    def _observation(self):
        d, c = self.data, self.state.lqr6
        angles = sim.euler(d)
        q, dq = d.qpos[c.qadr], d.qvel[c.vadr]
        legs = [leg_kinematics(q[i], dq[i]) for i in range(2)]
        yaw = angles[2]
        vx = sim.forward_component(d.qvel, yaw)
        body_v = [vx, -math.sin(yaw) * d.qvel[0] + math.cos(yaw) * d.qvel[1], d.qvel[2]]
        wheel_dofs = self.model.jnt_dofadr[self.model.actuator_trnid[c.wheel_actuators, 0]]
        return np.r_[angles, d.sensor('body_gyro').data, body_v, self._command(), vx-self._command(), yaw,
                     q.ravel(), dq.ravel(), d.qvel[wheel_dofs],
                     [np.linalg.norm(x[0]) for x in legs],
                     [legs[i][3][:, 0] @ dq[i] for i in range(2)],
                     self.residual.executed / TORQUE_SCALE].astype(np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('episode结束后必须reset')
        self.residual.set_action(action)
        reward = 0.
        terminated = truncated = False
        for _ in range(self.substeps):
            self.state.cmd_vel = self._command()
            try:
                sim.control(self.model, self.data, self.state, self.residual)
                mujoco.mj_step(self.model, self.data)
                if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
                    raise FloatingPointError('非有限物理状态')
            except (FloatingPointError, np.linalg.LinAlgError):
                self.reason, terminated = 'invalid', True
                reward -= 10.
                break
            self.steps += 1
            t = self.data.time
            angles = np.abs(sim.euler(self.data))
            self.peak = np.maximum(self.peak, angles)
            self.angle_sum += angles**2 * self.dt
            vx = sim.forward_component(self.data.qvel, sim.euler(self.data)[2])
            error = vx - self.state.cmd_vel
            if self.state.cmd_vel:
                self.velocity_sum += error**2 * self.dt
                self.velocity_time += self.dt
            active_bump = False
            nonwheel = False
            for contact in self.data.contact:
                pair = {contact.geom1, contact.geom2}
                nonwheel |= not bool(pair & self.wheels)
                for wid in pair & self.wheels:
                    self.contact_mu[self.model.geom(wid).name] = float(contact.friction[0])
                if pair & self.bumps:
                    active_bump = True
                    for bid in pair & self.bumps:
                        self.touched.add(bid)
                        self.first_contacts.setdefault(self.model.geom(bid).name, t)
            if active_bump:
                self.last_contact, self.recovery_start, self.recovered = t, None, None
            elif self.last_contact is not None and self.recovered is None:
                tolerance = .1 * abs(self.scenario.speed) if self.arrived is None else .03
                if max(angles) <= math.radians(1) and abs(error) <= tolerance:
                    if self.recovery_start is None:
                        self.recovery_start = t
                    if t - self.recovery_start >= .5:
                        self.recovered = self.recovery_start - self.last_contact
                else:
                    self.recovery_start = None
            delta = self.residual.executed / TORQUE_SCALE
            smooth = (delta - self.last_residual) / self.dt
            # Output-space costs, integrated on all physics substeps (same six axes for every mode).
            reward += self.dt * (math.exp(-(error / .25)**2) - np.sum((angles / .08726646)**2)
                                 - .05 * float(delta @ delta) - .0001 * float(smooth @ smooth))
            self.last_residual = delta.copy()
            self.peak_torque_rate = np.maximum(self.peak_torque_rate, abs(self.data.ctrl-self.last_total)/self.dt)
            self.peak_actual_torque = np.maximum(self.peak_actual_torque, abs(self.data.actuator_force))
            self.last_total = self.data.ctrl.copy()
            self.torque_saturated += int(np.any((self.data.ctrl <= self.residual.low+1e-9)
                                                | (self.data.ctrl >= self.residual.high-1e-9)))
            self.saturated += int(self.residual.lam < 1 - 1e-10)
            self.mapping_failures += int(self.residual.event == 'invalid_mapping')
            self.base_infeasible_steps += int(self.residual.event == 'base_infeasible')
            self.final_clip_steps += int(self.residual.final_clipped)
            if self.arrived is None and self.direction * self.data.qpos[0] >= self.goal:
                self.arrived, self.stop_origin = t, self.data.qpos[:2].copy()
            if self.arrived is not None:
                self.stop_distance = max(self.stop_distance, float(np.linalg.norm(self.data.qpos[:2] - self.stop_origin)))
                if t - self.arrived >= STOP_OBSERVATION_S - .5:
                    self.tail_speed = max(self.tail_speed, float(np.linalg.norm(self.data.qvel[:2])))
            obs = self._observation()
            self.history.append(obs)
            if max(angles[:2]) > math.radians(40) or self.data.qpos[2] < .02:
                self.reason, terminated = 'fall', True
            elif nonwheel:
                self.reason, terminated = 'nonwheel_contact', True
            elif self.residual.event == 'invalid_mapping':
                self.reason, terminated = 'invalid_mapping', True
            elif self.arrived is not None and t - self.arrived >= STOP_OBSERVATION_S - 1e-10:
                self.reason, terminated = 'completed', True
            elif self.arrived is None and t >= self.deadline - 1e-10:
                self.reason, truncated = 'timeout', True
            elif self.max_seconds is not None and t >= self.max_seconds - 1e-10:
                self.reason, truncated = 'debug_time_limit', True
            if terminated or truncated:
                break
        self.done = terminated or truncated
        info = self.metrics()
        if self.done:
            reward += 10. if info['success'] else (-10. if self.reason != 'invalid' else 0.)
        return self.history[0].copy(), float(reward), terminated, truncated, info

    def metrics(self):
        elapsed = max(self.steps * self.dt, self.dt)
        rmse = math.sqrt(self.velocity_sum / self.velocity_time) if self.velocity_time else None
        success = bool(self.reason == 'completed' and self.bumps <= self.touched
                       and max(self.peak) <= math.radians(5) and rmse is not None
                       and rmse <= .2 * abs(self.scenario.speed)
                       and self.stop_distance <= .60 and self.tail_speed <= .03)
        return dict(success=success, reason=self.reason, physical_steps=self.steps,
                    duration_s=float(self.data.time), arrival_s=self.arrived,
                    peak_deg=np.degrees(self.peak).tolist(),
                    rms_deg=np.degrees(np.sqrt(self.angle_sum / elapsed)).tolist(),
                    velocity_rmse=rmse, velocity_window_s=self.velocity_time,
                    stop_distance_m=self.stop_distance, tail_speed_m_s=self.tail_speed,
                    residual_saturation_fraction=self.saturated / max(1, self.steps),
                    motor_saturation_fraction=self.torque_saturated / max(1,self.steps),
                    peak_actual_torque_Nm=self.peak_actual_torque.tolist(),
                    peak_command_slew_Nm_s=self.peak_torque_rate.tolist(),
                    last_requested_residual_Nm=self.residual.requested.tolist(),
                    last_executed_residual_Nm=self.residual.executed.tolist(),
                    last_lambda=self.residual.lam,
                    base_infeasible_steps=self.base_infeasible_steps, final_clip_steps=self.final_clip_steps,
                    mapping_failures=self.mapping_failures, recovery_s=self.recovered,
                    first_contacts=self.first_contacts.copy(), actual_contact_mu=self.contact_mu.copy())
