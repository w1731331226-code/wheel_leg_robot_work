"""连续跳跃回归：可选择是否叠加有限时长车身碰撞扰动。"""
import argparse
import math
import os

import mujoco
import numpy as np

import wheelleg_sim as sim


LIMIT = math.radians(5.0)
TRANSIENT_LIMIT = math.radians(15.0)
XML = os.path.join(os.path.dirname(sim.__file__), '..', 'xml', 'wheelleg_dm8009.xml')


def run(speed, first_trigger, force, yaw_torque, duration, trace=False,
        actuator_mismatch=0.0, hardware=False, track_width=None, six_state=False):
    model, data = sim.load_model(XML, hardware, track_width)
    state = sim.make_state(model, hardware, six_state)
    for side, scale in (('L', 1.0 + actuator_mismatch),
                        ('R', 1.0 - actuator_mismatch)):
        for stem in ('motor_alpha', 'motor_beta', 'motor_wheel'):
            model.actuator_gainprm[model.actuator(stem + side).id, 0] *= scale
    triggers = [first_trigger + 4.0 * i for i in range(3)]
    next_trigger = 0
    push_until = -1.0
    push_sign = 0.0
    chassis = model.body('chassis').id
    active_until = 0.0
    maxima = [0.0, 0.0, 0.0]
    startup_max = np.zeros(3)
    fell_at = None
    phase_max = {name: [0.0, 0.0, 0.0]
                 for name in ('PREP', 'SQUAT', 'JUMP', 'FLY', 'LAND', 'DRIVE')}
    takeoff_speeds = []
    heights = []
    speed_floor = float('inf')
    speed_floor_at = (0.0, 'DRIVE')
    prep_reached_at = None
    prep_plateaus = []
    completed = 0
    invalid = nonwheel_contact = False
    wheels = {model.geom(n).id for n in ('wheel_collide_L', 'wheel_collide_R')}
    for step in range(int((triggers[-1] + 15.0) / model.opt.timestep)):
        t = step * model.opt.timestep
        state.cmd_vel = speed if t > 1.0 else 0.0
        speed_ready = abs(speed) < 0.01 or speed * state.vf_f >= 0.98
        if (next_trigger < len(triggers) and t >= triggers[next_trigger] and speed_ready
                and state.jp == 'DRIVE' and not state.jp_pending):
            state.cmd_jump = True
            push_sign = -1.0 if next_trigger % 2 else 1.0
            push_until = t + duration
            active_until = t + 3.5
            next_trigger += 1
            prep_reached_at = None
        data.xfrc_applied[chassis] = 0.0
        if t < push_until:
            data.xfrc_applied[chassis, 0] = force * push_sign
            data.xfrc_applied[chassis, 5] = -yaw_torque * push_sign
        previous, was_pending = state.jp, state.jp_pending
        sim.control(model, data, state)
        mujoco.mj_step(model, data)
        invalid |= not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                        and np.isfinite(data.ctrl).all())
        nonwheel_contact |= any(c.geom1 not in wheels and c.geom2 not in wheels for c in data.contact)
        if next_trigger == 0:
            startup_max = np.maximum(startup_max, np.abs(sim.euler(data)))
        prep_target = min(state.jp_L_keep, max(sim.L_SQUAT + 0.020, sim.L_PREP))
        if (state.jp_pending and state.jp_preparing and prep_reached_at is None
                and abs(state.L_cur - prep_target) < 1e-9):
            prep_reached_at = t
        if was_pending and previous == 'DRIVE' and state.jp == 'SQUAT':
            # 在腿长误差门内提前衔接 SQUAT 时，尚未到目标平台，停顿为 0。
            prep_plateaus.append(0.0 if prep_reached_at is None
                                  else t - prep_reached_at)
        if trace and (previous != state.jp or was_pending != state.jp_pending):
            roll, pitch, yaw = sim.euler(data)
            gyro = model.sensor('body_gyro').adr[0]
            qa = [data.qpos[model.joint(name).qposadr[0]]
                  for name in ('alphaL', 'alphaR', 'betaL', 'betaR')]
            print(f'  t={t:.3f} {"PREP" if was_pending else previous}->'
                  f'{"PREP" if state.jp_pending else state.jp} '
                  f'r/p/y={tuple(round(math.degrees(v), 2) for v in (roll, pitch, yaw))} '
                  f'gyro={tuple(round(float(v), 3) for v in data.sensordata[gyro:gyro + 3])} '
                  f'vx/vy={state.vf_f:.2f}/{data.qvel[1]:.2f} '
                  f'wdiff={state.ws1 - state.ws2:.2f} '
                  f'qdiff={qa[0] - qa[1]:.4f}/{qa[2] - qa[3]:.4f}')
        if abs(speed) >= 0.01 and next_trigger > 0:
            speed_now = speed * state.vf_f
            if speed_now < speed_floor:
                speed_floor = speed_now
                speed_floor_at = (t, 'PREP' if state.jp_pending else state.jp)
        if previous == 'SQUAT' and state.jp == 'JUMP':
            takeoff_speeds.append(state.jp_v_keep)
        if previous == 'LAND' and state.jp == 'DRIVE':
            completed += 1
            heights.append(state.jp_peak_z - state.jp_takeoff_z)
        if state.jp != 'DRIVE' or t < active_until:
            roll, pitch, yaw = sim.euler(data)
            yaw_error = math.atan2(math.sin(yaw - state.yaw_target),
                                   math.cos(yaw - state.yaw_target))
            maxima[0] = max(maxima[0], abs(roll))
            maxima[1] = max(maxima[1], abs(pitch))
            maxima[2] = max(maxima[2], abs(yaw_error))
            values = (abs(roll), abs(pitch), abs(yaw_error))
            phase = 'PREP' if state.jp_pending else state.jp
            phase_max[phase] = [max(old, value) for old, value in zip(phase_max[phase], values)]
        if data.qpos[2] < 0.02 or abs(sim.euler(data)[0]) > 0.7 or abs(sim.euler(data)[1]) > 0.7:
            fell_at = t
            break
    speed_ok = (len(takeoff_speeds) == len(triggers)
                and (abs(speed) < 0.01
                     or min(speed * value for value in takeoff_speeds) >= 0.90))
    height_ok = len(heights) == len(triggers) and min(heights) >= sim.JUMP_HEIGHT
    smooth_ok = len(prep_plateaus) == len(triggers) and max(prep_plateaus) <= 0.20
    jump_max = [max(phase_max[phase][axis]
                    for phase in ('SQUAT', 'JUMP', 'FLY', 'LAND'))
                for axis in range(3)]
    disturbed = force > 0.0 or yaw_torque > 0.0
    attitude_ok = (max(maxima) <= LIMIT if not disturbed else
                   max(maxima) <= TRANSIENT_LIMIT and max(jump_max) <= LIMIT)
    passed = (completed == len(triggers) and attitude_ok
              and fell_at is None and max(startup_max) <= LIMIT
              and not invalid and not nonwheel_contact
              and speed_ok and height_ok and smooth_ok
              and (abs(speed) < 0.01 or speed_floor >= 0.80))
    degrees = tuple(math.degrees(value) for value in maxima)
    print(f'v={speed:+.1f}: {"PASS" if passed else "FAIL"} jumps={completed}/3 '
          f'roll/pitch/yaw={degrees[0]:.2f}/{degrees[1]:.2f}/{degrees[2]:.2f}° '
          f'takeoff={tuple(round(v, 2) for v in takeoff_speeds)}m/s '
          f'height={tuple(round(v, 3) for v in heights)}m '
          f'plateau={tuple(round(v, 3) for v in prep_plateaus)}s '
          f'vmin={speed_floor if speed_floor < 10 else 0.0:.2f}m/s@'
          f'{speed_floor_at[0]:.2f}/{speed_floor_at[1]}')
    print(f'  valid={not invalid} nonwheel={nonwheel_contact} fell_at={fell_at} '
          f'startup_r/p/y={tuple(np.degrees(startup_max).round(3))}')
    print('  ' + ' '.join(f'{phase}={tuple(round(math.degrees(v), 1) for v in values)}'
                         for phase, values in phase_max.items()))
    if not takeoff_speeds:
        gyro = model.sensor('body_gyro').adr[0]
        q = [data.qpos[model.joint(name).qposadr[0]]
             for name in ('alphaL', 'alphaR', 'betaL', 'betaR')]
        qd = [data.qvel[model.joint(name).dofadr[0]]
              for name in ('alphaL', 'alphaR', 'betaL', 'betaR')]
        print(f'  blocked gyro={tuple(round(float(v), 4) for v in data.sensordata[gyro:gyro + 3])} '
              f'qdiff={q[0] - q[1]:.5f}/{q[2] - q[3]:.5f} '
              f'qddiff={qd[0] - qd[1]:.4f}/{qd[2] - qd[3]:.4f}')
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-disturbance', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--track-width', type=float)
    parser.add_argument('--six-state', action='store_true')
    parser.add_argument('--actuator-mismatch', type=float, default=0.0,
                        help='left actuator gain +ratio, right -ratio (signed)')
    args = parser.parse_args()
    if not math.isfinite(args.actuator_mismatch) or abs(args.actuator_mismatch) >= 1.0:
        parser.error('actuator mismatch must be finite and have magnitude below 1')
    force, yaw_torque = (0.0, 0.0) if args.no_disturbance else (4.0, 0.1)
    options = dict(hardware=args.hardware, track_width=args.track_width, six_state=args.six_state)
    ok = all((run(1.0, 3.5, force, yaw_torque, 0.05, args.trace,
                  args.actuator_mismatch, **options),
              run(-1.0, 3.75, force, yaw_torque, 0.05, args.trace,
                  args.actuator_mismatch, **options),
              run(0.0, 4.0, force, yaw_torque, 0.05, args.trace,
                  args.actuator_mismatch, **options)))
    raise SystemExit(0 if ok else 1)
