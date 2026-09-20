"""不同腿高跑跳与请求取消回归。"""
import math
import os
import argparse

import mujoco

import wheelleg_sim as sim


XML = os.path.join(os.path.dirname(sim.__file__), '..', 'xml', 'wheelleg_dm8009.xml')
LIMIT = math.radians(5.0)


def run_height(speed, target, hardware=False, track_width=None, six_state=False):
    model, data = sim.load_model(XML, hardware, track_width)
    state = sim.make_state(model, hardware, six_state)
    state.leg_ref = target - sim.L_STAND
    requested = completed = False
    takeoff_speed = 0.0
    squat_length = 0.0
    maxima = [0.0, 0.0, 0.0]
    land_x = None
    tail_x = []
    tail_w = []
    for step in range(int(14.0 / model.opt.timestep)):
        t = step * model.opt.timestep
        state.cmd_vel = speed if t > 1.0 else 0.0
        if (not requested and t > 3.5
                and (abs(speed) < 0.01 or speed * state.vf_f > 0.95)):
            state.cmd_jump = True
            requested = True
        previous = state.jp
        sim.control(model, data, state)
        mujoco.mj_step(model, data)
        if previous == 'DRIVE' and state.jp == 'SQUAT':
            takeoff_speed = state.jp_v_keep
        if previous == 'SQUAT' and state.jp == 'JUMP':
            squat_length = (sim.fk_joints(data.qpos[7], data.qpos[10])['leg_len']
                            + sim.fk_joints(data.qpos[12], data.qpos[15])['leg_len']) / 2
        if previous == 'LAND' and state.jp == 'DRIVE':
            completed = True
            land_x = float(data.qpos[0])
        if land_x is not None and t > 12.0:
            tail_x.append(float(data.qpos[0]))
            tail_w.append(max(abs(state.ws1), abs(state.ws2)))
        if requested:
            roll, pitch, yaw = sim.euler(data)
            yaw_error = math.atan2(math.sin(yaw - state.yaw_target),
                                   math.cos(yaw - state.yaw_target))
            maxima = [max(old, abs(value))
                      for old, value in zip(maxima, (roll, pitch, yaw_error))]
    actual = (sim.fk_joints(data.qpos[7], data.qpos[10])['leg_len']
              + sim.fk_joints(data.qpos[12], data.qpos[15])['leg_len']) / 2
    height = state.jp_peak_z - state.jp_takeoff_z
    speed_ok = abs(speed) < 0.01 or speed * takeoff_speed >= 0.90
    hold_ok = (target > sim.L_SQUAT_MIN or abs(speed) >= 0.01
               or (land_x is not None and bool(tail_x)
                   and abs(float(data.qpos[0]) - land_x) <= 0.02
                   and max(tail_x) - min(tail_x) <= 0.003
                   and max(tail_w) <= 0.055))
    passed = (requested and completed and not state.jp_pending and speed_ok
              and height >= sim.JUMP_HEIGHT and max(maxima) <= LIMIT
              and abs(state.L_cur - target) <= 0.001 and abs(actual - target) <= 0.006
              and hold_ok)
    print(f'v={speed:+.0f} L={target:.3f}: {"PASS" if passed else "FAIL"} '
          f'takeoff={takeoff_speed:.2f}m/s jump={height:.3f}m '
          f'squat={squat_length:.3f}m Lcur/actual={state.L_cur:.3f}/{actual:.3f}m '
          f'r/p/y={tuple(round(math.degrees(v), 2) for v in maxima)}° '
          f'hold={hold_ok}')
    return passed


def cancel_on_release(hardware=False, track_width=None, six_state=False):
    model, data = sim.load_model(XML, hardware, track_width)
    state = sim.make_state(model, hardware, six_state)
    requested = False
    for step in range(int(8.0 / model.opt.timestep)):
        t = step * model.opt.timestep
        state.cmd_vel = 1.0 if 1.0 < t < 3.52 else 0.0
        if not requested and t >= 3.50:
            state.cmd_jump = True
            requested = True
        sim.control(model, data, state)
        mujoco.mj_step(model, data)
    passed = state.jp == 'DRIVE' and not state.jp_pending and state.jp_takeoff_z == 0.0
    print(f'松键取消: {"PASS" if passed else "FAIL"} phase={state.jp} pending={state.jp_pending}')
    return passed


def prepare_then_commit(hardware=False, track_width=None, six_state=False):
    model, data = sim.load_model(XML, hardware, track_width)
    state = sim.make_state(model, hardware, six_state)
    prepared = committed = completed = launched_early = False
    for step in range(int(8.0 / model.opt.timestep)):
        t = step * model.opt.timestep
        if not prepared and t >= 2.0:
            state.cmd_jump_prepare = True
            prepared = True
        if not committed and t >= 3.2:
            state.cmd_jump = True
            committed = True
        previous = state.jp
        sim.control(model, data, state)
        mujoco.mj_step(model, data)
        launched_early |= not committed and state.jp != 'DRIVE'
        completed |= previous == 'LAND' and state.jp == 'DRIVE'
    passed = (completed and not launched_early and state.jp == 'DRIVE'
              and not state.jp_pending)
    print(f'预蹲/提交: {"PASS" if passed else "FAIL"} early={launched_early} '
          f'phase={state.jp} pending={state.jp_pending}')
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--track-width', type=float)
    parser.add_argument('--six-state', action='store_true')
    args = parser.parse_args()
    options = dict(hardware=args.hardware, track_width=args.track_width, six_state=args.six_state)
    assert all([run_height(speed, target, **options)
                for speed in (1.0, -1.0, 0.0)
                for target in (0.090, sim.L_STAND, 0.130)]
               + [run_height(0.0, sim.L_SQUAT_MIN, **options),
                  cancel_on_release(**options), prepare_then_commit(**options)])
