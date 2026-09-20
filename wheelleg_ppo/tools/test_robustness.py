"""车身物理碰撞回归：不改 qpos/qvel，只施加有限时长 xfrc_applied。"""
import argparse
import math
import os
import json
from pathlib import Path

import mujoco
import numpy as np

import wheelleg_sim as sim


XML = os.path.join(os.path.dirname(sim.__file__), '..', 'xml', 'wheelleg.xml')
PUSH_AT = 3.0
TRANSIENT_LIMIT = math.radians(15.0)


def run_turn(speed, sign, hardware=False, track_width=None, six_state=False, jump=False):
    """实际差速转到 ±90°，保持直行，再松键停车；不重写姿态或速度。"""
    model, data = sim.load_model(XML, hardware, track_width)
    state = sim.make_state(model, hardware, six_state)
    target = sign * math.pi / 2
    released = None
    stop_origin = None
    peak = np.zeros(2)
    heading_error = slide = tail_speed = speed_error = 0.0
    min_forward_speed = None
    invalid = nonwheel = fell = False
    completed = False
    scored_speed = scored_heading = scored_tail = 0
    wheels = {model.geom(n).id for n in ('wheel_collide_L', 'wheel_collide_R')}
    for step in range(round(13 / model.opt.timestep)):
        t = step * model.opt.timestep
        yaw = sim.euler(data)[2]
        if released is None and sign * yaw >= math.pi / 2:
            released = t
        state.cmd_turn = sign * 1.5 if 2 <= t < 8 and released is None else 0.0
        state.cmd_vel = speed if 1 <= t < 10 else 0.0
        state.cmd_jump = jump and step == round(6 / model.opt.timestep)
        if t >= 10 and stop_origin is None:
            stop_origin = data.qpos[:2].copy()
        previous = state.jp
        sim.control(model, data, state)
        completed |= previous == 'LAND' and state.jp == 'DRIVE'
        invalid |= not np.isfinite(data.ctrl).all() or bool(np.any(
            (data.ctrl < model.actuator_ctrlrange[:, 0] - 1e-9)
            | (data.ctrl > model.actuator_ctrlrange[:, 1] + 1e-9)))
        mujoco.mj_step(model, data)
        invalid |= not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        nonwheel |= any(c.geom1 not in wheels and c.geom2 not in wheels for c in data.contact)
        roll, pitch, yaw = sim.euler(data)
        peak = np.maximum(peak, np.abs([roll, pitch]))
        forward = math.cos(yaw) * data.qvel[0] + math.sin(yaw) * data.qvel[1]
        if speed != 0 and 2 <= t < 10:
            value = math.copysign(1, speed) * forward
            min_forward_speed = value if min_forward_speed is None else min(min_forward_speed, value)
        if 9 <= t < 10:
            scored_speed += 1
            speed_error = max(speed_error, abs(forward - speed))
        if released is not None:
            scored_heading += 1
            heading_error = max(heading_error, abs(math.atan2(
                math.sin(yaw - target), math.cos(yaw - target))))
        if stop_origin is not None:
            slide = max(slide, float(np.linalg.norm(data.qpos[:2] - stop_origin)))
        if t >= 12:
            scored_tail += 1
            tail_speed = max(tail_speed, float(np.linalg.norm(data.qvel[:2])))
        fell |= max(abs(roll), abs(pitch)) > math.radians(40) or data.qpos[2] < 0.02
        if fell or invalid:
            break
    jump_height = max(0.0, state.jp_peak_z - state.jp_takeoff_z)
    jump_ok = not jump or (completed and jump_height >= sim.JUMP_HEIGHT
                          and (speed == 0 or speed * state.jp_v_keep >= 0.90))
    passed = bool(released is not None and released < 8 and not (fell or invalid or nonwheel)
                  and scored_speed and scored_heading and scored_tail and jump_ok
                  and (speed == 0 or (min_forward_speed is not None and min_forward_speed >= 0.80))
                  and max(peak) <= math.radians(5) and heading_error <= math.radians(5)
                  and speed_error <= 0.10 and slide <= 0.60 and tail_speed <= 0.03)
    report = dict(scenario='turn', speed=speed, sign=sign, hardware=hardware,
                  wheel_radius=sim.hw.WHEEL_RADIUS if hardware else .025,
                  track_width=track_width, six_state=six_state, reached_at=released,
                  peak_roll_pitch_deg=np.degrees(peak).tolist(),
                  heading_error_deg=math.degrees(heading_error) if scored_heading else None,
                  speed_error=speed_error if scored_speed else None,
                  stop_distance=slide if stop_origin is not None else None,
                  tail_speed=tail_speed if scored_tail else None, fell=bool(fell),
                  jump=jump, jump_completed=bool(completed), jump_height=jump_height,
                  takeoff_speed=state.jp_v_keep,
                  min_forward_speed=min_forward_speed,
                  invalid=bool(invalid), nonwheel=bool(nonwheel), passed=passed)
    print(json.dumps(report), flush=True)
    return report


def run_terrain(speed, slope, height, side, hardware=False, track_width=None, six_state=False):
    """固定横坡或单轮实体凸台；地形仅影响物理接触，不传入控制器。"""
    width = track_width if track_width is not None else (sim.hw.TRACK_WIDTH if hardware else 0.108)
    direction = 1 if speed >= 0 else -1
    spec = mujoco.MjSpec.from_file(XML)
    if height:
        spec.worldbody.add_geom(name='single_wheel_bump', type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[2 * direction, side * width / 2, height / 2],
                                size=[0.25, 0.035, height / 2], friction=[0.8, 0.02, 0.001])
    angle = math.radians(slope)
    if slope:
        spec.geom('floor').quat = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]
    if hardware:
        sim.hw.configure_spec(spec, track_width)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe('stand').id)
    mujoco.mj_forward(model, data)
    if height:
        wheel = model.geom('wheel_collide_L' if side < 0 else 'wheel_collide_R').id
        assert math.isclose(data.geom_xpos[wheel, 1], side * width / 2, abs_tol=1e-6), '凸台未对齐目标轮'
    # 在水平名义工作点生成同一 K；不能根据测试地形重新整定。
    design_model = sim.load_model(XML, hardware, track_width)[0] if height or slope else model
    state = sim.make_state(design_model, hardware, six_state)
    assert [model.actuator(i).name for i in range(model.nu)] == [
        design_model.actuator(i).name for i in range(design_model.nu)]
    if slope:
        floor = model.geom('floor').id
        # 仅初始放置提高，避免高侧轮胎嵌入斜面；运行后不重写 qpos/qvel。
        data.qpos[2] += width / 2 * abs(math.tan(angle)) + 0.002
        mujoco.mj_forward(model, data)
        assert np.allclose(data.geom_xmat[floor].reshape(3, 3)[:, 2],
                           [0, -math.sin(angle), math.cos(angle)])
    wheels = {model.geom(n).id for n in ('wheel_collide_L', 'wheel_collide_R')}
    bump = model.geom('single_wheel_bump').id if height else -1
    peak = np.zeros(3)
    startup_peak = np.zeros(2)
    invalid = nonwheel = fell = touched = False
    stop_origin = None
    slide = tail_speed = 0.0
    penetration = 0.0
    min_forward_speed = None
    for step in range(round(7 / model.opt.timestep)):
        t = step * model.opt.timestep
        state.cmd_vel = speed if 1 <= t < 5 else 0.0
        if t >= 5 and stop_origin is None:
            stop_origin = data.qpos[:2].copy()
        sim.control(model, data, state)
        invalid |= not np.isfinite(data.ctrl).all() or bool(np.any(
            (data.ctrl < model.actuator_ctrlrange[:, 0] - 1e-9)
            | (data.ctrl > model.actuator_ctrlrange[:, 1] + 1e-9)))
        mujoco.mj_step(model, data)
        invalid |= not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        nonwheel |= any(c.geom1 not in wheels and c.geom2 not in wheels for c in data.contact)
        touched |= any(bump in (c.geom1, c.geom2) for c in data.contact)
        for c in data.contact:
            if bump in (c.geom1, c.geom2):
                penetration = max(penetration, -float(c.dist))
        angles = np.abs(sim.euler(data))
        if speed != 0 and 2 <= t < 5:
            forward = math.copysign(1, speed) * sim.forward_component(data.qvel, sim.euler(data)[2])
            min_forward_speed = forward if min_forward_speed is None else min(min_forward_speed, forward)
        startup_peak = np.maximum(startup_peak, angles[:2])
        if t >= 1:
            peak = np.maximum(peak, angles)
        if stop_origin is not None:
            slide = max(slide, float(np.linalg.norm(data.qpos[:2] - stop_origin)))
        if t >= 6:
            tail_speed = max(tail_speed, float(np.linalg.norm(data.qvel[:2])))
        fell |= max(angles[:2]) > math.radians(40) or data.qpos[2] < 0.02
        if fell or invalid:
            break
    traversed = not height or (touched and direction * data.qpos[0] > 2.5)
    passed = bool(t >= 7 - 2 * model.opt.timestep and not (fell or invalid or nonwheel)
                  and max(peak) <= math.radians(5) and traversed
                  and slide <= 0.60 and tail_speed <= 0.03)
    report = dict(scenario='terrain', speed=speed, slope_deg=slope, bump_height=height,
                  wheel_radius=sim.hw.WHEEL_RADIUS if hardware else .025,
                  side=side, hardware=hardware, track_width=width, six_state=six_state,
                  peak_roll_pitch_yaw_deg=np.degrees(peak).tolist(),
                  startup_peak_roll_pitch_deg=np.degrees(startup_peak).tolist(),
                  touched_bump=bool(touched), traversed=bool(traversed), final_time=t,
                  max_bump_penetration=penetration, min_forward_speed=min_forward_speed,
                  final_position=data.qpos[:3].tolist(), stop_distance=slide,
                  tail_speed=tail_speed, fell=bool(fell), invalid=bool(invalid),
                  nonwheel=bool(nonwheel), passed=passed)
    print(json.dumps(report), flush=True)
    return report


def run(speed, direction, sign, force_impulse, yaw_impulse, duration,
        hardware=False, track_width=None, six_state=False):
    model, data = sim.load_model(XML, hardware, track_width)
    state = sim.make_state(model, hardware, six_state)
    chassis = model.body('chassis').id
    push_end = PUSH_AT + duration
    settle_limit = math.radians(3.0 if speed == 0.0 else 5.0)
    peak = [0.0, 0.0, 0.0]
    last_unsettled = push_end
    push_x = None
    fell = invalid_output = False
    nonwheel_contact = False
    wheels = {model.geom(n).id for n in ('wheel_collide_L', 'wheel_collide_R')}

    for step in range(int((push_end + 2.5) / model.opt.timestep)):
        t = step * model.opt.timestep
        state.cmd_vel = speed if t > 1.0 else 0.0
        data.xfrc_applied[chassis] = 0.0
        if PUSH_AT <= t < push_end:
            if push_x is None:
                push_x = float(data.qpos[0])
            channel = 5 if direction == 'yaw' else (0 if direction == 'x' else 1)
            impulse = yaw_impulse if direction == 'yaw' else force_impulse
            data.xfrc_applied[chassis, channel] = sign * impulse / duration

        sim.control(model, data, state)
        for aid, value in enumerate(data.ctrl):
            low, high = model.actuator_ctrlrange[aid]
            invalid_output |= value < low - 1e-9 or value > high + 1e-9
        mujoco.mj_step(model, data)
        invalid_output |= not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                               and np.isfinite(data.ctrl).all())
        nonwheel_contact |= any(c.geom1 not in wheels and c.geom2 not in wheels for c in data.contact)

        roll, pitch, yaw = sim.euler(data)
        if t >= PUSH_AT:
            values = (abs(roll), abs(pitch), abs(yaw))
            peak = [max(old, value) for old, value in zip(peak, values)]
            if t >= push_end and max(values) > settle_limit:
                last_unsettled = t
        fell |= data.qpos[2] < 0.02 or max(abs(roll), abs(pitch)) > math.radians(40.0)

    recovery = max(0.0, last_unsettled - push_end)
    displacement = float(data.qpos[0]) - push_x
    passed = (not fell and not invalid_output and not nonwheel_contact and max(peak) <= TRANSIENT_LIMIT
              and recovery <= 2.0)
    angles = '/'.join(f'{math.degrees(value):.1f}' for value in peak)
    print(f'v={speed:+.1f} {direction}{sign:+d}: {"PASS" if passed else "FAIL"} '
          f'peak r/p/y={angles}° recovery={recovery:.2f}s dx={displacement:+.2f}m '
          f'nonwheel={nonwheel_contact} valid={not invalid_output}')
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-impulse', type=float, default=0.2, help='N·s')
    parser.add_argument('--yaw-impulse', type=float, default=0.02, help='N·m·s')
    parser.add_argument('--duration', type=float, default=0.05, help='s')
    parser.add_argument('--speeds', type=float, nargs='+', default=(0.0, 1.0), help='m/s')
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--track-width', type=float)
    parser.add_argument('--six-state', action='store_true')
    parser.add_argument('--scenario', choices=('push', 'turn', 'slope', 'bump'), default='push')
    parser.add_argument('--bump-heights', type=float, nargs='+', default=(0.020, 0.025))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--jump-after-turn', action='store_true')
    args = parser.parse_args()
    if not all(math.isfinite(h) and h > 0 for h in args.bump_heights):
        parser.error('--bump-heights 必须为有限正数（米）')
    if (args.track_width is not None or args.six_state) and not args.hardware:
        parser.error('--track-width/--six-state 需要 --hardware')
    if args.jump_after_turn and args.scenario != 'turn':
        parser.error('--jump-after-turn 需要 --scenario turn')
    if args.report and args.scenario == 'push':
        parser.error('--report 用于 turn/slope/bump 场景；push 保留文本报告')
    if (not all(math.isfinite(v) and v > 0 for v in
                (args.force_impulse, args.yaw_impulse, args.duration))
            or not all(math.isfinite(v) for v in args.speeds)):
        parser.error('impulse/duration must be finite and positive; speeds must be finite')

    if args.scenario in ('turn', 'slope', 'bump'):
        for yaw in (0.0, math.pi / 2, -math.pi / 2, math.pi):
            assert math.isclose(sim.forward_component([math.cos(yaw), math.sin(yaw)], yaw), 1)
            assert abs(sim.forward_component([-math.sin(yaw), math.cos(yaw)], yaw)) < 1e-12
        if args.scenario == 'turn':
            reports = [run_turn(speed, sign, args.hardware, args.track_width, args.six_state,
                                args.jump_after_turn)
                       for speed in args.speeds for sign in (-1, 1)]
        elif args.scenario == 'slope':
            reports = [run_terrain(speed, slope, 0, 0, args.hardware, args.track_width, args.six_state)
                       for speed in args.speeds for slope in (-8, 8)]
        else:
            if any(speed == 0 for speed in args.speeds):
                parser.error('bump 需要非零 --speeds，例如 --speeds 1 -1')
            reports = [run_terrain(speed, 0, height, side, args.hardware, args.track_width, args.six_state)
                       for speed in args.speeds for height in args.bump_heights for side in (-1, 1)]
        if args.report:
            args.report.write_text(json.dumps(reports, indent=2, allow_nan=False) + '\n')
        assert all(r['passed'] for r in reports), '转向/地形场景未过门'
    else:
        results = [run(speed, direction, sign, args.force_impulse, args.yaw_impulse, args.duration,
                   args.hardware, args.track_width, args.six_state)
               for speed in args.speeds for direction in ('x', 'y', 'yaw') for sign in (-1, 1)]
        assert all(results)
