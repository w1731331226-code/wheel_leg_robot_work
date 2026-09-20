"""15 cm 台阶压力测试：全速双向起跳、扰动、真实碰撞与台面落地。"""
import math
import os

import mujoco

import wheelleg_sim as sim


STEP_HEIGHT = 0.15
PREP_LEAD = 0.75
LIMIT = math.radians(5.0)
XML = os.path.join(os.path.dirname(sim.__file__), '..', 'xml', 'wheelleg.xml')


def load(direction):
    spec = mujoco.MjSpec.from_file(XML)
    lip = 3.0 * direction
    spec.worldbody.add_geom(
        name='step_15cm', type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[lip + 5.0 * direction, 0.0, STEP_HEIGHT / 2],
        size=[5.0, 0.5, STEP_HEIGHT / 2],
        friction=[2.0, 0.05, 0.0001], rgba=[0.45, 0.45, 0.48, 1.0])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe('stand').id)
    mujoco.mj_forward(model, data)
    return model, data, lip


def run(direction, request_distance, disturbance_sign):
    model, data, lip = load(direction)
    state = sim.St()
    prepared = requested = completed = False
    push_until = -1.0
    chassis = model.body('chassis').id
    takeoff_speed = height = 0.0
    maxima = [0.0, 0.0, 0.0]
    for step in range(int(12.0 / model.opt.timestep)):
        t = step * model.opt.timestep
        state.cmd_vel = float(direction) if t > 1.0 else 0.0
        if (not prepared and direction * data.qpos[0]
                >= direction * lip - request_distance - PREP_LEAD):
            state.cmd_jump_prepare = True
            prepared = True
        if not requested and direction * data.qpos[0] >= direction * lip - request_distance:
            state.cmd_jump = True
            push_until = t + 0.05
            requested = True
        data.xfrc_applied[chassis] = 0.0
        if t < push_until:
            data.xfrc_applied[chassis, 0] = 4.0 * disturbance_sign
            data.xfrc_applied[chassis, 5] = -0.1 * disturbance_sign
        previous = state.jp
        sim.control(model, data, state)
        mujoco.mj_step(model, data)
        if previous == 'DRIVE' and state.jp == 'SQUAT':
            takeoff_speed = state.jp_v_keep
        if previous == 'LAND' and state.jp == 'DRIVE':
            completed = True
            height = state.jp_peak_z - state.jp_takeoff_z
        if requested:
            roll, pitch, yaw = sim.euler(data)
            yaw_error = math.atan2(math.sin(yaw - state.yaw_target),
                                   math.cos(yaw - state.yaw_target))
            maxima = [max(old, abs(value))
                      for old, value in zip(maxima, (roll, pitch, yaw_error))]
        if data.qpos[2] < 0.02 or max(maxima[:2]) > math.radians(40):
            break
    on_step = direction * data.qpos[0] > direction * lip + 0.20 and data.qpos[2] > 0.24
    passed = (requested and completed and on_step
              and direction * takeoff_speed >= 0.90
              and height >= sim.JUMP_HEIGHT and max(maxima) <= LIMIT)
    angles = tuple(round(math.degrees(value), 2) for value in maxima)
    print(f'{"前进" if direction > 0 else "后退"} d={request_distance:.2f}m: '
          f'{"PASS" if passed else "FAIL"} v={takeoff_speed:.2f}m/s h={height:.3f}m '
          f'r/p/y={angles}° final=({data.qpos[0]:.2f},{data.qpos[2]:.3f})m')
    return passed


assert all([run(direction, distance, -1 if index % 2 else 1)
            for direction in (1, -1)
            for index, distance in enumerate((0.84, 0.90, 0.96))])
