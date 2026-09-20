#!/usr/bin/env python3
"""用当前回归负载核算 RI85-PH 硬件候选；厂家值不是实机标定值。"""
import math
import os
import argparse

import mujoco

import hardware_profile as hw
import wheelleg_sim as sim


XML = os.path.join(os.path.dirname(__file__), '..', 'xml', 'wheelleg_dm8009.xml')

def measure_baseline(hardware=False, track_width=None, six_state=False, wheel_samples=None):
    peak = dict(hip_torque=0.0, hip_speed=0.0, hip_power=0.0,
                hip_continuous=0.0, wheel_torque=0.0, wheel_speed=0.0,
                wheel_power=0.0, wheel_continuous=0.0, total_power=0.0,
                high_torque_run=0.0, average_power=0.0)
    for speed in (0.0, 1.0, -1.0):
        model, data = sim.load_model(XML, hardware, track_width)
        state = sim.make_state(model, hardware, six_state)
        dt = model.opt.timestep
        hip_ids = [model.actuator(name).id for name in
                   ('motor_alphaL', 'motor_betaL', 'motor_alphaR', 'motor_betaR')]
        wheel_ids = [model.actuator(name).id for name in ('motor_wheelL', 'motor_wheelR')]
        hip_dofs = [model.jnt_dofadr[model.joint(name).id] for name in
                    ('alphaL', 'betaL', 'alphaR', 'betaR')]
        wheel_dofs = [model.jnt_dofadr[model.joint(name).id] for name in
                      ('wheel1', 'wheel2')]
        run = energy = 0.0
        for step in range(int(8.0 / dt)):
            t = step * dt
            state.cmd_vel = speed if t > 1.0 else 0.0
            state.cmd_jump = ((4.0 < t < 4.05) if speed else (2.0 < t < 2.05))
            sim.control(model, data, state)
            hip_torque = [abs(float(data.ctrl[i])) for i in hip_ids]
            hip_speed = [abs(float(data.qvel[i])) for i in hip_dofs]
            wheel_torque = [abs(float(data.ctrl[i])) for i in wheel_ids]
            wheel_speed = [abs(float(data.qvel[i])) for i in wheel_dofs]
            if wheel_samples is not None:
                for side, a, d in zip(('L', 'R'), wheel_ids, wheel_dofs):
                    wheel_samples.append(dict(command_speed=speed, time=t, phase=state.jp,
                                              side=side, torque=float(data.ctrl[a]),
                                              rpm=float(data.qvel[d]) * 30 / math.pi))
            powers = ([float(data.ctrl[a]) * float(data.qvel[d])
                       for a, d in zip(hip_ids, hip_dofs)] +
                      [float(data.ctrl[a]) * float(data.qvel[d])
                       for a, d in zip(wheel_ids, wheel_dofs)])
            positive_power = sum(max(0.0, value) for value in powers)
            energy += positive_power * dt
            peak['hip_torque'] = max(peak['hip_torque'], max(hip_torque))
            peak['hip_speed'] = max(peak['hip_speed'], max(hip_speed))
            peak['hip_power'] = max(peak['hip_power'],
                                    max(t * w for t, w in zip(hip_torque, hip_speed)))
            peak['wheel_torque'] = max(peak['wheel_torque'], max(wheel_torque))
            peak['wheel_speed'] = max(peak['wheel_speed'], max(wheel_speed))
            peak['wheel_power'] = max(peak['wheel_power'],
                                      max(t * w for t, w in zip(wheel_torque, wheel_speed)))
            peak['total_power'] = max(peak['total_power'], positive_power)
            if state.jp != 'JUMP':
                peak['hip_continuous'] = max(peak['hip_continuous'], max(hip_torque))
            if state.jp == 'DRIVE':
                peak['wheel_continuous'] = max(peak['wheel_continuous'], max(wheel_torque))
            run = run + dt if max(hip_torque) > (hw.HIP_RATED_TORQUE if hardware else 2.5) else 0.0
            peak['high_torque_run'] = max(peak['high_torque_run'], run)
            mujoco.mj_step(model, data)
        peak['average_power'] = max(peak['average_power'], energy / 8.0)
    return sum(model.body_mass), peak


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='store_true', help='直接测量 180mm 六状态暂定硬件模型')
    args = parser.parse_args()
    baseline_mass, load = measure_baseline(args.candidate, .18 if args.candidate else None, args.candidate)
    scale = hw.DESIGN_MASS / baseline_mass
    hip_rated = hw.HIP_RATED_TORQUE
    hip_peak = hw.HIP_PEAK_TORQUE
    hip_rated_rpm = hw.HIP_RATED_RPM
    hip_required = load['hip_torque'] * scale
    wheel_required = load['wheel_torque'] * scale
    peak_bus_current = load['total_power'] * scale / (44.4 * 0.80)
    battery_current = 1.4 * 150 * 0.50  # C 值按 50% 使用，不把营销额定当工程极限。
    usable_wh = 44.4 * 1.4 * 0.80
    average_electrical = load['average_power'] * scale / 0.80 + 30.0
    stop_torque = hw.DESIGN_MASS * 1.4 ** 2 / (2 * 0.45) * hw.WHEEL_RADIUS / 2
    wheel_inertia = hw.MOTOR_INERTIA + hw.TIRE_MASS * hw.WHEEL_RADIUS ** 2

    print(f'load source: {"six-state simulation" if args.candidate else "scaled legacy baseline"}; '
          f'wheel radius={hw.WHEEL_RADIUS:.4f}m; wheel-drive packaging NOT validated')
    print(f'design mass: {hw.DESIGN_MASS:.3f} kg (measured model {baseline_mass:.3f} kg, x{scale:.2f})')
    for name, mass in hw.MASS_ITEMS.items():
        print(f'  {name:32s} {mass:.3f} kg')
    print(f'hip: need {hip_required:.2f} N·m @ {load["hip_speed"] * 60 / (2 * math.pi):.0f} rpm; '
          f'available {hip_peak:.2f} peak / {hip_rated:.2f} rated N·m @ {hip_rated_rpm:.0f} rpm')
    print(f'wheel: need {wheel_required:.2f} N·m @ {load["wheel_speed"] * 60 / (2 * math.pi):.0f} rpm; '
          f'available {hw.MOTOR_PEAK_TORQUE:.2f} peak / {hw.MOTOR_RATED_TORQUE:.2f} rated N·m')
    print(f'peak bus: {peak_bus_current:.1f} A; derated battery budget {battery_current:.1f} A; '
          f'estimated runtime {usable_wh / average_electrical * 60:.1f} min')
    print(f'1.4 m/s -> 0 in 0.45 m: {stop_torque:.3f} N·m/wheel; '
          f'wheel inertia estimate {wheel_inertia:.7f} kg·m²')

    assert hip_peak >= 1.2 * hip_required
    assert hip_rated >= 1.5 * load['hip_continuous'] * scale
    assert hip_rated_rpm >= 1.2 * load['hip_speed'] * 60 / (2 * math.pi)
    assert hw.MOTOR_PEAK_TORQUE >= 1.15 * wheel_required
    assert hw.MOTOR_RATED_RPM >= 2.0 * load['wheel_speed'] * 60 / (2 * math.pi)
    assert battery_current >= 1.15 * peak_bus_current
    assert load['high_torque_run'] <= 0.020

    model, data = sim.load_model(XML, True)
    assert abs(sum(model.body_mass) - hw.DESIGN_MASS) < 1e-9
    assert abs(model.geom_size[model.geom('wheel_collide_L').id, 0] - hw.WHEEL_RADIUS) < 1e-9
    assert tuple(model.actuator_ctrlrange[model.actuator('motor_alphaL').id]) == (
        -hw.HIP_PEAK_TORQUE, hw.HIP_PEAK_TORQUE)
