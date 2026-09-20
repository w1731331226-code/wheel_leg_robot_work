"""动作验收；硬件尺寸/质量保留，供电、热和回生按理想模型处理。"""
import argparse
import contextlib
import hashlib
import io
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

import hardware_profile as hw
import wheelleg_sim as sim


def run(mode, action=None):
    original = sim.control
    peak_torque = np.zeros(6)
    peak_rpm = np.zeros(6)
    overload_run = np.zeros(6)
    overload_max = np.zeros(6)
    overload_total = np.zeros(6)
    peak_run = np.zeros(6)
    peak_max = np.zeros(6)
    torque_squared_integral = np.zeros(6)
    rail_peaks = np.zeros((2, 2))
    rail_energy = np.zeros((2, 2))
    duration = positive_peak = negative_peak = positive_energy = negative_energy = 0.0
    names = dofs = hip = None

    def observe(model, data, state):
        nonlocal duration, positive_peak, negative_peak, positive_energy, negative_energy
        nonlocal names, dofs, hip
        if names is None:
            names = [model.actuator(i).name for i in range(model.nu)]
            dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
            hip = np.array(['wheel' not in name for name in names])
        original(model, data, state)
        dt = model.opt.timestep
        torque = abs(data.ctrl)
        speed = data.qvel[dofs]
        np.maximum(peak_torque, torque, out=peak_torque)
        np.maximum(peak_rpm, abs(speed)*30/np.pi, out=peak_rpm)
        overload_run[:] = np.where(torque > np.where(hip, hw.HIP_RATED_TORQUE,
                                                    hw.MOTOR_RATED_TORQUE), overload_run+dt, 0)
        peak_run[:] = np.where(torque >= np.where(hip, hw.HIP_PEAK_TORQUE,
                                                 hw.MOTOR_PEAK_TORQUE)-1e-9, peak_run+dt, 0)
        np.maximum(overload_max, overload_run, out=overload_max)
        overload_total[:] += (overload_run > 0)*dt
        np.maximum(peak_max, peak_run, out=peak_max)
        torque_squared_integral[:] += torque**2*dt
        power = data.ctrl*speed
        positive = float(np.maximum(power, 0).sum())
        negative = float(np.maximum(-power, 0).sum())
        rail_power = np.array([[np.maximum(power[mask], 0).sum(),
                                np.maximum(-power[mask], 0).sum()] for mask in (hip, ~hip)])
        np.maximum(rail_peaks, rail_power, out=rail_peaks)
        rail_energy[:] += rail_power*dt
        positive_peak, negative_peak = max(positive_peak, positive), max(negative_peak, negative)
        positive_energy += positive*dt
        negative_energy += negative*dt
        duration += dt

    log = io.StringIO()
    with patch.object(sim, 'control', observe), contextlib.redirect_stdout(log):
        passed = action() if action is not None else sim.run_headless(mode, True, None, True)
    assert np.allclose(rail_energy.sum(axis=0), [positive_energy, negative_energy],
                       rtol=1e-10, atol=1e-8), '分支机械能与总量不一致'
    return dict(mode=mode, action_pass=bool(passed), duration_s=duration,
                motors={name: dict(peak_torque_Nm=float(peak_torque[i]),
                    peak_rpm=float(peak_rpm[i]), over_rated_longest_s=float(overload_max[i]),
                    at_peak_longest_s=float(peak_max[i]),
                    over_rated_total_s=float(overload_total[i]),
                    torque_squared_integral_Nm2_s=float(torque_squared_integral[i]),
                    torque_rms_Nm=float(np.sqrt(torque_squared_integral[i]/duration)))
                    for i, name in enumerate(names)},
                rails={name: dict(mechanical_positive_peak_W=float(rail_peaks[i, 0]),
                    mechanical_absorbed_peak_W=float(rail_peaks[i, 1]),
                    mechanical_positive_J=float(rail_energy[i, 0]),
                    mechanical_absorbed_J=float(rail_energy[i, 1]))
                    for i, name in enumerate(('hip_bus', 'wheel_24V'))},
                mechanical_positive_peak_W=positive_peak, mechanical_absorbed_peak_W=negative_peak,
                mechanical_positive_J=positive_energy, mechanical_absorbed_J=negative_energy,
                lossless_battery_current_equivalent_A=positive_peak/hw.HIP_BUS_VOLTAGE,
                peak_window_pass=bool(np.all(overload_max <= hw.PEAK_WINDOW+1e-9))), log.getvalue()


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mass', type=float, default=round(hw.DESIGN_MASS, 6))
    parser.add_argument('--controller-mass', type=float, default=round(hw.DESIGN_MASS, 6))
    parser.add_argument('--hip-voltage', type=float, default=hw.HIP_BUS_VOLTAGE,
                        help='理想恒压下用于名义转速边界的仿真值')
    parser.add_argument('--modes', nargs='+', choices=('stand', 'stop', 'jump', 'jumpfwd', 'jumpback', 'landing'),
                        default=['jump', 'jumpfwd', 'jumpback', 'stop'])
    parser.add_argument('--output', type=Path,
                        default=root/'tools/results/continuation_2026-09-09/dm8009_mf9025')
    args = parser.parse_args()
    if not 7.0 <= args.mass <= 8.0:
        parser.error('--mass must be within 7–8 kg')
    if not 7.0 <= args.controller_mass <= 8.0:
        parser.error('--controller-mass must be within 7–8 kg')
    if not 24.0 <= args.hip_voltage <= 42.0:
        parser.error('--hip-voltage must be within 24–42 V for the 10S model')
    hw.HIP_BUS_VOLTAGE = args.hip_voltage
    hw.HIP_RATED_RPM, hw.HIP_NO_LOAD_RPM = (rpm*args.hip_voltage/24 for rpm in (100, 160))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    configure, make_state = hw.configure_spec, sim.make_state

    def nominal_spec(spec, track_width=None):
        configure(spec, track_width)
        spec.geom('chassis_lid').mass += round(args.controller_mass-hw.DESIGN_MASS, 9)

    with patch.object(hw, 'configure_spec', nominal_spec):
        nominal_model, _ = sim.load_model(str(root/'xml/wheelleg_dm8009.xml'), True)
    from rm_controller import design_controller
    controller_design = design_controller(nominal_model)

    def payload_spec(spec, track_width=None):
        configure(spec, track_width)
        spec.geom('chassis_lid').mass += round(args.mass-hw.DESIGN_MASS, 9)

    def fixed_controller(model, hardware=False, six_state=False):
        assert abs(sum(model.body_mass)-args.mass) < 1e-9
        # 只改变被控对象；各质量复用指定名义质量的增益、前馈和质量缩放。
        return make_state(model, hardware, six_state, design=controller_design)

    rows = []
    cases = [(mode, None) for mode in args.modes if mode != 'landing']
    if 'landing' in args.modes:
        from test_jump_stop import run as landing_stop
        cases.extend((f'landing_{direction}_{delay}', partial(landing_stop, direction, delay, True, None, True))
                     for direction in (1, -1) for delay in (0.0, 0.25, 0.5))
    with patch.object(hw, 'configure_spec', payload_spec), patch.object(sim, 'make_state', fixed_controller):
        for mode, action in cases:
            row, log = run(mode, action)
            rows.append(row)
            (out/f'10s_{mode}.log').write_text(log)
            print(args.mass, mode, 'PASS' if row['action_pass'] else 'FAIL', flush=True)
    files = ['tools/hardware_profile.py', 'tools/wheelleg_sim.py', 'tools/rm_controller.py',
             'tools/model_lqr.py', 'tools/state_estimation.py', 'tools/test_dm8009_dynamics.py',
             'tools/test_jump_stop.py', 'xml/wheelleg_dm8009.xml']
    report = dict(hip_bus_V=hw.HIP_BUS_VOLTAGE, wheel_bus_V=hw.WHEEL_BUS_VOLTAGE,
        mass_kg=args.mass, controller_mass_kg=args.controller_mass,
        added_payload_position_m=nominal_model.geom('chassis_lid').pos.tolist(),
        track_width_m=hw.TRACK_WIDTH, runs=rows, hardware_dynamic_acceptance=False,
        idealized_power_model=hw.IDEAL_POWER_MODEL,
        note='Ideal constant-bus, lossless, thermally unlimited and regen-absorbing simulation. '
             'Geometry, mass, inertia and nominal torque/speed boundaries are retained; '
             'current, temperature, voltage sag and regenerative hardware are intentionally omitted.',
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})
    (out/'10s_dynamics.json').write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if all(r['action_pass'] and r['peak_window_pass'] for r in rows) else 1)
