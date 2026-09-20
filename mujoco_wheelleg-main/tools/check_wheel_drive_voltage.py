#!/usr/bin/env python3
"""RO100 38/24 齿候选的电压初筛；只重放旧负载，不是新机械方案验收。

来源/质量假设见 HARDWARE_SELECTION_2026-09-02.md 第 0.1 节。
"""
import argparse
import contextlib
import json
import math
import sys


EFFICIENCY = 0.90
CANDIDATE_MASS = 4.9246291


def required_voltage(motor_torque, motor_rpm):
    """按 48V 下 (0Nm,2550rpm)、(4Nm,2000rpm) 两点拟合。

    ponytail: 仅稳态电动等效直线；>4Nm 为外推，厂家联合曲线/台架数据后替换。
    对制动样本取绝对值作电动上界初筛，不能证明回生母线安全。
    """
    if not all(math.isfinite(x) for x in (motor_torque, motor_rpm)):
        raise ValueError('motor torque and rpm must be finite')
    return 48 * (abs(motor_rpm) / 2550 + abs(motor_torque) / 4 * (1 - 2000 / 2550))


def self_test():
    assert required_voltage(0, 0) == 0
    assert math.isclose(required_voltage(0, 2550), 48)
    assert math.isclose(required_voltage(4, 2000), 48)
    assert math.isclose(required_voltage(-4, -2000), 48)
    assert math.isclose(required_voltage(4, 2 * 1275 * (24 / 36)), 42.35294117647)
    # 同一42V设计下限和原转速门，36齿失败、38齿通过；38.4V压力点仍失败。
    assert required_voltage(4, 2 * 1275 * (24 / 36)) > 42
    assert 38.4 < required_voltage(4, 2 * 1275 * (24 / 38)) < 42
    assert 12 * (24 / 38) * EFFICIENCY > 5 * CANDIDATE_MASS / 4.2740465 * 1.15
    try:
        required_voltage(float('nan'), 0)
    except ValueError:
        pass
    else:
        raise AssertionError('non-finite input accepted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--motor-teeth', type=int, choices=(36, 38), default=38)
    parser.add_argument('--min-bus-voltage', type=float, default=42.0,
                        help='设计下限：默认12×3.5V厂家通用参考；38.4仅作压力对照')
    args = parser.parse_args()
    if not math.isfinite(args.min_bus_voltage) or not 0 < args.min_bus_voltage <= 50.4:
        parser.error('--min-bus-voltage must be finite and in (0, 50.4]')
    self_test()
    if args.self_test:
        print('PASS: voltage fit endpoints, sign symmetry, speed reserve, input validation')
        sys.exit(0)

    import check_hardware_sizing as sizing
    ratio = 24 / args.motor_teeth  # motor speed / wheel speed
    samples = []
    with contextlib.redirect_stdout(sys.stderr):
        mass, load = sizing.measure_baseline(True, .18, True, samples)
    # 质量缩放+15%只用于旧轨迹筛选，不能预测新惯量/偏置机构的运动。
    torque_scale = CANDIDATE_MASS / float(mass) * 1.15
    worst = max(samples, key=lambda s: required_voltage(
        s['torque'] * torque_scale / (ratio * EFFICIENCY), s['rpm'] * ratio))
    demand = required_voltage(worst['torque'] * torque_scale / (ratio * EFFICIENCY),
                              worst['rpm'] * ratio)
    speed_gate = required_voltage(4, 2 * load['wheel_speed'] * 30 / math.pi * ratio)
    motor_peak = load['wheel_torque'] * torque_scale / (ratio * EFFICIENCY)
    report = dict(status='PRELIMINARY_ONLY', source='50mm six-state baseline; not 60.325mm dynamics',
                  sample_count=len(samples), torque_scale=torque_scale, ratio=ratio,
                  motor_teeth=args.motor_teeth, wheel_teeth=24,
                  design_min_bus_voltage=args.min_bus_voltage,
                  design_voltage_source='https://genstattu.com/bw/ (generic reference, not measured)',
                  worst_paired_sample=worst, paired_voltage_estimate=demand,
                  motor_peak_torque_estimate=motor_peak, extrapolated_above_rated=motor_peak > 4,
                  voltage_for_2x_speed_at_rated_torque=speed_gate,
                  unresolved=['Standard KV55 joint curve (Lite PDF curve labeled KV50)',
                              'peak duration/thermal', 'FOC current convention', 'regeneration',
                              'loaded battery voltage and driver/wiring voltage loss',
                              'new mass/inertia/geometry and transmission strength'])
    report['voltage_screens'] = {str(v): dict(paired_load=demand <= v,
        original_speed_reserve=speed_gate <= v, peak_torque=motor_peak <= 12)
        for v in (50.4, 48.0, 44.4, args.min_bus_voltage)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if all(all(gates.values()) for gates in report['voltage_screens'].values()) else 1)
