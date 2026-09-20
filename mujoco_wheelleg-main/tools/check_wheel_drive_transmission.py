#!/usr/bin/env python3
"""厂家目录传动初筛：重放旧50mm负载，不代表新轮驱闭环或实机验收。"""
import bisect
import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path

import check_wheel_drive_voltage as voltage

ROOT = Path(__file__).resolve().parents[1]
# SDP/SI D820, T-72/Table 35, 24齿5mm GT3，15mm基准宽；原PDF第75页。
RPM = (200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800,
       2000, 2400, 2800, 3200, 3600, 4000, 5000, 6000, 8000, 10000, 12000, 14000)
TORQUE = (10.22, 9.71, 9.34, 9.05, 8.82, 8.45, 8.17, 7.93, 7.73, 7.56,
          7.41, 7.27, 7.03, 6.82, 6.64, 6.47, 6.32, 5.98, 5.69, 5.16, 4.67, 4.19, 3.69)
WIDTH_FACTOR = {15: 1.0, 20: 1.33, 25: 1.67}
LENGTH, LENGTH_FACTOR, SERVICE = .375, .85, 1.5
MOTOR_TEETH, WHEEL_TEETH, PITCH = 38, 24, .005


def belt_length(center):
    small, large = WHEEL_TEETH * PITCH / (2 * math.pi), MOTOR_TEETH * PITCH / (2 * math.pi)
    if not math.isfinite(center) or center <= small + large:
        raise ValueError('center distance must separate the pitch circles')
    angle = math.asin((large - small) / center)
    return 2 * math.sqrt(center**2 - (large-small)**2) + math.pi*(large+small) + 2*angle*(large-small)


def capacity(rpm, width):
    if not math.isfinite(rpm) or width not in WIDTH_FACTOR:
        raise ValueError('finite rpm and catalog belt width required')
    # ponytail: 向上取转速档，低速用200rpm档避开灰区的较高额定值；超表返回0。
    # 这是保守目录初筛，反向冲击、预紧和寿命仍须按实际工况复核。
    index = bisect.bisect_left(RPM, abs(rpm))
    return (TORQUE[index] * LENGTH_FACTOR * WIDTH_FACTOR[width] / SERVICE
            if index < len(RPM) else 0.0)


def center_distance():
    low, high = (MOTOR_TEETH+WHEEL_TEETH)*PITCH/(2*math.pi) + 1e-9, LENGTH/2
    for _ in range(60):
        mid = (low+high)/2
        if belt_length(mid) < LENGTH:
            low = mid
        else:
            high = mid
    return (low+high)/2


def self_test():
    assert math.isclose(belt_length(center_distance()), LENGTH, abs_tol=1e-12)
    assert capacity(200, 15) == capacity(-200, 15)
    assert capacity(200.01, 15) < capacity(200, 15)
    assert capacity(0, 15) == capacity(200, 15)
    assert capacity(14001, 25) == 0
    assert capacity(1000, 15) < 6 < capacity(1000, 25)
    for bad in (float('nan'), float('inf')):
        try:
            capacity(bad, 25)
        except ValueError:
            pass
        else:
            raise AssertionError('non-finite speed accepted')


if __name__ == '__main__':
    self_test()
    if sys.argv[1:] == ['--self-test']:
        print('PASS: belt geometry, catalog bin boundaries, width and invalid inputs')
        raise SystemExit(0)
    if sys.argv[1:]:
        raise SystemExit('usage: check_wheel_drive_transmission.py [--self-test]')
    import check_hardware_sizing as sizing
    samples = []
    with contextlib.redirect_stdout(sys.stderr):
        mass, _ = sizing.measure_baseline(True, .18, True, samples)
    scale = voltage.CANDIDATE_MASS / float(mass) * 1.15
    # 轮侧负载加原15%余量；再除效率，保守覆盖传动上游张力。
    results = {}
    for width in WIDTH_FACTOR:
        worst = max(samples, key=lambda s: abs(s['torque'])*scale/voltage.EFFICIENCY /
                    max(capacity(s['rpm'], width), 1e-12))
        demand = abs(worst['torque'])*scale/voltage.EFFICIENCY
        available = capacity(worst['rpm'], width)
        results[width] = dict(pass_screen=demand <= available, utilization=demand/max(available, 1e-12),
                             demand_Nm=demand, allowed_Nm=available, worst_sample=worst)
    # 20mm仅比较；选已核对目录的标准375-5MGT-25（9400-8075）。
    selected = 25 if results[25]['pass_screen'] else None
    center = center_distance()
    angle = math.asin((MOTOR_TEETH-WHEEL_TEETH)*PITCH/(2*math.pi*center))
    old_center = math.hypot(.0683, .0798)
    report = dict(status='CATALOG_SCREEN_ONLY', wheel_diameter_mm=60.325, sample_count=len(samples),
                  nominal_mass_kg=voltage.CANDIDATE_MASS, torque_scale=scale, service_factor=SERVICE,
                  belt_pitch_length_mm=LENGTH*1000, center_distance_mm=center*1000,
                  old_center_distance_mm=old_center*1000, old_required_belt_length_mm=belt_length(old_center)*1000,
                  motor_offset_along_leg_mm=(center-old_center)*1000,
                  small_pulley_wrap_deg=180-2*math.degrees(angle),
                  small_pulley_teeth_in_mesh=WHEEL_TEETH*(math.pi-2*angle)/(2*math.pi),
                  belt_width_comparison=results, selected_width_mm=selected,
                  selected_belt='Gates PowerGrip GT3 375-5MGT-25 / 9400-8075',
                  belt_catalog='https://www.gates.com/content/dam/documents-library/catalogs/power-transmission-catalog.pdf',
                  command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/check_wheel_drive_transmission.py',
                  source='https://www.sdp-si.com/D820/PDFS/Technical-Section.pdf (T-60, T-61, T-65, T-72)',
                  limitations=['old trajectory and nominal mass scaling; not new drivetrain dynamics',
                               'catalog screening excludes shock fatigue, pretension, mounting and temperature',
                               'shaft/hub/support mass and complete assembly still require engineering validation'])
    report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT/'tools/check_wheel_drive_voltage.py', ROOT/'tools/check_hardware_sizing.py',
                  ROOT/'tools/wheelleg_sim.py', ROOT/'tools/hardware_profile.py', ROOT/'tools/rm_controller.py',
                  ROOT/'tools/model_lqr.py', ROOT/'xml/wheelleg.xml']}
    report['returncode'] = 0 if selected else 1
    out = ROOT/'tools/results/continuation_2026-09-09/wheel_drive_transmission.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(report['returncode'])
