"""height-115 站立连通支路的静态腿长/极角工作空间；不代表动态安全。"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wheelleg_ppo/tools"))
import wheelleg_sim as sim

HEIGHTS = (.115, .12, .14, .16, .20, .25, .30, .35, .38)
CAPS = (1.5, 1.4)  # 后者是每个主动关节额外保留 0.1 rad，并非实机安全标定。
SCAN_STEP = .0005


def inverse(length, angle):
    """轮心 C=(L5/2+L*sin(angle), -L*cos(angle)) 的原模型站立支逆解。"""
    x = sim.L5 / 2 + length * math.sin(angle)
    z = -length * math.cos(angle)
    ac = math.hypot(x, z)
    ce = math.hypot(x - sim.L5, z)
    ca = (sim.L1**2 + ac**2 - sim.L2**2) / (2 * sim.L1 * ac)
    cb = (sim.L4**2 + ce**2 - sim.L3**2) / (2 * sim.L4 * ce)
    if not (-1 <= ca <= 1 and -1 <= cb <= 1):
        return None  # 不像旧 IK 那样夹住非法 acos；越过串联伸直/折叠即不可达。
    qa = sim.PHI1_STAND - math.atan2(z, x) + math.acos(ca)
    qb = sim.PHI4_STAND - math.atan2(z, x - sim.L5) - math.acos(cb)
    return qa, qb


def feasible(length, angle, cap):
    q = inverse(length, angle)
    return q is not None and max(map(abs, q)) <= cap


def boundary(length, cap, sign):
    assert feasible(length, 0, cap)
    inside = 0.
    outside = None
    reentry = False
    for i in range(1, 3001):
        angle = sign * i * SCAN_STEP
        if feasible(length, angle, cap):
            if outside is not None:
                reentry = True
        elif outside is None:
            outside = angle
            inside = sign * (i - 1) * SCAN_STEP
    assert outside is not None and not reentry, (length, cap, sign)
    for _ in range(50):
        mid = (inside + outside) / 2
        if mid == inside or mid == outside:
            break
        if feasible(length, mid, cap):
            inside = mid
        else:
            outside = mid
    assert feasible(length, inside, cap) and not feasible(length, outside, cap)
    return inside, abs(outside - inside)


def run(output):
    max_length_error = max_angle_error = max_asymmetry = max_bracket = 0.
    samples = 0
    rows = []
    for length in HEIGHTS:
        row = {"length_m": length}
        for cap, name in ((1.5, "hard"), (1.4, "joint_reserve_0p1rad")):
            low, low_bracket = boundary(length, cap, -1)
            high, high_bracket = boundary(length, cap, 1)
            max_bracket = max(max_bracket, low_bracket, high_bracket)
            max_asymmetry = max(max_asymmetry, abs(low + high))
            for i in range(1, 200):
                angle = low + (high - low) * i / 200
                q = inverse(length, angle)
                assert q is not None and max(map(abs, q)) <= cap + 1e-12
                fk = sim.fk_joints(*q)
                max_length_error = max(max_length_error, abs(fk["leg_len"] - length))
                max_angle_error = max(max_angle_error, abs(fk["phi5"] + math.pi / 2 - angle))
                samples += 1
            q = inverse(length, high)
            assert q is not None
            x = sim.L5 / 2 + length * math.sin(high)
            z = -length * math.cos(high)
            ac = math.hypot(x, z)
            ce = math.hypot(x - sim.L5, z)
            binding = []
            if abs(abs(q[0]) - cap) < 1e-9:
                binding.append("alpha_joint")
            if abs(abs(q[1]) - cap) < 1e-9:
                binding.append("beta_joint")
            if abs(ac - (sim.L1 + sim.L2)) < 1e-9:
                binding.append("A_chain_straight")
            if abs(ce - abs(sim.L4 - sim.L3)) < 1e-9:
                binding.append("D_chain_folded")
            assert binding, (length, cap)
            row[name] = {"min_deg": math.degrees(low), "max_deg": math.degrees(high),
                         "positive_boundary_q_rad": q, "positive_binding": binding,
                         "A_chain_outer_reach_margin_m": sim.L1 + sim.L2 - ac}
        rows.append(row)

    peaks = []
    for cap in CAPS:
        pose = sim.fk_joints(-cap, -cap)
        length = pose["leg_len"]
        angle = pose["phi5"] + math.pi / 2
        edge, bracket = boundary(length, cap, 1)
        assert abs(edge - angle) < 1e-10
        assert all(boundary(length + delta, cap, 1)[0] <= edge + 1e-10
                   for delta in (-.0005, .0005))
        max_bracket = max(max_bracket, bracket)
        peaks.append({"joint_cap_rad": cap, "length_m": length, "half_angle_deg": math.degrees(angle)})

    # 高端硬限位包络从主动关节 α=-1.5 切换到 A 链串联伸直。
    low, high = .37, .38
    for _ in range(50):
        mid = (low + high) / 2
        sine = ((sim.L1 + sim.L2)**2 - (sim.L5 / 2)**2 - mid**2) / (sim.L5 * mid)
        angle = math.asin(sine)
        x = sim.L5 / 2 + mid * math.sin(angle)
        z = -mid * math.cos(angle)
        alpha = sim.PHI1_STAND - math.atan2(z, x)
        if alpha < -1.5:
            low = mid
        else:
            high = mid

    sources = ("wheelleg_ppo/tools/wheelleg_sim.py", "wheelleg_warp/probe_height_angle_envelope.py")
    model = "wheelleg_ppo/xml/wheelleg.xml"
    payload = {
        "protocol": "static connected standing-branch geometry only; no passive-joint, load, collision or dynamic-safety claim",
        "nominal_height_range_m": [.115, .38],
        "joint_caps_rad": {"hard": 1.5, "joint_reserve_0p1rad": 1.4},
        "angle_definition": "phi5 + pi/2; positive wheel center forward relative to hip midpoint",
        "method": {"first_infeasible_scan_step_rad": SCAN_STEP, "bisection_iterations_max": 50,
                   "invalid_acos_domain": "infeasible, never clamped"},
        "rows": rows, "peaks": peaks, "hard_A_chain_reach_switch_length_m": high,
        "verification": {"connected_interior_samples": samples,
                         "max_FK_length_error_m": max_length_error,
                         "max_FK_angle_error_rad": max_angle_error,
                         "max_positive_negative_asymmetry_rad": max_asymmetry,
                         "max_bisection_bracket_rad": max_bracket,
                         "scan_reentries": 0},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "model_sha256": {model: hashlib.sha256((ROOT / model).read_bytes()).hexdigest()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"verification": payload["verification"], "peaks": peaks}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
