"""静态检查：名义腿长范围内用左右差动构型抬高一侧轮心25 mm。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wheelleg_ppo/tools"))
import wheelleg_sim as sim

HEIGHTS = (.16, .20, .25, .30, .35, .38)
LIFT = .025


def model_joint_min():
    low, high = .094, .16
    for _ in range(60):
        mid = (low + high) / 2
        if max(abs(angle) for angle in sim.ik(mid)) > 1.5:
            low = mid
        else:
            high = mid
    assert abs(sim.fk_joints(*sim.ik(high))["leg_len"] - high) < 1e-10
    return high


MODEL_JOINT_MIN = model_joint_min()
SIM_TRANSIENT_MIN = MODEL_JOINT_MIN + .02  # 仅仿真代理下限；非实机标定值。


def leg_pose(model, qpos, side, length):
    a, b = sim.ik(length)
    assert abs(sim.fk_joints(a, b)["leg_len"] - length) < 1e-10
    wheel = model.body("wheelL")
    coupler = model.site("couplerB_L_end")
    bpos = sim.L1 * np.array([np.cos(sim.PHI1_STAND - a), np.sin(sim.PHI1_STAND - a)])
    dpos = np.array([sim.L5, 0.]) + sim.L4 * np.array([np.cos(sim.PHI4_STAND - b), np.sin(sim.PHI4_STAND - b)])
    cpos = np.array([sim.L5 / 2, -length])
    vec_a, vec_c = cpos - bpos, cpos - dpos
    pa = np.arctan2(wheel.pos[2], wheel.pos[0]) - np.arctan2(vec_a[1], vec_a[0]) - a
    pc = np.arctan2(coupler.pos[2], coupler.pos[0]) - np.arctan2(vec_c[1], vec_c[0]) - b
    names = (("alphaL", "betaL", "passA_L", "passC_L"),
             ("alphaR", "betaR", "passA_R", "passC_R"))[side]
    values = (a, b, pa, pc)
    for name, value in zip(names, values):
        qpos[model.joint(name).qposadr[0]] = value
    return names, values


def run(output, leg_min):
    assert SIM_TRANSIENT_MIN <= leg_min <= sim.L_SQUAT_MIN
    model = mujoco.MjModel.from_xml_path(str(ROOT / "wheelleg_ppo/xml/wheelleg.xml"))
    data = mujoco.MjData(model)
    rows = []
    for height in HEIGHTS:
        # 优先收短障碍腿；在0.16 m下限用另一腿伸长补足差高。
        hit_length = max(leg_min, height - LIFT)
        support_length = hit_length + LIFT
        assert leg_min <= hit_length <= support_length <= sim.L_MAX
        nominal = sim.ik(height)
        for hit in (0, 1):
            data.qpos[:] = model.key_qpos[model.key("stand").id]
            data.qpos[2] = support_length + .05
            poses = [leg_pose(model, data.qpos, side, hit_length if side == hit else support_length)
                     for side in (0, 1)]
            mujoco.mj_forward(model, data)
            clearance = [float(data.xpos[model.body(name).id, 2] - .05)
                         for name in ("wheelL", "wheelR")]
            loop_error = max(float(np.linalg.norm(data.site_xpos[model.site(end).id] -
                                                  data.site_xpos[model.site(axle).id]))
                             for end, axle in (("couplerB_L_end", "wheel_axle_L"),
                                               ("couplerB_R_end", "wheel_axle_R")))
            margins = []
            for names, values in poses:
                for name, value in zip(names, values):
                    low, high = model.jnt_range[model.joint(name).id]
                    margins.append(float(min(value - low, high - value)))
            caster_clearance = min(float(data.geom_xpos[model.geom(name).id, 2] -
                                         model.geom_size[model.geom(name).id, 0])
                                   for name in ("caster_front", "caster_back"))
            mean_joint_rpm_80ms = max(abs(value - nominal[j]) for _, values in poses
                                       for j, value in enumerate(values[:2])) / .08 * 60 / (2 * np.pi)
            assert abs(clearance[hit] - LIFT) < 1e-8 and abs(clearance[1 - hit]) < 1e-8
            assert loop_error < 1e-8 and min(margins) > 0 and caster_clearance > 0
            rows.append(dict(height_m=height, obstacle_side="L" if hit == 0 else "R",
                             obstacle_leg_m=hit_length, support_leg_m=support_length,
                             wheel_clearance_m=clearance, loop_error_m=loop_error,
                             min_joint_margin_rad=min(margins), caster_clearance_m=caster_clearance,
                             mean_joint_speed_for_80ms_ramp_rpm=float(mean_joint_rpm_80ms)))
    sources = ("wheelleg_ppo/tools/wheelleg_sim.py", "wheelleg_ppo/xml/wheelleg.xml",
               "wheelleg_warp/probe_height_lift_workspace.py")
    payload = dict(protocol="static level chassis, one wheel supported, 25mm differential wheel-center lift",
                   leg_min_m=leg_min, nominal_task_range_m=[sim.L_SQUAT_MIN, sim.L_MAX],
                   model_joint_limit_min_m=MODEL_JOINT_MIN, simulated_hardware_proxy_min_m=SIM_TRANSIENT_MIN,
                   training=False, dynamic_feasibility=False, rows=rows,
                   summary=dict(total=len(rows), max_loop_error_m=max(r["loop_error_m"] for r in rows),
                                min_joint_margin_rad=min(r["min_joint_margin_rad"] for r in rows),
                                min_caster_clearance_m=min(r["caster_clearance_m"] for r in rows),
                                max_mean_joint_speed_80ms_rpm=max(r["mean_joint_speed_for_80ms_ramp_rpm"] for r in rows)),
                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bound", choices=("nominal", "transient"), required=True)
    args = parser.parse_args()
    run(args.output, sim.L_SQUAT_MIN if args.bound == "nominal" else SIM_TRANSIENT_MIN)
