"""0.115 m 六个正常场景：原径向参考与低侧目标投影的同图配对。"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import mujoco_warp as mjw
import numpy as np
import warp as wp
from native.controller import D, allowed, control
from native.environment import NativeEnv, after, begin, command_step, reduce_contacts
from native.terrain import HEIGHT_115_GEOMETRIC_MIN, bank_height_115
from probe_height_115_margin import cases
from probe_height_115_safety_reflex import audit


@wp.kernel
def monitor(q: wp.array2d[float], v: wp.array2d[float], ctrl: wp.array2d[float],
            ids: wp.array[int], active: wp.array[int], state: wp.array2d[D],
            reference: wp.array2d[D], stats: wp.array2d[D]):
    w = wp.tid()
    if active[w] == 0:
        return
    qw = D(q[w, 3]); qx = D(q[w, 4]); qy = D(q[w, 5]); qz = D(q[w, 6])
    roll = wp.atan2(D(2) * (qw * qx + qy * qz), D(1) - D(2) * (qx * qx + qy * qy))
    offset = wp.clamp(D(.30) * roll + D(.12) * state[w, 5], D(-.035), D(.035))
    left = wp.max(D(0), reference[w, 3] - (reference[w, 2] + offset))
    right = wp.max(D(0), reference[w, 3] - (reference[w, 2] - offset))
    if left > D(1.e-12) or right > D(1.e-12):
        stats[w, 0] += D(1)
        if left > D(1.e-12): stats[w, 1] += D(1)
        if right > D(1.e-12): stats[w, 2] += D(1)
        stats[w, 3] = wp.max(stats[w, 3], wp.max(left, right))
        if stats[w, 4] < D(0): stats[w, 4] = state[w, 0] * D(.0005)
    for j in range(6):
        bound = allowed(D(v[w, ids[4 + j]]), j < 4)
        excess = wp.abs(D(ctrl[w, j])) - bound
        if excess > D(1.e-6): stats[w, 5] += D(1)
        stats[w, 6] = wp.max(stats[w, 6], excess)
        stats[w, 7] = wp.max(stats[w, 7], wp.abs(D(ctrl[w, j])) / wp.max(bound, D(1.e-12)))


def run(output):
    assert not output.exists(), output
    base = cases()[:6]
    scenarios = base + base
    n = len(scenarios)
    env = NativeEnv(n=n, scenario=scenarios, bank_factory=bank_height_115,
                    height_conditioned=True, height_design="range115", residual_scale=0)
    env.reset()
    refs = env.k["reference"].numpy()
    assert refs.shape == (n, 3) and np.allclose(refs[:, 2], .115)
    ref4 = wp.array(np.column_stack((refs, [0.] * len(base) + [.115] * len(base))), dtype=D)
    initial = np.zeros((n, 11)); initial[:, 4] = 1.; initial[:, 7] = -1.; initial[:, 8] = 1.
    chain = wp.array(initial, dtype=D)
    initial = np.zeros((n, 8)); initial[:, 4] = -1.
    metrics = wp.array(initial, dtype=D)
    passive = wp.array([env.cpu.joint(name).qposadr[0] for name in
                        ("passA_L", "passC_L", "passA_R", "passC_R")], dtype=wp.int32)
    b_offsets = wp.array(np.tile(np.asarray([[
        env.cpu.body("leg" + side + "_D").pos,
        env.cpu.body("kneeB_" + side).pos,
        env.cpu.site("couplerB_" + side + "_end").pos] for side in ("L", "R")]),
        (n, 1, 1, 1)), dtype=wp.vec3d)
    data = env.data
    with wp.ScopedCapture() as capture:
        wp.launch(begin, n, [env.reward])
        for _ in range(40):
            wp.launch(command_step, n, [env.state, env.param, env.command, env.active, data.qpos, data.qvel,
                data.qacc_warmstart, env.stopped_q, env.stopped_v, env.stopped_w, env.contact_flags])
            wp.launch(control, n, [data.qpos, data.qvel, data.sensordata, env.targets, env.command, env.active,
                env.k["state"], env.ids, env.k["heights"], env.k["gains"], env.k["feed"], env.k["angles"],
                ref4, env.k["yaw"], data.ctrl, env.diag, 0, 0], block_dim=32)
            wp.launch(monitor, n, [data.qpos, data.qvel, data.ctrl, env.ids, env.active,
                env.k["state"], ref4, metrics])
            mjw.step(env.model, data)
            wp.launch(audit, n, [data.qpos, data.qvel, data.ctrl, env.ids, passive, env.active,
                env.wheel_offsets, b_offsets, chain])
            wp.launch(reduce_contacts, data.naconmax, [data.nacon, data.contact.worldid, data.contact.geom,
                env.ids, env.contact_flags])
            wp.launch(after, n, [data.qpos, data.qvel, data.sensordata, data.qacc_warmstart, data.time,
                env.contact_flags, env.ids, env.param, env.command, env.state, env.k["state"], env.diag,
                env.residual, env.active, env.done, env.reward, env.obs, env.history,
                env.stopped_q, env.stopped_v, env.stopped_w, env.wheel_offsets], block_dim=32)
    env.graph = capture.graph
    rows = [None] * n
    zero = np.zeros((n, 3), np.float32)
    for _ in range(700):
        env.step_async(zero)
        done_before = env.done.numpy(); chain_before = chain.numpy(); metrics_before = metrics.numpy()
        _, _, _, infos = env.step_wait()
        for i in np.flatnonzero(done_before):
            if rows[i] is not None: continue
            info = infos[i]; a = chain_before[i]; m = metrics_before[i]
            pose = info["relative_peak_deg"] if info["attitude_mode"] == "terrain_relative" else info["peak_deg"]
            pose_pass = bool(max(pose) <= 5. + 1.e-9 and
                             (info["attitude_mode"] != "terrain_relative" or max(info["peak_deg"][:2]) <= 10. + 1.e-9))
            task_info = {key: value for key, value in info.items() if key != "terminal_observation"}
            rows[i] = dict(scenario=asdict(scenarios[i]), mode="baseline" if i < len(base) else "target_floor_115",
                task_info=task_info, min_fk_leg_m=float(info["min_leg_m"]),
                min_actual_chain_leg_m=float(a[8]), min_joint_margin_rad=float(a[4]),
                max_chain_vs_fk_m=float(a[9]), max_closed_chain_error_m=float(a[10]),
                peak_hip_rpm=float(a[5]), peak_hip_torque_Nm=float(a[6]), pose_passed=pose_pass,
                floor_physical_steps=int(m[0]), floor_left_steps=int(m[1]), floor_right_steps=int(m[2]),
                max_floor_raise_m=float(m[3]), first_floor_s=float(m[4]) if m[4] >= 0 else None,
                motor_bound_violations=int(m[5]), max_motor_bound_excess_Nm=float(m[6]),
                max_motor_bound_utilization=float(m[7]),
                full_gate=bool(info["success"] and pose_pass and a[8] >= HEIGHT_115_GEOMETRIC_MIN
                               and a[4] >= 0 and m[5] == 0))
        if all(row is not None for row in rows): break
    assert all(row is not None for row in rows)
    assert all(row["floor_physical_steps"] == 0 for row in rows[:len(base)])
    assert all(row["motor_bound_violations"] == 0 for row in rows)
    summary = {}
    for name, arm in (("baseline", rows[:len(base)]), ("target_floor_115", rows[len(base):])):
        summary[name] = dict(total=6, task_success=sum(r["task_info"]["success"] for r in arm),
            full_gate=sum(r["full_gate"] for r in arm),
            floor_physical_steps=sum(r["floor_physical_steps"] for r in arm),
            worst_actual_chain_leg_m=min(r["min_actual_chain_leg_m"] for r in arm),
            worst_joint_margin_rad=min(r["min_joint_margin_rad"] for r in arm))
    paired = [dict(terrain=base[i].terrain, speed=base[i].speed,
        baseline_full_gate=rows[i]["full_gate"], candidate_full_gate=rows[i+6]["full_gate"],
        baseline_actual_m=rows[i]["min_actual_chain_leg_m"], candidate_actual_m=rows[i+6]["min_actual_chain_leg_m"],
        baseline_joint_margin_rad=rows[i]["min_joint_margin_rad"], candidate_joint_margin_rad=rows[i+6]["min_joint_margin_rad"],
        floor_physical_steps=rows[i+6]["floor_physical_steps"], max_floor_raise_m=rows[i+6]["max_floor_raise_m"])
        for i in range(6)]
    sources = ("wheelleg_warp/probe_height_115_target_floor.py", "wheelleg_warp/probe_height_115_safety_reflex.py",
        "wheelleg_warp/probe_height_115_margin.py", "wheelleg_warp/native/controller.py",
        "wheelleg_warp/native/environment.py", "wheelleg_warp/native/terrain.py", "wheelleg_warp/native/models.py",
        "wheelleg_ppo/tools/rm_controller.py", "wheelleg_ppo/tools/model_lqr.py",
        "wheelleg_ppo/tools/wheelleg_sim.py", "wheelleg_ppo/tools/hardware_profile.py",
        "wheelleg_ppo/xml/wheelleg.xml")
    output.mkdir(parents=True)
    rows_path = output / "rows.json"
    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (output / "verification.json").write_text(json.dumps(dict(
        protocol="six public 0.115 m normal cases x baseline vs radial target floor at nominal height",
        nominal_target_m=.115, geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,
        floor_m=.115, training=False, final_holdout_opened=False,
        summary=summary, paired=paired,
        rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}),
        ensure_ascii=False, indent=2) + "\n")
    print(summary)
    if summary["target_floor_115"]["full_gate"] < 6:
        raise AssertionError("目标下限投影未过六个正常场景联合门，停止该固定候选")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
