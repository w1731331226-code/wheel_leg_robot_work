"""原CPU硬件模型上的独立左右腿25 mm抬轮轨迹诊断；不改默认控制。"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import wheelleg_sim as sim
from probe_height_lift_workspace import HEIGHTS, LIFT, MODEL_JOINT_MIN, SIM_TRANSIENT_MIN, leg_pose

KP, KD = 80., 4.
COLUMNS = ("time_s", "hit_clearance_m", "support_clearance_m", "hit_ground_N",
           "support_ground_N", "roll_deg", "pitch_deg", "yaw_deg", "peak_hip_rpm",
           "peak_hip_torque_Nm", "min_joint_margin_rad", "hit_leg_m", "support_leg_m",
           "trajectory_fraction", "chassis_z_m")


def phase(t):
    if .5 <= t < .6:
        return (t - .5) / .1, 10.
    if .6 <= t < .7:
        return 1., 0.
    if .7 <= t < .8:
        return (.8 - t) / .1, -10.
    return 0., 0.


def ground_forces(model, data, floor, wheels):
    out = [0., 0.]
    force = np.zeros(6)
    nonwheel = False
    for i in range(data.ncon):
        contact = data.contact[i]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor not in pair:
            continue
        side = 0 if wheels[0] in pair else 1 if wheels[1] in pair else None
        if side is None:
            nonwheel = True
            continue
        mujoco.mj_contactForce(model, data, i, force)
        out[side] += max(0., float(force[0]))
    return out, nonwheel


def rollout(model, data, height, hit, intervention, leg_min):
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.qpos[2] = height + sim.hw.WHEEL_RADIUS
    for side in (0, 1):
        leg_pose(model, data.qpos, side, height)
    mujoco.mj_forward(model, data)
    state = sim.make_state(model, hardware=True, six_state=True)
    state.leg_ref = height - sim.L_STAND
    state.L_cur = state.L_prev = height
    target_hit = max(leg_min, height - LIFT)
    target_support = target_hit + LIFT
    target = [target_support, target_support]
    target[hit] = target_hit
    q0 = np.tile(np.asarray(sim.ik(height)), 2)
    q1 = np.concatenate([sim.ik(length) for length in target])
    motors = ("motor_alphaL", "motor_betaL", "motor_alphaR", "motor_betaR")
    joints = ("alphaL", "betaL", "alphaR", "betaR")
    passive = ("passA_L", "passC_L", "passA_R", "passC_R")
    qids = [model.joint(name).qposadr[0] for name in joints]
    vids = [model.joint(name).dofadr[0] for name in joints]
    aids = [model.actuator(name).id for name in motors]
    floor = model.geom("floor").id
    wheels = (model.geom("wheel_collide_L").id, model.geom("wheel_collide_R").id)
    bodies = (model.body("wheelL").id, model.body("wheelR").id)
    trace = np.zeros((2000, len(COLUMNS)), np.float64)
    clipped_steps, nonwheel_contact, peak_bound_violation = 0, False, 0.
    recorded = 0
    for step in range(2000):
        t = float(data.time)
        prior = state.motor_peak_t.copy()
        sim.control(model, data, state)
        fraction, derivative = phase(t) if intervention else (0., 0.)
        if intervention and (derivative or fraction):
            clipped = False
            for j, (name, aid, qid, vid) in enumerate(zip(motors, aids, qids, vids)):
                speed = float(data.qvel[vid])
                desired_q = q0[j] + fraction * (q1[j] - q0[j])
                desired_speed = derivative * (q1[j] - q0[j])
                request = float(data.ctrl[aid]) + KP * (desired_q - data.qpos[qid]) + KD * (desired_speed - speed)
                applied, timer = sim.hw.torque_limit(request, speed, True, prior.get(name, 0.), model.opt.timestep)
                low, high = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = np.clip(applied, low, high)
                state.motor_peak_t[name] = timer
                clipped |= abs(request - data.ctrl[aid]) > 1e-8
            clipped_steps += clipped
        for aid, vid in zip(aids, vids):
            speed = float(data.qvel[vid])
            bound, _ = sim.hw.torque_limit(float("inf"), speed, True, 0., model.opt.timestep)
            peak_bound_violation = max(peak_bound_violation, abs(float(data.ctrl[aid])) - bound)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            break
        forces, bad_contact = ground_forces(model, data, floor, wheels)
        nonwheel_contact |= bad_contact
        clearance = [float(data.xpos[body, 2] - sim.hw.WHEEL_RADIUS) for body in bodies]
        angles = np.degrees(sim.euler(data))
        margin = min(float(min(data.qpos[model.joint(name).qposadr[0]] - model.jnt_range[model.joint(name).id, 0],
                               model.jnt_range[model.joint(name).id, 1] - data.qpos[model.joint(name).qposadr[0]]))
                     for name in joints + passive)
        rpm = max(abs(float(data.qvel[vid])) * 60 / (2 * math.pi) for vid in vids)
        torque = max(abs(float(data.ctrl[aid])) for aid in aids)
        lengths = [sim.fk_joints(float(data.qpos[qids[2 * side]]),
                                 float(data.qpos[qids[2 * side + 1]]))["leg_len"] for side in (0, 1)]
        trace[step] = (data.time, clearance[hit], clearance[1 - hit], forces[hit], forces[1 - hit],
                       *angles, rpm, torque, margin, lengths[hit], lengths[1 - hit], fraction, data.qpos[2])
        recorded = step + 1
    trace = trace[:recorded]
    hold = trace[(trace[:, 0] >= .6) & (trace[:, 0] < .7)]
    valid = ((hold[:, 1] >= .02) & (hold[:, 3] <= 1.) & (hold[:, 4] > 1.) &
             (np.max(np.abs(hold[:, 5:8]), axis=1) <= 5.)) if len(hold) else np.zeros(0, bool)
    longest = current = 0
    for value in valid:
        current = current + 1 if value else 0
        longest = max(longest, current)
    result = dict(height_m=height, obstacle_side="L" if hit == 0 else "R", intervention=intervention,
                  obstacle_leg_target_m=target_hit, support_leg_target_m=target_support,
                  recorded_steps=recorded, max_hit_clearance_m=float(np.max(trace[:, 1])) if recorded else None,
                  max_support_clearance_m=float(np.max(trace[:, 2])) if recorded else None,
                  peak_attitude_deg=np.max(np.abs(trace[:, 5:8]), axis=0).tolist() if recorded else None,
                  max_hip_rpm=float(np.max(trace[:, 8])) if recorded else None,
                  max_hip_torque_Nm=float(np.max(trace[:, 9])) if recorded else None,
                  min_joint_margin_rad=float(np.min(trace[:, 10])) if recorded else None,
                  clipped_physical_steps=int(clipped_steps), max_motor_bound_violation_Nm=float(peak_bound_violation),
                  nonwheel_floor_contact=nonwheel_contact, longest_valid_hold_steps=longest,
                  lift_gate=bool(longest >= 40 and recorded == 2000 and not nonwheel_contact and
                                 np.max(np.abs(trace[:, 5:8])) <= 5. and np.min(trace[:, 10]) > 0 and
                                 peak_bound_violation <= 1e-8))
    return result, trace


def run(output, leg_min):
    assert not output.exists()
    assert SIM_TRANSIENT_MIN <= leg_min <= sim.L_SQUAT_MIN
    model, data = sim.load_model(str(ROOT / "wheelleg_ppo/xml/wheelleg.xml"), hardware=True)
    rows, traces = [], []
    for height in HEIGHTS:
        for hit in (0, 1):
            for intervention in (False, True):
                row, trace = rollout(model, data, height, hit, intervention, leg_min)
                rows.append(row)
                traces.append(trace)
                print(height, row["obstacle_side"], intervention, row["lift_gate"],
                      round(row["max_hit_clearance_m"] or 0., 4), flush=True)
    output.mkdir(parents=True)
    np.savez_compressed(output / "trace.npz", **{f"world_{i}": trace for i, trace in enumerate(traces)},
                        columns=np.asarray(COLUMNS))
    names = ("wheelleg_warp/probe_height_lift_cpu.py", "wheelleg_warp/probe_height_lift_workspace.py",
             "wheelleg_ppo/tools/wheelleg_sim.py", "wheelleg_ppo/tools/hardware_profile.py",
             "wheelleg_ppo/tools/rm_controller.py", "wheelleg_ppo/xml/wheelleg.xml")
    payload = dict(protocol="CPU hardware six-state flat, 25mm leg pair, 100ms up/hold/down; KP80 KD4; same motor envelope",
                   leg_min_m=leg_min, nominal_task_range_m=[sim.L_SQUAT_MIN, sim.L_MAX],
                   model_joint_limit_min_m=MODEL_JOINT_MIN, simulated_hardware_proxy_min_m=SIM_TRANSIENT_MIN,
                   training=False, obstacle_present=False, columns=COLUMNS, rows=rows,
                   summary=dict(lift_gate=sum(row["lift_gate"] for row in rows if row["intervention"]),
                                total_intervention=12, baseline_gate=sum(row["lift_gate"] for row in rows if not row["intervention"])),
                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
                   trace_sha256=hashlib.sha256((output / "trace.npz").read_bytes()).hexdigest())
    (output / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("SUMMARY", payload["summary"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bound", choices=("nominal", "transient"), required=True)
    args = parser.parse_args()
    run(args.output, sim.L_SQUAT_MIN if args.bound == "nominal" else SIM_TRANSIENT_MIN)
