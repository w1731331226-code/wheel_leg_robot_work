"""独立 2 kHz 回放：核对 0.115 m 主动 FK、被动闭链与模型关节限位。"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import mujoco
import numpy as np
import warp as wp
import wheelleg_sim as sim
from native.environment import NativeEnv
from native.terrain import bank_height_115
from probe_height_115_margin import cases, lengths
from probe_height_lift_workspace import SIM_TRANSIENT_MIN
from trace_first_divergence_v2 import graph

OLD = ROOT / "wheelleg_warp/results/height_115_margin_20260928"
JOINTS = ("alphaL", "betaL", "passA_L", "passC_L",
          "alphaR", "betaR", "passA_R", "passC_R")


def ry(point, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c * point[0] + s * point[2], np.full_like(c, point[1]),
                     -s * point[0] + c * point[2]), axis=-1)


def geometry(model, q):
    fk_len, actual_len, closure, point_error = [], [], [], []
    local_a, local_b = [], []
    for side in ("L", "R"):
        alpha = q[..., model.joint("alpha" + side).qposadr[0]]
        beta = q[..., model.joint("beta" + side).qposadr[0]]
        pass_a = q[..., model.joint("passA_" + side).qposadr[0]]
        pass_c = q[..., model.joint("passC_" + side).qposadr[0]]
        leg = model.body_pos[model.body("leg" + side).id]
        leg_d = model.body_pos[model.body("leg" + side + "_D").id]
        a = leg + ry(model.body_pos[model.body("kneeA_" + side).id], alpha)
        a += ry(model.body_pos[model.body("wheel" + side).id], alpha + pass_a)
        b = leg_d + ry(model.body_pos[model.body("kneeB_" + side).id], beta)
        b += ry(model.site_pos[model.site("couplerB_" + side + "_end").id], beta + pass_c)

        # 与生产控制 FK 相同的低位几何支；保留轮心坐标以发现单纯长度比较遗漏的误差。
        p1, p4 = sim.PHI1_STAND - alpha, sim.PHI4_STAND - beta
        bx, bz = sim.L1 * np.cos(p1), sim.L1 * np.sin(p1)
        dx, dz = sim.L5 + sim.L4 * np.cos(p4), sim.L4 * np.sin(p4)
        ux, uz = dx - bx, dz - bz
        aa, bb = 2 * sim.L2 * ux, 2 * sim.L2 * uz
        cc = sim.L2 ** 2 + ux ** 2 + uz ** 2 - sim.L3 ** 2
        phi2 = 2 * np.arctan2(bb - np.sqrt(np.maximum(0, aa ** 2 + bb ** 2 - cc ** 2)), aa + cc)
        ideal = np.stack((leg[0] + bx + sim.L2 * np.cos(phi2),
                          np.full_like(alpha, leg[1]), leg[2] + bz + sim.L2 * np.sin(phi2)), axis=-1)
        middle = (leg + leg_d) / 2
        fk_len.append(np.linalg.norm(ideal - middle, axis=-1))
        actual_len.append(np.linalg.norm(a - middle, axis=-1))
        closure.append(np.linalg.norm(a - b, axis=-1))
        point_error.append(np.linalg.norm(a - ideal, axis=-1))
        local_a.append(a)
        local_b.append(b)
    return (np.stack(fk_len, axis=-1), np.stack(actual_len, axis=-1),
            np.stack(closure, axis=-1), np.stack(point_error, axis=-1),
            np.stack(local_a, axis=-2), np.stack(local_b, axis=-2))


def first_time(times, valid, condition):
    hit = np.flatnonzero(valid & condition)
    return float(times[hit[0]]) if len(hit) else None


def cpu_site_check(model, q, local_a, local_b, indices):
    data = mujoco.MjData(model)
    peak = 0.0
    for t, w in indices:
        pose = q[t, w].astype(float)
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, pose[3:7])
        rotation = mat.reshape(3, 3)
        for side, name in enumerate(("L", "R")):
            for local, site in ((local_a, "wheel_axle_" + name),
                                (local_b, "couplerB_" + name + "_end")):
                point = pose[:3] + rotation @ local[t, w, side]
                peak = max(peak, float(np.max(np.abs(point - data.site_xpos[model.site(site).id]))))
    assert peak < 3e-7, peak
    return peak


def run(output):
    assert not output.exists(), output
    old = json.loads((OLD / "verification.json").read_text())
    scenarios = cases()
    assert old["role"] == "public_2khz_margin_diagnostic"
    assert [asdict(s) for s in scenarios] == [r["scenario"] for r in old["rows"]]
    for name, digest in old["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    env = NativeEnv(n=len(scenarios), scenario=scenarios, bank_factory=bank_height_115,
                    height_conditioned=True, height_design="range115", residual_scale=0)
    try:
        env.reset()
        model = env.cpu
        joint_ids = np.array([model.joint(name).qposadr[0] for name in JOINTS])
        joint_ranges = np.array([model.jnt_range[model.joint(name).id] for name in JOINTS])
        graph_2khz, buffer, _ = graph(env, False)
        env.targets.assign(np.zeros((len(scenarios), 3), np.float32))
        nq = model.nq
        columns = {name: [] for name in ("time_s", "qpos", "fk_length_m", "actual_length_m",
                                        "closure_error_m", "fk_wheel_error_m", "joint_margin_rad",
                                        "wheel_a_local_m", "coupler_b_local_m")}
        for _ in range(700):
            wp.capture_launch(graph_2khz)
            chunk = buffer.numpy()
            q = chunk[:, :, 1:1 + nq]
            fk, actual, closure, point_error, local_a, local_b = geometry(model, q)
            np.testing.assert_allclose(fk[:, :, 0], lengths(q[:, :, joint_ids[0]], q[:, :, joint_ids[1]]), atol=1e-10, rtol=0)
            np.testing.assert_allclose(fk[:, :, 1], lengths(q[:, :, joint_ids[4]], q[:, :, joint_ids[5]]), atol=1e-10, rtol=0)
            angles = q[:, :, joint_ids]
            margins = np.minimum(angles - joint_ranges[:, 0], joint_ranges[:, 1] - angles)
            for name, value in (("time_s", chunk[:, :, 0]), ("qpos", q.astype(np.float32)), ("fk_length_m", fk),
                                ("actual_length_m", actual), ("closure_error_m", closure),
                                ("fk_wheel_error_m", point_error), ("joint_margin_rad", margins),
                                ("wheel_a_local_m", local_a), ("coupler_b_local_m", local_b)):
                columns[name].append(value.copy())
            if np.all(env.done.numpy() != 0):
                break
        assert np.all(env.done.numpy() != 0)
        trace = {name: np.concatenate(value) for name, value in columns.items()}
        time = trace["time_s"]
        live = np.ones(time.shape, bool)
        live[1:] = time[1:] - time[:-1] > 1e-9
        assert np.all(np.diff(time, axis=0)[live[1:]] > 0)
        trace["live_sample"] = live
        old_trace = np.load(OLD / "length_trace.npz")
        paired = {"old_samples": int(old_trace["time_s"].shape[0]), "new_samples": int(time.shape[0])}
        if old_trace["time_s"].shape == time.shape:
            paired["max_time_delta_s"] = float(np.max(np.abs(time - old_trace["time_s"])))
            paired["max_fk_delta_m"] = float(np.max(np.abs(trace["fk_length_m"] - old_trace["leg_length_m"])))

        rows, selected = [], set()
        for w, scenario in enumerate(scenarios):
            valid = live[:, w]
            row = {"scenario": asdict(scenario), "terminal_reason_code": int(env.done.numpy()[w]),
                   "first_active_joint_limit_s": first_time(time[:, w], valid,
                       np.any(trace["joint_margin_rad"][:, w, (0, 1, 4, 5)] < 0, axis=-1)),
                   "first_passive_joint_limit_s": first_time(time[:, w], valid,
                       np.any(trace["joint_margin_rad"][:, w, (2, 3, 6, 7)] < 0, axis=-1)),
                   "min_active_joint_margin_rad": float(np.min(trace["joint_margin_rad"][valid, w][:, (0, 1, 4, 5)])),
                   "min_passive_joint_margin_rad": float(np.min(trace["joint_margin_rad"][valid, w][:, (2, 3, 6, 7)]))}
            for side, label in enumerate(("L", "R")):
                fk = trace["fk_length_m"][:, w, side]
                actual = trace["actual_length_m"][:, w, side]
                close = trace["closure_error_m"][:, w, side]
                point = trace["fk_wheel_error_m"][:, w, side]
                ix = np.flatnonzero(valid)
                row[label] = {"min_fk_length_m": float(fk[valid].min()),
                              "min_actual_length_m": float(actual[valid].min()),
                              "max_closure_error_m": float(close[valid].max()),
                              "max_fk_wheel_error_m": float(point[valid].max()),
                              "max_abs_length_difference_m": float(np.max(np.abs(actual[valid] - fk[valid]))),
                              "first_fk_below_proxy_s": first_time(time[:, w], valid, fk < SIM_TRANSIENT_MIN - 1e-9),
                              "first_actual_below_proxy_s": first_time(time[:, w], valid, actual < SIM_TRANSIENT_MIN - 1e-9),
                              "first_closure_over_0p1mm_s": first_time(time[:, w], valid, close > 1e-4)}
                selected.update(((int(ix[np.argmin(actual[valid])]), w),
                                 (int(ix[np.argmax(close[valid])]), w)))
                breach = np.flatnonzero(valid & (actual < SIM_TRANSIENT_MIN - 1e-9))
                if len(breach):
                    selected.add((int(breach[0]), w))
            selected.add((0, w))
            rows.append(row)

        cpu_error = cpu_site_check(model, trace["qpos"], trace["wheel_a_local_m"],
                                   trace["coupler_b_local_m"], selected)
        trace.pop("wheel_a_local_m")
        trace.pop("coupler_b_local_m")
        output.mkdir(parents=True)
        np.savez_compressed(output / "passive_trace.npz", **trace)
        sources = list(old["source_sha256"]) + ["wheelleg_warp/probe_height_115_passive.py"]
        summary = {"total": len(rows),
                   "actual_below_proxy": sum(any(row[s]["first_actual_below_proxy_s"] is not None for s in ("L", "R")) for row in rows),
                   "fk_below_proxy": sum(any(row[s]["first_fk_below_proxy_s"] is not None for s in ("L", "R")) for row in rows),
                   "active_joint_limit": sum(row["first_active_joint_limit_s"] is not None for row in rows),
                   "passive_joint_limit": sum(row["first_passive_joint_limit_s"] is not None for row in rows),
                   "max_closure_error_m": max(row[s]["max_closure_error_m"] for row in rows for s in ("L", "R")),
                   "max_fk_wheel_error_m": max(row[s]["max_fk_wheel_error_m"] for row in rows for s in ("L", "R")),
                   "min_actual_length_m": min(row[s]["min_actual_length_m"] for row in rows for s in ("L", "R")),
                   "cpu_site_check_max_abs_m": cpu_error}
        payload = {"role": "public_2khz_passive_chain_diagnostic", "training": False,
                   "final_holdout_opened": False, "nominal_target_m": .115,
                   "simulated_proxy_min_m": SIM_TRANSIENT_MIN,
                   "closure_0p1mm_is_diagnostic_only": True,
                   "source_margin_verification_sha256": hashlib.sha256((OLD / "verification.json").read_bytes()).hexdigest(),
                   "source_margin_trace_sha256": old["trace_sha256"],
                   "model_xml_sha256": hashlib.sha256((ROOT / "wheelleg_ppo/xml/wheelleg.xml").read_bytes()).hexdigest(),
                   "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
                   "trace_sha256": hashlib.sha256((output / "passive_trace.npz").read_bytes()).hexdigest(),
                   "paired_old_margin": paired, "summary": summary, "rows": rows}
        (output / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"summary": summary, "paired_old_margin": paired}, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
