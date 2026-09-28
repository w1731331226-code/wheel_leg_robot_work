"""2 kHz核查新0.115 m名义目标的实际腿长是否越过模型下限加20 mm。"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import numpy as np
import warp as wp
import wheelleg_sim as sim
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario, bank_height_115
from probe_height_lift_workspace import SIM_TRANSIENT_MIN
from trace_first_divergence_v2 import graph


def lengths(alpha, beta):
    p1, p4 = sim.PHI1_STAND - alpha, sim.PHI4_STAND - beta
    bx, bz = sim.L1 * np.cos(p1), sim.L1 * np.sin(p1)
    dx, dz = sim.L5 + sim.L4 * np.cos(p4), sim.L4 * np.sin(p4)
    ux, uz = dx - bx, dz - bz
    aa, bb = 2 * sim.L2 * ux, 2 * sim.L2 * uz
    cc = sim.L2 ** 2 + ux ** 2 + uz ** 2 - sim.L3 ** 2
    phi2 = 2 * np.arctan2(bb - np.sqrt(np.maximum(0., aa ** 2 + bb ** 2 - cc ** 2)), aa + cc)
    cx, cz = bx + sim.L2 * np.cos(phi2), bz + sim.L2 * np.sin(phi2)
    return np.hypot(cx - sim.L5 / 2, cz)


def cases():
    h = .115
    normal = [HeightTerrainScenario(speed=.5, stand_height_m=h, terrain=name,
                                    terrain_seed=1151234, relative_attitude=True, **extra)
              for name, extra in (("legacy", {}), ("step", {"step_height_m": .02}),
                                  ("cross_slope", {"grade_deg": 3.}), ("rough", {"roughness_m": .004}))]
    normal += [HeightTerrainScenario(speed=speed, stand_height_m=h) for speed in (1., -1.)]
    boundary = [HeightTerrainScenario(speed=speed, stand_height_m=h,
                                      **({"height_l": .02} if side == "L" else {"height_r": .02}))
                for side in ("L", "R") for speed in (.5, 1., -.5, -1.)]
    return normal + boundary


def run(output):
    assert not output.exists()
    scenarios = cases()
    env = NativeEnv(n=len(scenarios), scenario=scenarios, bank_factory=bank_height_115,
                    height_conditioned=True, height_design="range115", residual_scale=0)
    env.reset()
    captured, buffer, _ = graph(env, False)
    env.targets.assign(np.zeros((len(scenarios), 3), np.float32))
    ids = env.ids.numpy()
    qcol, vcol = 1, 1 + env.cpu.nq
    ccol = vcol + env.cpu.nv
    sample_times, sample_lengths = [], []
    min_joint_margin = np.full(len(scenarios), np.inf)
    peak_hip_rpm = np.zeros(len(scenarios))
    peak_hip_torque = np.zeros(len(scenarios))
    for _ in range(700):
        wp.capture_launch(captured)
        chunk = buffer.numpy()
        length = np.stack((lengths(chunk[:, :, qcol + ids[0]], chunk[:, :, qcol + ids[1]]),
                           lengths(chunk[:, :, qcol + ids[2]], chunk[:, :, qcol + ids[3]])), axis=-1)
        if not sample_lengths:
            for side in range(2):
                a = float(chunk[0, 0, qcol + ids[2 * side]])
                b = float(chunk[0, 0, qcol + ids[2 * side + 1]])
                assert abs(length[0, 0, side] - sim.fk_joints(a, b)["leg_len"]) < 1e-10
        sample_times.append(chunk[:, :, 0].copy())
        sample_lengths.append(length)
        angles = chunk[:, :, qcol + ids[:4]]
        min_joint_margin = np.minimum(min_joint_margin, np.min(1.5 - np.abs(angles), axis=(0, 2)))
        peak_hip_rpm = np.maximum(peak_hip_rpm, np.max(np.abs(chunk[:, :, vcol + ids[4:8]]), axis=(0, 2)) * 60 / (2 * np.pi))
        peak_hip_torque = np.maximum(peak_hip_torque, np.max(np.abs(chunk[:, :, ccol:ccol + 4]), axis=(0, 2)))
        if np.all(env.done.numpy() != 0):
            break
    assert np.all(env.done.numpy() != 0)
    times = np.concatenate(sample_times)
    actual = np.concatenate(sample_lengths)
    rows = []
    for w, scenario in enumerate(scenarios):
        first = []
        for side in range(2):
            hit = np.flatnonzero(actual[:, w, side] < SIM_TRANSIENT_MIN - 1e-9)
            first.append(float(times[hit[0], w]) if len(hit) else None)
        rows.append(dict(scenario=asdict(scenario), min_leg_m=np.min(actual[:, w], axis=0).tolist(),
                         first_below_proxy_s=first, min_active_joint_margin_rad=float(min_joint_margin[w]),
                         peak_hip_rpm=float(peak_hip_rpm[w]), peak_hip_torque_Nm=float(peak_hip_torque[w]),
                         terminal_reason_code=int(env.done.numpy()[w])))
    output.mkdir(parents=True)
    np.savez_compressed(output / "length_trace.npz", time_s=times, leg_length_m=actual)
    names = ("wheelleg_warp/probe_height_115_margin.py", "wheelleg_warp/probe_height_lift_workspace.py",
             "wheelleg_warp/trace_first_divergence_v2.py", "wheelleg_warp/native/controller.py",
             "wheelleg_warp/native/environment.py", "wheelleg_warp/native/terrain.py",
             "wheelleg_warp/native/models.py", "wheelleg_ppo/tools/rm_controller.py",
             "wheelleg_ppo/tools/model_lqr.py", "wheelleg_ppo/tools/wheelleg_sim.py",
             "wheelleg_ppo/tools/hardware_profile.py", "wheelleg_ppo/tools/state_estimation.py",
             "wheelleg_ppo/xml/wheelleg.xml")
    summary = dict(total=len(rows), any_below_proxy=sum(any(t is not None for t in r["first_below_proxy_s"]) for r in rows),
                   lowest_observed_leg_m=min(min(r["min_leg_m"]) for r in rows),
                   simulated_proxy_min_m=SIM_TRANSIENT_MIN)
    (output / "verification.json").write_text(json.dumps(dict(role="public_2khz_margin_diagnostic",
        nominal_target_m=.115, training=False, final_holdout_opened=False, summary=summary, rows=rows,
        trace_sha256=hashlib.sha256((output / "length_trace.npz").read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}),
        ensure_ascii=False, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
