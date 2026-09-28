"""0.115 m 首次失效前：仅用当前主动关节与上一帧速度预测 5/10/20 ms 安全量。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import numpy as np
from native.terrain import HEIGHT_115_GEOMETRIC_MIN, HeightTerrainScenario, model
from probe_height_115_margin import lengths
from probe_height_115_passive import JOINTS, geometry

DT = .0005
HORIZONS = (10, 20, 40)
ACTIVE = ("alphaL", "betaL", "alphaR", "betaR")


def run(output):
    assert not output.exists(), output
    source = ROOT / "wheelleg_warp/results/height_115_local_states_20260928/verification.json"
    archive = json.loads(source.read_text())
    data_path = source.parent / "windows.npz"
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == archive["windows_sha256"]
    windows = np.load(data_path)
    columns = archive["columns"]
    selected = [min((i for i, e in enumerate(archive["events"]) if e["world"] == w),
                    key=lambda i: archive["events"][i]["reference_s"]) for w in range(6)]
    rows = []
    for event_id in selected:
        event = archive["events"][event_id]
        m = model(HeightTerrainScenario(**event["scenario"]))
        pre = windows[f"event_{event_id}_pre"]
        post = windows[f"event_{event_id}_post"]
        assert np.max(abs(post[:, 0] - pre[:, 0] - DT)) < 1e-6
        qcol, vcol = columns["qpos_start"], columns["qvel_start"]
        qa = np.array([m.joint(name).qposadr[0] for name in ACTIVE])
        va = np.array([m.joint(name).dofadr[0] for name in ACTIVE])
        joint_ids = np.array([m.joint(name).qposadr[0] for name in JOINTS])
        joint_ranges = np.asarray([m.jnt_range[m.joint(name).id] for name in JOINTS])
        current = pre[:, qcol:qcol + m.nq]
        future = post[:, qcol:qcol + m.nq]
        fk, _, _, _, _, _ = geometry(m, current)
        check = np.stack((lengths(current[:, qa[0]], current[:, qa[1]]),
                          lengths(current[:, qa[2]], current[:, qa[3]])), axis=1)
        assert np.max(abs(fk - check)) < 1e-9
        _, actual, _, _, _, _ = geometry(m, future)
        true_length = actual.min(axis=1)
        angles = future[:, joint_ids]
        true_joint = np.minimum(angles - joint_ranges[:, 0], joint_ranges[:, 1] - angles).min(axis=1)
        for steps in HORIZONS:
            starts = np.flatnonzero(pre[:, 0] < event["reference_s"] - 1e-7)
            starts = starts[(starts > 0) & (starts + steps <= len(pre))]
            assert len(starts) > 0
            q = current[starts[:, None], qa].reshape(-1, 4)
            v = pre[starts[:, None], vcol + va].reshape(-1, 4)
            previous = pre[(starts - 1)[:, None], vcol + va].reshape(-1, 4)
            t = (np.arange(1, steps + 1) * DT)[:, None, None]
            observed_length = np.array([true_length[i:i + steps].min() for i in starts])
            observed_joint = np.array([true_joint[i:i + steps].min() for i in starts])
            for name, acceleration in (("constant_velocity", np.zeros_like(v)),
                                       ("last_step_acceleration", (v - previous) / DT)):
                predicted = q[None, :, :] + v[None, :, :] * t + .5 * acceleration[None, :, :] * t * t
                forecast_length = np.minimum(lengths(predicted[..., 0], predicted[..., 1]),
                                             lengths(predicted[..., 2], predicted[..., 3])).min(axis=0)
                forecast_joint = (1.5 - np.abs(predicted)).min(axis=(0, 2))
                false_safe = (forecast_length >= HEIGHT_115_GEOMETRIC_MIN) & (forecast_joint >= 0) & \
                             ((observed_length < HEIGHT_115_GEOMETRIC_MIN) | (observed_joint < 0))
                rows.append(dict(world=event["world"], event=event["kind"], horizon_ms=steps * DT * 1000,
                    model=name, starts=len(starts), false_safe=int(false_safe.sum()),
                    max_optimistic_length_error_m=float(np.max(forecast_length - observed_length)),
                    max_optimistic_joint_error_rad=float(np.max(forecast_joint - observed_joint)),
                    first_false_safe_s=float(pre[starts[np.flatnonzero(false_safe)[0]], 0]) if false_safe.any() else None,
                    max_true_length_shortfall_m=float(np.max(HEIGHT_115_GEOMETRIC_MIN - observed_length))))
    summary = {name: {str(h * DT * 1000): dict(false_safe=sum(r["false_safe"] for r in rows
        if r["model"] == name and r["horizon_ms"] == h * DT * 1000),
        worst_optimistic_length_error_m=max(r["max_optimistic_length_error_m"] for r in rows
        if r["model"] == name and r["horizon_ms"] == h * DT * 1000)) for h in HORIZONS}
        for name in ("constant_velocity", "last_step_acceleration")}
    names = ("wheelleg_warp/probe_height_115_predictability.py", "wheelleg_warp/probe_height_115_margin.py",
             "wheelleg_warp/probe_height_115_passive.py", "wheelleg_warp/native/terrain.py",
             "wheelleg_warp/native/models.py", "wheelleg_ppo/tools/wheelleg_sim.py",
             "wheelleg_ppo/xml/wheelleg.xml")
    output.mkdir(parents=True)
    (output / "verification.json").write_text(json.dumps(dict(
        role="offline_baseline_forecast_error_not_a_safety_controller", nominal_target_m=.115,
        geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN, source_window_sha256=archive["windows_sha256"],
        source_state_verification_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        prediction_inputs="current active q/dq and optional previous active dq only",
        evaluation_uses="future actual A-chain length and all eight joint ranges",
        no_motor_action_modeled=True, training=False, final_holdout_opened=False,
        summary=summary, rows=rows,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}),
        ensure_ascii=False, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
