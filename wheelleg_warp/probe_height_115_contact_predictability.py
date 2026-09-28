"""按首次新增接触是否已进入主动关节速度，复核 0.115 m 的 5/10 ms 外推。"""
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
ACTIVE = ("alphaL", "betaL", "alphaR", "betaR")


def run(output):
    assert not output.exists(), output
    source = ROOT / "wheelleg_warp/results/height_115_local_states_20260928/verification.json"
    archive = json.loads(source.read_text())
    path = source.parent / "windows.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == archive["windows_sha256"]
    windows = np.load(path)
    cols = archive["columns"]
    rows = []
    for world, terrain in ((1, "step"), (2, "cross_slope"), (3, "rough")):
        event_id = min((i for i, e in enumerate(archive["events"]) if e["world"] == world),
                       key=lambda i: archive["events"][i]["reference_s"])
        event = archive["events"][event_id]
        assert event["kind"] == "actual_leg" and event["scenario"]["terrain"] == terrain
        m = model(HeightTerrainScenario(**event["scenario"]))
        pre = windows[f"event_{event_id}_pre"]
        post = windows[f"event_{event_id}_post"]
        contacts = windows[f"event_{event_id}_contacts"]
        assert np.max(abs(post[:, 0] - pre[:, 0] - DT)) < 2e-6
        sets = [{tuple(sorted((int(c[1]), int(c[2])))) for c in step if int(c[0]) == world}
                for step in contacts]
        # contacts[k] 是 pre[k] 的求解接触；冲击影响 post[k]，从 pre[k+1] 才可由 dq 观测。
        changes = [k for k in range(1, len(pre)) if sets[k] - sets[k - 1]
                   and event["reference_s"] - .02 < pre[k, 0] < event["reference_s"]]
        assert len(changes) == 1, (world, changes)
        onset = changes[0]
        assert onset + 1 < len(pre) and abs(pre[onset + 1, 0] - post[onset, 0]) < 2e-6
        assert np.max(abs(pre[1:, cols["qpos_start"]:cols["warmstart_start"]] -
                          post[:-1, cols["qpos_start"]:cols["warmstart_start"]])) == 0
        qa = np.array([m.joint(name).qposadr[0] for name in ACTIVE])
        va = np.array([m.joint(name).dofadr[0] for name in ACTIVE])
        ji = np.array([m.joint(name).qposadr[0] for name in JOINTS])
        jr = np.array([m.jnt_range[m.joint(name).id] for name in JOINTS])
        q = pre[:, cols["qpos_start"]:cols["qpos_start"] + m.nq]
        future = post[:, cols["qpos_start"]:cols["qpos_start"] + m.nq]
        _, actual, _, _, _, _ = geometry(m, future)
        actual_min = actual.min(axis=1)
        angles = future[:, ji]
        joint_min = np.minimum(angles - jr[:, 0], jr[:, 1] - angles).min(axis=1)
        for steps in (10, 20):
            starts = np.flatnonzero(pre[:, 0] < event["reference_s"] - 1e-7)
            starts = starts[(starts > 0) & (starts + steps <= len(pre))]
            v = pre[starts[:, None], cols["qvel_start"] + va].reshape(-1, 4)
            old_v = pre[(starts - 1)[:, None], cols["qvel_start"] + va].reshape(-1, 4)
            t = (np.arange(1, steps + 1) * DT)[:, None, None]
            predicted = q[starts[:, None], qa].reshape(-1, 4)[None] + v[None] * t + \
                .5 * ((v - old_v) / DT)[None] * t * t
            forecast_length = np.minimum(lengths(predicted[..., 0], predicted[..., 1]),
                                         lengths(predicted[..., 2], predicted[..., 3])).min(axis=0)
            forecast_joint = (1.5 - abs(predicted)).min(axis=(0, 2))
            observed_length = np.array([actual_min[i:i + steps].min() for i in starts])
            observed_joint = np.array([joint_min[i:i + steps].min() for i in starts])
            false_safe = (forecast_length >= HEIGHT_115_GEOMETRIC_MIN) & (forecast_joint >= 0) & \
                ((observed_length < HEIGHT_115_GEOMETRIC_MIN) | (observed_joint < 0))
            for phase, mask in (("before_detectable", starts <= onset),
                                ("after_detectable", starts > onset)):
                assert mask.any(), (world, steps, phase)
                error = forecast_length[mask] - observed_length[mask]
                joint_error = forecast_joint[mask] - observed_joint[mask]
                rows.append(dict(world=world, terrain=terrain, event_id=event_id,
                                 first_new_contact_pre_step_s=float(pre[onset, 0]),
                                 first_detectable_pre_step_s=float(pre[onset + 1, 0]),
                                 first_new_contact_geom_pairs=sorted(sets[onset] - sets[onset - 1]),
                                 horizon_ms=steps * DT * 1000, phase=phase, starts=int(mask.sum()),
                                 false_safe=int(false_safe[mask].sum()),
                                 max_forecast_minus_actual_length_m=float(error.max()),
                                 max_optimistic_length_error_m=float(max(0., error.max())),
                                 max_forecast_minus_actual_eight_joint_margin_rad=float(joint_error.max()),
                                 first_false_safe_s=float(pre[starts[mask][np.flatnonzero(false_safe[mask])[0]], 0])
                                 if false_safe[mask].any() else None))
    prior_path = ROOT / "wheelleg_warp/results/height_115_predictability_20260928/verification.json"
    prior = json.loads(prior_path.read_text())
    for old in prior["rows"]:
        if (old["world"] not in (1, 2, 3) or old["model"] != "last_step_acceleration"
                or old["horizon_ms"] not in (5., 10.)):
            continue
        split = [row for row in rows if row["world"] == old["world"]
                 and row["horizon_ms"] == old["horizon_ms"]]
        assert len(split) == 2 and sum(row["starts"] for row in split) == old["starts"]
        assert sum(row["false_safe"] for row in split) == old["false_safe"]
        assert abs(max(row["max_forecast_minus_actual_length_m"] for row in split)
                   - old["max_optimistic_length_error_m"]) < 1e-12
    names = ("wheelleg_warp/probe_height_115_contact_predictability.py",
             "wheelleg_warp/probe_height_115_predictability.py",
             "wheelleg_warp/probe_height_115_margin.py", "wheelleg_warp/probe_height_115_passive.py",
             "wheelleg_warp/native/terrain.py", "wheelleg_warp/native/models.py",
             "wheelleg_ppo/tools/wheelleg_sim.py", "wheelleg_ppo/xml/wheelleg.xml")
    output.mkdir(parents=True)
    payload = dict(role="offline_first_contact_detectability_split_not_action_predictor",
                   training=False, final_holdout_opened=False, nominal_target_m=.115,
                   geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,
                   prediction_inputs="current active q/dq and prior active dq only",
                   contact_sets_used_for_offline_phase_label_only=True,
                   actual_A_and_eight_joints_used_for_offline_score_only=True,
                   contact_timing="contacts[k] solve pre[k]; first affected velocity in post[k]=pre[k+1]",
                   first_detectable_means="earliest ideal q/dq access after impact, not a tested contact classifier or noisy sensor",
                   source_state_verification_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   prior_prediction_verification_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                   source_windows_sha256=archive["windows_sha256"], rows=rows,
                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names})
    (output / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(row["terrain"], row["horizon_ms"], row["phase"], row["starts"],
              row["max_forecast_minus_actual_length_m"], row["false_safe"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
