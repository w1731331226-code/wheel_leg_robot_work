"""首次状态预警动作的10ms同图跟踪；5ms预测之外只作物理后验。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import numpy as np
from native.terrain import HeightTerrainScenario, model, HEIGHT_115_GEOMETRIC_MIN
from probe_height_115_action_predict_loow import compare_start, sha
from probe_height_115_continuous_common_1nm import GAMMA, LENGTH_SCALE_M, JOINT_SCALE_RAD
from probe_height_115_contact_action_pair import command, margins
from probe_height_115_live_common_local import common_basis, simulate
from probe_height_115_margin import lengths
from probe_height_115_passive import geometry
from probe_height_115_radial_authority import pose

DT = .0005
ACTIVE = ("alphaL", "betaL", "alphaR", "betaR")


def run(output):
    assert not output.exists(), output
    alarm_path = ROOT / "wheelleg_warp/results/height_115_alarm_timing_integer_20260929/verification.json"
    fitted_path = ROOT / "wheelleg_warp/results/height_115_action_predict_1nm_single_graph_20260929/verification.json"
    archive_path = ROOT / "wheelleg_warp/results/height_115_local_states_20260928/verification.json"
    alarm, fitted, archive = [json.loads(p.read_text()) for p in (alarm_path, fitted_path, archive_path)]
    assert alarm["fitted_verification_sha256"] == sha(fitted_path)
    assert alarm["archive_sha256"] == sha(archive_path)
    assert all(sha(ROOT / n) == h for n, h in alarm["source_sha256"].items())
    output.mkdir(parents=True)
    payload = dict(status="running", role="archived_state_alarm_fixed_action_live_nominal_10ms_followthrough",
        training=False, final_holdout_opened=False, online_trigger_or_reoptimization_tested=False,
        control_duration_s=.02, scoring_horizons_ms=[5, 10],
        note="first alarm selected on archived baseline; candidate rechecked on current state; no future safety in action",
        alarm_verification_sha256=sha(alarm_path), rows=[])
    raw = {}
    try:
        for item in alarm["rows"]:
            if not item["candidate"]["feasible"]:
                continue
            w = item["world"]
            event = archive["events"][fitted["selected_event_ids"][w]]
            scenario = HeightTerrainScenario(**event["scenario"])
            m = model(scenario)
            qadr = np.array([m.joint(n).qposadr[0] for n in ACTIVE])
            vadr = np.array([m.joint(n).dofadr[0] for n in ACTIVE])
            action = np.asarray(item["candidate"]["action"])
            coeff = np.zeros((9, 3)); coeff[1] = action
            records, state, _, _ = simulate([scenario], [round(item["first_alarm_s"] / DT)], coeff)
            for arm, record in enumerate(records):
                compare_start(state[arm], state[0], m.nq, m.nv, f"world{w} arm{arm}")
                assert np.all(record["pre"][:20, -1] == 1), (w, arm, "inactive")
            q = records[0]["pre"][0, qadr]
            v = records[0]["pre"][0, m.nq + vadr]
            previous = records[0]["previous_qvel"][vadr]
            fold = fitted["folds"][w]; gain = np.asarray(fold["gain"])
            reserve = fold["reserves"]
            t = np.arange(1, 11)[:, None] * DT
            pred = q + v * t + .5 * ((v - previous) / DT + gain @ action) * t * t
            pl = min(np.minimum(lengths(pred[:, 0], pred[:, 1]), lengths(pred[:, 2], pred[:, 3])))
            pj = float((1.5 - abs(pred)).min())
            normalized = min((pl - HEIGHT_115_GEOMETRIC_MIN - reserve["actual_A_length_m"]) / LENGTH_SCALE_M,
                             (pj - reserve["eight_joint_margin_rad"]) / JOINT_SCALE_RAD)
            assert normalized >= GAMMA - 1e-6, (w, "current prediction failed", normalized)
            wheels = {m.geom("wheel_collide_" + s).id for s in ("L", "R")}
            arms = []
            for arm in (0, 1):
                record = records[arm]
                applied_action = coeff[arm]
                clipping = mismatch = peak = 0.
                for k in range(20):
                    pre = record["pre"][k]
                    base = pre[m.nq + m.nv:m.nq + m.nv + 6]
                    request = base + common_basis(m, pre[:m.nq], pre[m.nq:m.nq + m.nv]) @ applied_action
                    expected, _, _ = command(m, pre[:m.nq], pre[m.nq:m.nq + m.nv], base, applied_action)
                    clipping = max(clipping, float(max(abs(request - expected))))
                    mismatch = max(mismatch, float(max(abs(record["applied"][k] - expected))))
                    peak = max(peak, float(max(abs(record["applied"][k] - base))))
                assert clipping <= 1e-6 and mismatch <= 1e-5, (w, arm, clipping, mismatch)
                actual = geometry(m, record["post"][:20])[1].min(axis=1)
                joint = margins(m, record["post"][:20]).min(axis=1)
                attitude = np.asarray([pose(p, scenario)["terrain_relative_abs_deg"] for p in record["post"][:20]])
                world_pose = np.asarray([pose(p, scenario)["world_abs_deg"] for p in record["post"][:20]])
                nonwheel = np.asarray([any(not wheels.intersection(pair) for pair in p) for p in record["contacts"][:20]])
                metrics = []
                for count in (10, 20):
                    safe = bool(actual[:count].min() >= HEIGHT_115_GEOMETRIC_MIN and joint[:count].min() >= 0 and
                        attitude[:count].max() <= 5 and world_pose[:count, :2].max() <= 10 and not nonwheel[:count].any())
                    metrics.append(dict(horizon_ms=count / 2, min_actual_A_m=float(actual[:count].min()),
                        min_eight_joint_margin_rad=float(joint[:count].min()), safe=safe,
                        max_relative_attitude_deg=float(attitude[:count].max()), nonwheel_steps=int(nonwheel[:count].sum())))
                first_bad = np.flatnonzero((actual < HEIGHT_115_GEOMETRIC_MIN) | (joint < 0))
                arms.append(dict(arm="zero" if arm == 0 else "candidate", horizons=metrics,
                    first_geometry_failure_after_alarm_ms=float((first_bad[0] + 1) / 2) if len(first_bad) else None,
                    clipping_Nm=clipping, command_match_error_Nm=mismatch, peak_motor_delta_Nm=peak))
                raw[f"world{w}_arm{arm}_post"] = record["post"][:20]
                raw[f"world{w}_arm{arm}_ctrl"] = record["applied"][:20]
            payload["rows"].append(dict(world=w, alarm_lead_ms=item["lead_ms"], action=action.tolist(),
                model_current_gamma=normalized, normalized_action_l1_support=item["candidate"]["normalized_action_l1_support"],
                contact_equal_steps=sum(a == b for a, b in zip(records[0]["contacts"][:20], records[1]["contacts"][:20])),
                arms=arms))
            print(w, [(a["arm"], [h["safe"] for h in a["horizons"]]) for a in arms], flush=True)
        payload["status"] = "completed"
    except Exception as error:
        payload["status"] = "failed_fixed_gate"
        payload["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        np.savez_compressed(output / "trace.npz", **raw)
        payload["trace_sha256"] = sha(output / "trace.npz")
        names = ("wheelleg_warp/probe_height_115_alarm_response.py", "wheelleg_warp/probe_height_115_alarm_timing.py",
            "wheelleg_warp/probe_height_115_continuous_common_1nm.py", "wheelleg_warp/probe_height_115_live_common_local.py",
            "wheelleg_warp/probe_height_115_contact_action_pair.py", "wheelleg_warp/probe_height_115_radial_authority.py",
            "wheelleg_warp/probe_height_115_passive.py", "wheelleg_warp/native/controller.py",
            "wheelleg_warp/native/environment.py", "wheelleg_warp/native/models.py", "wheelleg_warp/native/terrain.py",
            "wheelleg_ppo/tools/state_estimation.py", "wheelleg_ppo/tools/wheelleg_sim.py", "wheelleg_ppo/xml/wheelleg.xml")
        payload["source_sha256"] = {n: sha(ROOT / n) for n in names}
        (output / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
