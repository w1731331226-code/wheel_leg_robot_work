"""固定5ms留出预测器：原轨迹上首次因果报警、调度延迟与当时动作可行性。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import numpy as np
from native.terrain import HEIGHT_115_GEOMETRIC_MIN, HeightTerrainScenario, model
from probe_height_115_action_predict_loow import sha
from probe_height_115_continuous_common_1nm import solve_one, GAMMA, LENGTH_SCALE_M, JOINT_SCALE_RAD
from probe_height_115_live_common_local import common_basis
from probe_height_115_margin import lengths
from probe_height_115_passive import geometry

DT = .0005
STEPS = 10
ACTIVE = ("alphaL", "betaL", "alphaR", "betaR")


def run(output):
    assert not output.exists(), output
    archive_path = ROOT / "wheelleg_warp/results/height_115_local_states_20260928/verification.json"
    fitted_path = ROOT / "wheelleg_warp/results/height_115_action_predict_1nm_single_graph_20260929/verification.json"
    archive = json.loads(archive_path.read_text())
    fitted = json.loads(fitted_path.read_text())
    path = archive_path.parent / "windows.npz"
    assert sha(path) == archive["windows_sha256"] == fitted["source_windows_sha256"]
    assert sha(archive_path) == fitted["source_state_verification_sha256"]
    windows = np.load(path)
    cols = archive["columns"]
    rows = []
    for world, event_id in enumerate(fitted["selected_event_ids"]):
        event = archive["events"][event_id]
        assert event["world"] == world
        m = model(HeightTerrainScenario(**event["scenario"]))
        pre = windows[f"event_{event_id}_pre"]
        post = windows[f"event_{event_id}_post"]
        qfull = pre[:, cols["qpos_start"]:cols["qpos_start"] + m.nq]
        vfull = pre[:, cols["qvel_start"]:cols["qvel_start"] + m.nv]
        qa = np.array([m.joint(n).qposadr[0] for n in ACTIVE])
        va = np.array([m.joint(n).dofadr[0] for n in ACTIVE])
        q, v = qfull[:, qa], vfull[:, va]
        fold = fitted["folds"][world]
        assert fold["heldout_world"] == world
        reserve = (fold["reserves"]["actual_A_length_m"], fold["reserves"]["eight_joint_margin_rad"])
        gain = np.asarray(fold["gain"])
        # 真值失败时刻只评分；报警逐帧仅使用当前q/dq及前一帧dq。
        future = post[:, cols["qpos_start"]:cols["qpos_start"] + m.nq]
        actual_length = geometry(m, future)[1].min(axis=1)
        joints = [m.joint(n) for n in ("alphaL", "betaL", "passA_L", "passC_L",
                                      "alphaR", "betaR", "passA_R", "passC_R")]
        ranges = np.asarray([m.jnt_range[j.id] for j in joints])
        angles = future[:, [j.qposadr[0] for j in joints]]
        actual_margin = np.minimum(angles - ranges[:, 0], ranges[:, 1] - angles).min(axis=1)
        bad = np.flatnonzero((actual_length < HEIGHT_115_GEOMETRIC_MIN) | (actual_margin < 0))
        assert len(bad), world
        failure_step = int(round(post[bad[0], 0] / DT))
        physical_steps = np.rint(pre[:, 0] / DT).astype(int)
        failure_s = failure_step * DT
        assert failure_step == round(event["reference_s"] / DT)
        assert np.all(np.diff(physical_steps) == 1)
        eligible = np.flatnonzero((physical_steps < failure_step) & (np.arange(len(pre)) > 0))
        t = (np.arange(1, STEPS + 1) * DT)[None, :, None]
        pred = q[eligible, None] + v[eligible, None] * t + \
            .5 * ((v[eligible] - v[eligible - 1]) / DT)[:, None] * t * t
        leg = np.minimum(lengths(pred[..., 0], pred[..., 1]), lengths(pred[..., 2], pred[..., 3])).min(axis=1)
        joint = (1.5 - abs(pred)).min(axis=(1, 2))
        alarm = (leg < HEIGHT_115_GEOMETRIC_MIN + reserve[0] + GAMMA * LENGTH_SCALE_M) | \
                (joint < reserve[1] + GAMMA * JOINT_SCALE_RAD)
        hits = eligible[alarm]
        assert len(hits), f"world{world}: no pre-failure warning"
        first = int(hits[0])
        basis = common_basis(m, qfull[first], vfull[first])
        eps = 1. / np.max(abs(basis), axis=0)
        candidate = solve_one(q[first], v[first], v[first - 1], gain, reserve, eps, GAMMA)
        schedules = []
        for stride in (1, 2, 4, 10, 40):
            leads = []
            for phase in range(stride):
                sampled = hits[physical_steps[hits] % stride == phase]
                leads.append(float((failure_step - physical_steps[sampled[0]]) * DT * 1000) if len(sampled) else None)
            schedules.append(dict(period_ms=stride * DT * 1000,
                phases=stride, missed_phases=sum(x is None for x in leads),
                minimum_detected_lead_ms=min((x for x in leads if x is not None), default=None),
                maximum_detected_lead_ms=max((x for x in leads if x is not None), default=None)))
        contacts = windows[f"event_{event_id}_contacts"]
        sets = [{tuple(sorted((int(c[1]), int(c[2])))) for c in step if int(c[0]) == world}
                for step in contacts]
        changes = [k for k in range(1, len(pre)) if sets[k] - sets[k - 1] and
                   failure_s - .02 < pre[k, 0] < failure_s]
        onset_step = int(physical_steps[changes[0]]) if changes else None
        onset = onset_step * DT if onset_step is not None else None
        rows.append(dict(world=world, terrain=event["scenario"]["terrain"], speed=event["scenario"]["speed"],
            event=event["kind"], failure_s=failure_s, first_alarm_s=float(physical_steps[first] * DT),
            lead_ms=float((failure_step - physical_steps[first]) * DT * 1000),
            scan_start_s=float(pre[eligible[0], 0]), first_alarm_left_censored=bool(first == eligible[0]),
            first_new_contact_s=onset,
            alarm_after_first_affected_dq=bool(physical_steps[first] >= onset_step + 1) if onset_step is not None else None,
            alarm_leg=bool(leg[np.flatnonzero(alarm)[0]] < HEIGHT_115_GEOMETRIC_MIN + reserve[0] + GAMMA * LENGTH_SCALE_M),
            alarm_joint=bool(joint[np.flatnonzero(alarm)[0]] < reserve[1] + GAMMA * JOINT_SCALE_RAD),
            candidate=candidate, schedule_phase_audit=schedules))
    sources = ("wheelleg_warp/probe_height_115_alarm_timing.py", "wheelleg_warp/probe_height_115_continuous_common_1nm.py",
               "wheelleg_warp/probe_height_115_live_common_local.py", "wheelleg_warp/probe_height_115_margin.py",
               "wheelleg_warp/probe_height_115_passive.py", "wheelleg_warp/native/terrain.py",
               "wheelleg_warp/native/models.py", "wheelleg_ppo/tools/state_estimation.py",
               "wheelleg_ppo/tools/wheelleg_sim.py", "wheelleg_ppo/xml/wheelleg.xml")
    output.mkdir(parents=True)
    summary = dict(worlds=6, pre_failure_alarms=len(rows),
        left_censored=sum(r["first_alarm_left_censored"] for r in rows),
        model_lp_feasible_at_first_alarm=sum(r["candidate"]["feasible"] for r in rows),
        nonlinear_forecast_gamma_pass=sum(r["candidate"].get("nonlinear_forecast_gamma_pass", False) for r in rows),
        minimum_lead_ms=min(r["lead_ms"] for r in rows))
    payload = dict(role="archived_baseline_fixed_forecast_first_alarm_timing_not_online_control",
        training=False, final_holdout_opened=False, actual_action_executed=False,
        alarm_input="current active q/dq and prior active dq only; frozen other-world reserve",
        contact_and_actual_geometry="posterior timing labels and failure scoring only",
        timing="integer 2kHz physical steps; same-tick float timestamp differences are not reaction time",
        limitation="selected failure neighborhoods only; no whole-episode false-alarm or modified-trajectory claim",
        summary=summary, rows=rows, archive_sha256=sha(archive_path), windows_sha256=sha(path),
        fitted_verification_sha256=sha(fitted_path), source_sha256={n: sha(ROOT / n) for n in sources})
    (output / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(summary)
    for r in rows:
        print(r["world"], r["lead_ms"], r["candidate"]["feasible"], r["candidate"].get("minimum_peak_motor_delta_Nm"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
