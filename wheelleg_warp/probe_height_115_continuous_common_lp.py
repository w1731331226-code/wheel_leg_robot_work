"""六个0.115 m晚窗：冻结留出预测器的连续三维小动作可行性。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import numpy as np
from scipy.optimize import linprog
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
from probe_height_115_margin import lengths
from state_estimation import leg_kinematics

DT = .0005
STEPS = 10
TRUST_NM = .08
LENGTH_SCALE_M = .001
JOINT_SCALE_RAD = .1


def row_for_world(world, q, v, previous, gain, reserve):
    b = np.zeros((6, 3))
    for side in range(2):
        a = 2 * side
        b[a:a + 2, :2] = leg_kinematics(q[a:a + 2], v[a:a + 2])[3]
    b[4:, 2] = 1.
    constraints, bounds = [], []
    for step in range(1, STEPS + 1):
        t = step * DT
        base = q + v * t + .5 * (v - previous) / DT * t * t
        response = .5 * t * t * gain
        for side in range(2):
            a = 2 * side
            length = lengths(base[a], base[a + 1])
            jac = leg_kinematics(base[a:a + 2], v[a:a + 2])[3][:, 0]
            sensitivity = jac @ response[a:a + 2]
            constraints.append(-np.r_[sensitivity / LENGTH_SCALE_M, -1.])
            bounds.append((length - HEIGHT_115_GEOMETRIC_MIN - reserve[0]) / LENGTH_SCALE_M)
        for joint in range(4):
            # Both ±1.5 rad active-joint sides, each tightened by the heldout reserve.
            constraints.extend((np.r_[response[joint] / JOINT_SCALE_RAD, 1.],
                                np.r_[-response[joint] / JOINT_SCALE_RAD, 1.]))
            bounds.extend(((1.5 - reserve[1] - base[joint]) / JOINT_SCALE_RAD,
                           (1.5 - reserve[1] + base[joint]) / JOINT_SCALE_RAD))
    # First three entries are the action, last entry is the minimum normalized margin.
    for motor in b:
        constraints.extend((np.r_[motor, 0.], np.r_[-motor, 0.]))
        bounds.extend((TRUST_NM, TRUST_NM))
    result = linprog([0., 0., 0., -1.], A_ub=np.asarray(constraints), b_ub=np.asarray(bounds),
                     bounds=[(None, None)] * 4, method="highs")
    assert result.success, (world, result.message)
    action = result.x[:3]
    predicted_length = []
    predicted_joint = []
    for step in range(1, STEPS + 1):
        t = step * DT
        state = q + v * t + .5 * ((v - previous) / DT + gain @ action) * t * t
        predicted_length.append(min(lengths(state[0], state[1]), lengths(state[2], state[3])))
        predicted_joint.append(1.5 - max(abs(state)))
    length_margin = min(predicted_length) - HEIGHT_115_GEOMETRIC_MIN - reserve[0]
    joint_margin = min(predicted_joint) - reserve[1]
    return dict(world=world, linearized_max_min_normalized_margin=float(result.x[3]),
                linearized_feasible=bool(result.x[3] >= -1e-9),
                common_virtual_action=action.tolist(),
                peak_start_motor_delta_Nm=float(np.max(abs(b @ action))),
                nonlinear_forecast_length_margin_after_reserve_m=float(length_margin),
                nonlinear_forecast_joint_margin_after_reserve_rad=float(joint_margin),
                nonlinear_forecast_both_safe=bool(length_margin >= 0 and joint_margin >= 0),
                training_reserve_length_m=float(reserve[0]), training_reserve_joint_rad=float(reserve[1]))


def run(output):
    assert not output.exists(), output
    folder = ROOT / "wheelleg_warp/results/height_115_action_predict_single_graph_20260928"
    source = folder / "verification.json"
    verification = json.loads(source.read_text())
    inputs = folder / "heldout_inputs.npz"
    assert verification["status"] == "completed" and verification["summary"]["total"] == 108
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == verification["heldout_inputs_sha256"]
    data = np.load(inputs)
    rows = []
    for world, fold in enumerate(verification["folds"]):
        assert fold["heldout_world"] == world
        index = 2 * world + 1  # first-failure minus 5 ms; the earlier window is kept in source evidence.
        reserve = (fold["reserves"]["actual_A_length_m"], fold["reserves"]["eight_joint_margin_rad"])
        rows.append(row_for_world(world, data["q0"][index], data["v0"][index],
                                  data["vprev"][index], np.asarray(fold["gain"]), reserve))
    summary = dict(worlds=len(rows), linearized_feasible=sum(row["linearized_feasible"] for row in rows),
                   nonlinear_forecast_both_safe=sum(row["nonlinear_forecast_both_safe"] for row in rows))
    names = ("wheelleg_warp/probe_height_115_continuous_common_lp.py",
             "wheelleg_warp/probe_height_115_margin.py", "wheelleg_ppo/tools/state_estimation.py",
             "wheelleg_ppo/tools/wheelleg_sim.py", "wheelleg_ppo/xml/wheelleg.xml")
    output.mkdir(parents=True)
    (output / "verification.json").write_text(json.dumps(dict(
        role="offline_heldout_fixed_model_continuous_common3_small_trust_lp",
        training=False, final_holdout_opened=False, target_height_m=.115,
        horizon_ms=5., action_set="constant common3, start Jacobian |B c|<=0.08 Nm, no absolute motor box",
        predictor="fixed heldout-fold G and inner-world reserves; current active q/dq and same-graph prior dq",
        linearization="leg FK only about each baseline predicted step; active joint constraints exact",
        no_future_actual_A_or_passive_or_contact_in_optimizer=True,
        no_warp_action_candidate_executed=True, summary=summary, rows=rows,
        source_loow_verification_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_heldout_inputs_sha256=verification["heldout_inputs_sha256"],
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}),
        ensure_ascii=False, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
