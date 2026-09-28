"""冻结六世界留出模型：1 Nm 共模最小峰值 LP 与独立 5 ms Warp 复核。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import numpy as np
from scipy.optimize import linprog
from native.terrain import HEIGHT_115_GEOMETRIC_MIN, HeightTerrainScenario, model
from probe_height_115_action_predict_loow import START_LIMITS, compare_start, require, sha
from probe_height_115_contact_action_pair import command, margins
from probe_height_115_live_common_local import common_basis, simulate
from probe_height_115_margin import cases, lengths
from probe_height_115_passive import geometry
from state_estimation import leg_kinematics

SOURCE = ROOT / "wheelleg_warp/results/height_115_action_predict_1nm_single_graph_20260929"
OUTPUT = ROOT / "wheelleg_warp/results/height_115_continuous_common_1nm_20260929"
DT, STEPS, PEAK_NM = .0005, 10, 1.
LENGTH_SCALE_M, JOINT_SCALE_RAD, GAMMA = .001, .1, .001
ACTIVE = ("alphaL", "betaL", "alphaR", "betaR")
VALIDATE_WORLDS = (0, 1)  # 固定数值裕量后唯一有解的公开晚窗。


def load_inputs():
    source = SOURCE / "verification.json"
    record = json.loads(source.read_text())
    require(record["status"] == "completed" and record["summary"]["total"] == 108 and
            record["late_zero_actually_failed_worlds"] == 6 and
            record["summary"]["combined_false_safe"] == 0,
            "1 Nm 留出源未完成或六个晚窗定义改变")
    for name, digest in record["source_sha256"].items():
        require(sha(ROOT / name) == digest, f"1 Nm 源码已变：{name}")
    trace = SOURCE / "heldout_inputs.npz"
    require(sha(trace) == record["heldout_inputs_sha256"], "1 Nm 输入 NPZ 哈希不符")
    data = np.load(trace)
    for key, shape in (("q0", (12, 4)), ("v0", (12, 4)), ("vprev", (12, 4)),
                       ("actions", (12, 9, 3)), ("actual_A_length_m", (12, 9, 10, 2)),
                       ("actual_eight_joint_margin_rad", (12, 9, 10, 8))):
        require(data[key].shape == shape and np.isfinite(data[key]).all(), f"无效源数据：{key}")
    archive_path = ROOT / "wheelleg_warp/results/height_115_local_states_20260928/verification.json"
    archive = json.loads(archive_path.read_text())
    require(sha(archive_path) == record["source_state_verification_sha256"] and
            sha(archive_path.parent / "windows.npz") == record["source_windows_sha256"],
            "场景档案与 1 Nm 标定源不一致")
    require([archive["events"][i]["scenario"] for i in record["selected_event_ids"]] ==
            [vars(s) for s in cases()[:6]], "六个正常场景或顺序改变")
    return source, trace, record, data, archive


def solve_one(q, v, vprev, gain, reserve, eps, gamma):
    """c=(F,H,W)；第四变量是起点六电机增量的峰值 p。"""
    basis = np.zeros((6, 3))
    for side in range(2):
        a = 2 * side
        basis[a:a + 2, :2] = leg_kinematics(q[a:a + 2], v[a:a + 2])[3]
    basis[4:, 2] = 1.
    constraints, bounds = [], []
    for step in range(1, STEPS + 1):
        t = step * DT
        base = q + v * t + .5 * (v - vprev) / DT * t * t
        response = .5 * t * t * gain
        for side in range(2):
            a = 2 * side
            jac = leg_kinematics(base[a:a + 2], v[a:a + 2])[3][:, 0]
            sensitivity = jac @ response[a:a + 2]
            constraints.append(np.r_[-sensitivity / LENGTH_SCALE_M, 0.])
            bounds.append((lengths(base[a], base[a + 1]) - HEIGHT_115_GEOMETRIC_MIN - reserve[0]) /
                          LENGTH_SCALE_M - gamma)
        for joint in range(4):
            constraints.extend((np.r_[response[joint] / JOINT_SCALE_RAD, 0.],
                                np.r_[-response[joint] / JOINT_SCALE_RAD, 0.]))
            bounds.extend(((1.5 - reserve[1] - base[joint]) / JOINT_SCALE_RAD - gamma,
                           (1.5 - reserve[1] + base[joint]) / JOINT_SCALE_RAD - gamma))
    for motor in basis:
        constraints.extend((np.r_[motor, -1.], np.r_[-motor, -1.]))
        bounds.extend((0., 0.))
    result = linprog([0., 0., 0., 1.], A_ub=np.asarray(constraints), b_ub=np.asarray(bounds),
                     bounds=[(None, None)] * 3 + [(0., PEAK_NM)], method="highs")
    row = dict(feasible=bool(result.success), solver_status=int(result.status),
               solver_message=result.message, numerical_gamma=gamma)
    if not result.success:
        return row
    action = result.x[:3]
    predicted = [q + v * (step * DT) +
                 .5 * ((v - vprev) / DT + gain @ action) * (step * DT) ** 2
                 for step in range(1, STEPS + 1)]
    leg_margin = min(min(lengths(x[0], x[1]), lengths(x[2], x[3])) for x in predicted) - \
        HEIGHT_115_GEOMETRIC_MIN - reserve[0]
    joint_margin = min(1.5 - max(abs(x)) for x in predicted) - reserve[1]
    normalized = min(leg_margin / LENGTH_SCALE_M, joint_margin / JOINT_SCALE_RAD)
    row.update(action=action.tolist(), minimum_peak_motor_delta_Nm=float(result.x[3]),
               reconstructed_peak_motor_delta_Nm=float(np.max(abs(basis @ action))),
               nonlinear_forecast_leg_margin_after_reserve_m=float(leg_margin),
               nonlinear_forecast_joint_margin_after_reserve_rad=float(joint_margin),
               nonlinear_forecast_min_normalized_margin=float(normalized),
               nonlinear_forecast_gamma_pass=bool(normalized >= gamma - 1e-8),
               normalized_action_l1_support=float(np.sum(abs(action / eps))),
               outside_calibration_convex_hull=bool(np.sum(abs(action / eps)) > 1. + 1e-8))
    return row


def solve_all(record, data):
    rows = []
    for world, fold in enumerate(record["folds"]):
        require(fold["heldout_world"] == world, "留出折顺序改变")
        index = 2 * world + 1  # 首次失败前 5 ms。
        q, v, vprev = (data[key][index] for key in ("q0", "v0", "vprev"))
        gain = np.asarray(fold["gain"])
        reserve = (fold["reserves"]["actual_A_length_m"],
                   fold["reserves"]["eight_joint_margin_rad"])
        eps = np.array([data["actions"][index, 1 + 2 * k, k] for k in range(3)])
        require(gain.shape == (4, 3) and np.isfinite(gain).all() and np.all(eps > 0),
                f"world{world} G 或动作尺度无效")
        rows.append(dict(world=world, start_s=next(r["start_s"] for r in record["heldout_rows"]
                    if r["world"] == world and r["lag_ms"] == 5. and r["action"] == "zero"),
                         reserve_actual_A_m=float(reserve[0]), reserve_eight_joint_rad=float(reserve[1]),
                         zero_extra_margin=solve_one(q, v, vprev, gain, reserve, eps, 0.),
                         numerical_margin=solve_one(q, v, vprev, gain, reserve, eps, GAMMA)))
    require([r["world"] for r in rows if r["numerical_margin"]["feasible"]] == list(VALIDATE_WORLDS),
            "固定 0.001 数值裕量下的可行世界改变，停止 Warp")
    for world in VALIDATE_WORLDS:
        require(rows[world]["numerical_margin"]["nonlinear_forecast_gamma_pass"],
                f"world{world} 非线性 FK 复算未过固定数值裕量")
    return rows


def validate_world(world, row, record, data, archive):
    event = archive["events"][record["selected_event_ids"][world]]
    scenario = HeightTerrainScenario(**event["scenario"])
    require(scenario.terrain in ("legacy", "step") and scenario.grade_deg == 0 and
            scenario.relative_attitude,
            "姿态参考不再为零，停止本次局部复核")
    m = model(scenario)
    index = 2 * world + 1
    start = round(row["start_s"] / DT)
    require(abs(start * DT - row["start_s"]) < 1e-9, "起点不是 2 kHz 物理步")
    coefficients = np.zeros((9, 3))
    coefficients[1] = row["numerical_margin"]["action"]
    records, states, _, _ = simulate([scenario], [start], coefficients)
    qa = np.array([m.joint(name).qposadr[0] for name in ACTIVE])
    va = np.array([m.joint(name).dofadr[0] for name in ACTIVE])
    for arm in range(9):
        compare_start(states[arm], states[0], m.nq, m.nv, f"world{world} arm{arm}")
        require(np.all(records[arm]["pre"][:STEPS, -1] == 1),
                f"world{world} arm{arm} 首 5 ms 非活动")
    require(np.max(abs(states[0, 1 + qa] - data["q0"][index])) <= START_LIMITS["qpos"] and
            np.max(abs(states[0, 1 + m.nq + va] - data["v0"][index])) <= START_LIMITS["qvel"] and
            np.max(abs(records[0]["previous_qvel"][va] - data["vprev"][index])) <= START_LIMITS["qvel"],
            f"world{world} 当前主动 q/dq/前一 dq 与冻结输入不同")
    zero, candidate = records[:2]
    zero_actual = geometry(m, zero["post"][:STEPS])[1]
    zero_joints = margins(m, zero["post"][:STEPS])
    require(np.max(abs(zero_actual - data["actual_A_length_m"][index, 0])) <= 1e-6 and
            np.max(abs(zero_joints - data["actual_eight_joint_margin_rad"][index, 0])) <= 1e-4,
            f"world{world} 零臂结果未重现 1 Nm 标定图")
    zero_clone_error = 0.
    zero_clone_velocity_error = 0.
    zero_clone_command_error = 0.
    zero_clone_contacts = 0
    for arm in range(2, 9):
        other = records[arm]
        zero_clone_error = max(zero_clone_error,
                               float(np.max(abs(other["post"][:STEPS] - zero["post"][:STEPS]))))
        zero_clone_velocity_error = max(zero_clone_velocity_error,
            float(np.max(abs(other["pre"][:STEPS, m.nq:m.nq + m.nv] -
                             zero["pre"][:STEPS, m.nq:m.nq + m.nv]))))
        zero_clone_command_error = max(zero_clone_command_error,
            float(np.max(abs(other["applied"][:STEPS] - zero["applied"][:STEPS]))))
        zero_clone_contacts += sum(a == b for a, b in zip(zero["contacts"][:STEPS],
                                                           other["contacts"][:STEPS]))
    require(zero_clone_error <= 2e-6 and zero_clone_velocity_error <= 1e-6 and
            zero_clone_command_error <= 1e-6 and
            zero_clone_contacts == 7 * STEPS,
            f"world{world} 九臂零动作副本不一致")
    action = np.asarray(row["numerical_margin"]["action"])
    motor_match = motor_clip = peak_requested_delta = peak_applied_delta = 0.
    for step in range(STEPS):
        pre = candidate["pre"][step]
        q, v = pre[:m.nq], pre[m.nq:m.nq + m.nv]
        baseline = pre[m.nq + m.nv:m.nq + m.nv + 6]
        request = baseline + common_basis(m, q, v) @ action
        expected, _, _ = command(m, q, v, baseline, action)
        motor_match = max(motor_match, float(np.max(abs(candidate["applied"][step] - expected))))
        motor_clip = max(motor_clip, float(np.max(abs(request - expected))))
        peak_requested_delta = max(peak_requested_delta, float(np.max(abs(request - baseline))))
        peak_applied_delta = max(peak_applied_delta, float(np.max(abs(candidate["applied"][step] - baseline))))
    require(motor_match <= 1e-5 and motor_clip <= 1e-6 and
            abs(float(np.max(abs(common_basis(m, candidate["pre"][0, :m.nq],
                                               candidate["pre"][0, m.nq:m.nq + m.nv]) @ action))) -
                row["numerical_margin"]["reconstructed_peak_motor_delta_Nm"]) <= 1e-5,
            f"world{world} 电机命令、动态限幅或起点动作映射不一致")
    actual = geometry(m, candidate["post"][:STEPS])[1]
    joints = margins(m, candidate["post"][:STEPS])
    require(np.max(abs(actual.min(axis=1) - candidate["summary"][:STEPS, 1])) <= 1e-5 and
            np.max(abs(joints.min(axis=1) - candidate["summary"][:STEPS, 2])) <= 1e-4,
            f"world{world} Warp 采样与独立几何/关节复算不符")
    contact_equal = sum(a == b for a, b in zip(zero["contacts"][:STEPS],
                                                candidate["contacts"][:STEPS]))
    wheels = {m.geom(name).id for name in ("wheel_collide_L", "wheel_collide_R")}
    nonwheel = lambda pairs: {pair for pair in pairs if not wheels.intersection(pair)}
    zero_nonwheel_steps = sum(bool(nonwheel(pairs)) for pairs in zero["contacts"][:STEPS])
    candidate_nonwheel_steps = sum(bool(nonwheel(pairs)) for pairs in candidate["contacts"][:STEPS])
    new_nonwheel_steps = sum(bool(nonwheel(now) - nonwheel(before))
                             for before, now in zip(zero["contacts"][:STEPS],
                                                    candidate["contacts"][:STEPS]))
    attitude = np.max(candidate["summary"][:STEPS, 3:6], axis=0)
    leg_margin = float(actual.min() - HEIGHT_115_GEOMETRIC_MIN)
    joint_margin = float(joints.min())
    raw_safe = bool(leg_margin >= 0 and joint_margin >= 0 and
                    np.max(attitude) <= np.deg2rad(5) and candidate_nonwheel_steps == 0)
    gamma_safe = bool(leg_margin >= GAMMA * LENGTH_SCALE_M and
                      joint_margin >= GAMMA * JOINT_SCALE_RAD and
                      np.max(attitude) <= np.deg2rad(5) and candidate_nonwheel_steps == 0)
    return dict(world=world, physical_steps=STEPS, same_graph_zero_clone_max_qpos_error=zero_clone_error,
                same_graph_zero_clone_max_qvel_error=zero_clone_velocity_error,
                same_graph_zero_clone_max_command_error_Nm=zero_clone_command_error,
                same_graph_zero_clone_contact_pairs_equal=zero_clone_contacts,
                zero_min_actual_A_leg_m=float(zero_actual.min()),
                zero_min_eight_joint_margin_rad=float(zero_joints.min()),
                candidate_min_actual_A_leg_m=float(actual.min()),
                candidate_min_actual_A_margin_m=leg_margin,
                candidate_min_eight_joint_margin_rad=joint_margin,
                candidate_peak_abs_relative_roll_pitch_yaw_rad=attitude.tolist(),
                relative_attitude_reference_zero=True, scenario_grade_deg=scenario.grade_deg,
                candidate_motor_request_vs_kernel_max_error_Nm=motor_match,
                candidate_dynamic_motor_clip_max_Nm=motor_clip,
                candidate_peak_requested_motor_delta_Nm=peak_requested_delta,
                candidate_peak_applied_motor_delta_Nm=peak_applied_delta,
                contact_pair_steps_equal=contact_equal,
                contact_mode_changed=bool(contact_equal != STEPS),
                zero_nonwheel_contact_steps=zero_nonwheel_steps,
                candidate_nonwheel_contact_steps=candidate_nonwheel_steps,
                new_nonwheel_contact_steps=new_nonwheel_steps,
                actual_5ms_joint_A_attitude_nonwheel_safe=raw_safe,
                actual_5ms_numerical_gamma_pass=gamma_safe,
                same_contact_and_5ms_safe=bool(raw_safe and contact_equal == STEPS),
                trace=dict(zero=zero, candidate=candidate))


def save_result(output, payload):
    (output / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def run(output):
    require(not output.exists(), f"输出目录已存在：{output}")
    source, trace, record, data, archive = load_inputs()
    rows = solve_all(record, data)
    output.mkdir(parents=True)
    sources = ("wheelleg_warp/probe_height_115_continuous_common_1nm.py",
               "wheelleg_warp/probe_height_115_action_predict_loow.py",
               "wheelleg_warp/probe_height_115_live_common_local.py",
               "wheelleg_warp/probe_height_115_contact_action_pair.py",
               "wheelleg_warp/probe_height_115_passive.py",
               "wheelleg_warp/probe_height_115_margin.py", "wheelleg_warp/native/controller.py",
               "wheelleg_warp/probe_height_115_local_lp.py",
               "wheelleg_warp/probe_height_115_warp_local_lp.py",
               "wheelleg_warp/trace_first_divergence_v2.py",
               "wheelleg_warp/native/environment.py", "wheelleg_warp/native/terrain.py",
               "wheelleg_warp/native/models.py", "wheelleg_ppo/tools/state_estimation.py",
               "wheelleg_ppo/tools/wheelleg_sim.py", "wheelleg_ppo/tools/hardware_profile.py",
               "wheelleg_ppo/xml/wheelleg.xml")
    payload = dict(status="lp_ready", role="public_115m_frozen_1Nm_common3_minpeak_5ms_challenge",
                   training=False, final_holdout_opened=False, default_controller_changed=False,
                   nominal_height_m=.115, simulated_geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,
                   horizon_steps=STEPS, numerical_gamma=GAMMA,
                   gamma_meaning="extra 1 um leg or 0.0001 rad active joint; numerical only, not hardware safety",
                   action_set="constant common3; current-start |B(q0)c|<=1 Nm; live B(q) and nominal controller",
                   predictor="frozen outer-world G and inner-world single-sided reserve; current active q/dq/prior dq",
                   optimizer_uses_heldout_future_actual_A_passive_or_contact=False,
                   windows_selected_offline_from_first_failure_times=True,
                   calibration_limitation="continuous mixes outside calibrated action convex hull are out of calibration",
                   zero_margin_feasible_worlds=[r["world"] for r in rows if r["zero_extra_margin"]["feasible"]],
                   gamma_feasible_worlds=list(VALIDATE_WORLDS), rows=rows, warp_rows=[],
                   source_verification_sha256=sha(source), source_heldout_inputs_sha256=sha(trace),
                   source_sha256={name: sha(ROOT / name) for name in sources})
    save_result(output, payload)
    try:
        tested = []
        for world in VALIDATE_WORLDS:
            result = validate_world(world, rows[world], record, data, archive)
            raw = result.pop("trace")
            tested.append(raw)
            payload["warp_rows"].append(result)
            save_result(output, payload)
        np.savez_compressed(output / "trace.npz",
            worlds=np.asarray(VALIDATE_WORLDS),
            zero_pre=np.stack([r["zero"]["pre"][:STEPS] for r in tested]),
            candidate_pre=np.stack([r["candidate"]["pre"][:STEPS] for r in tested]),
            zero_applied=np.stack([r["zero"]["applied"][:STEPS] for r in tested]),
            candidate_applied=np.stack([r["candidate"]["applied"][:STEPS] for r in tested]),
            zero_post=np.stack([r["zero"]["post"][:STEPS] for r in tested]),
            candidate_post=np.stack([r["candidate"]["post"][:STEPS] for r in tested]),
            zero_contacts=np.stack([r["zero"]["contact_raw"][:STEPS] for r in tested]),
            candidate_contacts=np.stack([r["candidate"]["contact_raw"][:STEPS] for r in tested]))
        payload["trace_sha256"] = sha(output / "trace.npz")
        payload["status"] = "completed"
        payload["summary"] = dict(warp_candidates=len(tested),
            actual_5ms_joint_A_attitude_nonwheel_safe=sum(
                r["actual_5ms_joint_A_attitude_nonwheel_safe"] for r in payload["warp_rows"]),
            actual_5ms_numerical_gamma_pass=sum(
                r["actual_5ms_numerical_gamma_pass"] for r in payload["warp_rows"]),
            same_contact_and_5ms_safe=sum(r["same_contact_and_5ms_safe"] for r in payload["warp_rows"]),
            outside_calibration_convex_hull=sum(rows[w]["numerical_margin"]["outside_calibration_convex_hull"]
                                                for w in VALIDATE_WORLDS),
            robust_continuous_action_prediction_proven=False,
            six_world_5ms_rescue_proven=False, online_controller_proven=False,
            full_episode_or_hardware_safety_proven=False)
        save_result(output, payload)
    except Exception as exc:
        payload["status"] = "failed_pre_registered_gate_or_execution"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        save_result(output, payload)
        raise
    print(payload["summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="只核来源和 LP，不运行 Warp 或写结果")
    args = parser.parse_args()
    if args.check:
        _, _, record, data, _ = load_inputs()
        rows = solve_all(record, data)
        print(dict(zero_margin=[r["world"] for r in rows if r["zero_extra_margin"]["feasible"]],
                   gamma=[r["world"] for r in rows if r["numerical_margin"]["feasible"]]))
    else:
        run(args.output)
