"""height-115独立工程面板：先平地，再按结果决定常规与单侧上边界。"""
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
from native.environment import NativeEnv
from native.terrain import HEIGHT_115_MIN, HEIGHT_115_GEOMETRIC_MIN, HeightTerrainScenario, bank_height_115, sample_height_terrain_115

HEIGHTS = (.115, .12, .14, .16, .20, .25, .30, .35, .38)


def cases(panel):
    if panel == "normal115":
        from probe_height_115_margin import cases as low_cases
        return low_cases()[:6]
    if panel == "flat":
        return [HeightTerrainScenario(speed=.5, stand_height_m=h) for h in HEIGHTS]
    if panel == "regular":
        terrains = (("step", {"step_height_m": .02}), ("cross_slope", {"grade_deg": 3.}),
                    ("rough", {"roughness_m": .004}))
        return [HeightTerrainScenario(speed=.5, stand_height_m=h, terrain=name,
                                      terrain_seed=1151234, relative_attitude=True, **extra)
                for name, extra in terrains for h in HEIGHTS]
    return [HeightTerrainScenario(speed=speed, stand_height_m=h,
                                  **({"height_l": .02} if side == "L" else {"height_r": .02}))
            for side in ("L", "R") for speed in (.5, 1., -.5, -1.) for h in HEIGHTS]


def initial_geometry(env):
    model = env.cpu
    data = mujoco.MjData(model)
    q0 = env.q0.numpy()
    names = ("alphaL", "betaL", "alphaR", "betaR", "passA_L", "passC_L", "passA_R", "passC_R")
    joints = [model.joint(name) for name in names]
    margins, errors = [], []
    for row in q0:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        margins.append(min(float(min(row[joint.qposadr[0]] - model.jnt_range[joint.id, 0],
                                     model.jnt_range[joint.id, 1] - row[joint.qposadr[0]]))
                           for joint in joints))
        errors.append(max(float(np.linalg.norm(data.site_xpos[model.site(end).id] -
                                               data.site_xpos[model.site(axle).id]))
                          for end, axle in (("couplerB_L_end", "wheel_axle_L"),
                                            ("couplerB_R_end", "wheel_axle_R"))))
    assert min(margins) > 0 and max(errors) < 1e-6, (min(margins), max(errors))
    return dict(min_joint_margin_rad=min(margins), max_loop_error_m=max(errors))


def run(panel, output):
    assert panel in ("flat", "regular", "boundary", "normal115") and not output.exists()
    assert HEIGHTS[0] == HEIGHT_115_MIN
    assert [sample_height_terrain_115(1150000 + i).stand_height_m for i in range(3)] == [.115, .38, .3]
    for invalid in (.114, .381, float("nan")):
        try:
            HeightTerrainScenario(stand_height_m=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法目标腿高被接受：{invalid}")
    try:
        NativeEnv(n=1, scenario=HeightTerrainScenario(stand_height_m=.115),
                  bank_factory=bank_height_115, height_conditioned=True)
    except ValueError as error:
        assert "目标腿长超出0.160" in str(error), error
    else:
        raise AssertionError("旧高度设计意外接受0.115 m")
    scenarios = cases(panel)
    env = NativeEnv(n=len(scenarios), scenario=scenarios, bank_factory=bank_height_115,
                    height_conditioned=True, height_design="range115", residual_scale=0)
    obs = env.reset()
    assert np.allclose(obs[:, 11], [s.stand_height_m - .3 for s in scenarios], atol=1e-6)
    assert np.allclose(obs[:, 22:24], np.asarray([s.stand_height_m for s in scenarios])[:, None], atol=1e-6)
    assert np.allclose(env.k["heights"].numpy(), [.115, .16, .25, .3, .38])
    geometry = initial_geometry(env)
    found = [None] * len(scenarios)
    for _ in range(700):
        _, _, done, infos = env.step(np.zeros((len(scenarios), 3), dtype=np.float32))
        for i in np.flatnonzero(done):
            if found[i] is None:
                info = infos[i]
                assert info["height_design"] == "range115"
                assert info['height_safety_contract']=='physical_v1'
                found[i] = dict(scenario=asdict(scenarios[i]), success=bool(info["success"]),
                                reason=info["reason"], peak_deg=info["peak_deg"],
                                height_rmse_m=info["height_rmse_m"],
                                min_leg_m=info["min_leg_m"],
                                geometric_margin_passed=info["geometric_margin_passed"],
                                velocity_rmse=info["velocity_rmse"],
                                terrain_evidence_passed=info["terrain_evidence_passed"],
                                base_infeasible_steps=info["base_infeasible_steps"],
                                **{key:info[key] for key in ('height_safety_contract','physical_safety_passed',
                                    'min_fk_leg_m','min_actual_A_leg_m','min_actual_B_leg_m','min_eight_joint_margin_rad',
                                    'max_loop_error_m','max_actual_torque_excess_Nm','max_command_torque_excess_Nm',
                                    'physical_evidence_steps','physical_steps')})
        if all(row is not None for row in found):
            break
    assert all(row is not None for row in found)
    summary = dict(total=len(found), completed=sum(row["reason"] == "completed" for row in found),
                   success=sum(row["success"] for row in found),
                   geometric_margin_passed=sum(row["geometric_margin_passed"] for row in found),
                   physical_safety_passed=sum(row['physical_safety_passed'] for row in found),
                   geometric_limit_m=HEIGHT_115_GEOMETRIC_MIN,
                   by_height={str(h): sum(row["success"] for row in found if row["scenario"]["stand_height_m"] == h)
                              for h in HEIGHTS}, **geometry)
    names = ("wheelleg_warp/test_height_115.py", "wheelleg_warp/native/terrain.py",
             "wheelleg_warp/native/environment.py", "wheelleg_warp/native/controller.py",
             "wheelleg_warp/native/models.py", "wheelleg_ppo/tools/rm_controller.py",
             "wheelleg_ppo/tools/model_lqr.py", "wheelleg_ppo/tools/wheelleg_sim.py",
             "wheelleg_ppo/tools/hardware_profile.py", "wheelleg_ppo/tools/state_estimation.py",
             "wheelleg_ppo/xml/wheelleg.xml", "wheelleg_warp/probe_height_115_margin.py")
    output.mkdir(parents=True)
    (output / "verification.json").write_text(json.dumps(dict(panel=panel, role="public_engineering_not_final_holdout",
        nominal_height_range_m=[.115, .38], height_design="range115", height_safety_contract='physical_v1',residual_scale=0,
        summary=summary, rows=found,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}),
        ensure_ascii=False, indent=2) + "\n")
    print(panel, summary)
    if panel == "flat":
        assert summary["success"] == len(found), "0.115～0.38 m平地工程门未通过，先诊断"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("flat", "regular", "boundary", "normal115"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.panel, args.output)
