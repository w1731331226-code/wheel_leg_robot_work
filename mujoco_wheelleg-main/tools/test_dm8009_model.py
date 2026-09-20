"""新尺寸模型最小检查：几何闭合、硬件参数、2×6增益及插值稳定性。"""
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import hardware_profile as hw
import model_lqr as ml
import wheelleg_sim as sim
from rm_controller import SixStateController


def check():
    model, data = sim.load_model(ml.XML, True)
    raw = mujoco.MjModel.from_xml_path(ml.XML)
    assert model.nq == 17 and model.nu == 6 and model.neq == 2
    assert math.isclose(sum(model.body_mass), hw.DESIGN_MASS, abs_tol=1e-10)
    assert math.isfinite(float(sum(raw.body_mass)))
    for m in (raw, model):
        box, lid = m.geom('chassis_box'), m.geom('chassis_lid')
        for side in ('L', 'R'):
            assert math.isclose(abs(m.body('leg'+side).pos[1]), hw.TRACK_WIDTH/2)
            for joint in ('alpha', 'beta'):
                motor = m.geom('motor_'+joint+side)
                assert abs(motor.pos[1])-motor.size[1]-box.size[1] >= .008
                assert lid.pos[2]-lid.size[2]-motor.size[0] >= .0039
    for side in ('L', 'R'):
        a, b = model.body('leg'+side), model.body('leg'+side+'_D')
        assert math.isclose(np.linalg.norm(a.pos-b.pos), .150, abs_tol=1e-12)
        for name in ('kneeA_', 'kneeB_'):
            assert math.isclose(np.linalg.norm(model.body(name+side).pos), .150, abs_tol=1e-12)
        assert math.isclose(np.linalg.norm(model.body('wheel'+side).pos), .270, abs_tol=1e-12)
        assert math.isclose(np.linalg.norm(model.site('couplerB_'+side+'_end').pos), .270, abs_tol=1e-12)
        assert np.allclose(model.geom('wheel_collide_'+side).size, [.05, .0275, .05])
    for key in ('stand', 'crouch'):
        mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
        mujoco.mj_forward(model, data)
        eq = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        assert max(abs(data.efc_pos[eq])) < 1e-12
    for height in np.linspace(sim.L_SQUAT_MIN, sim.L_JUMP, 25):
        qa, qb = sim.ik(height)
        fk = sim.fk_joints(qa, qb)
        assert math.isclose(fk['leg_len'], height, abs_tol=1e-12)
        assert math.isclose(fk['xC'], sim.L5/2, abs_tol=1e-12)
    for hip, rated, peak, rated_rpm, no_load in (
        (True, 20, 40, 175, 280), (False, 2.42, 4.5, 490, 710)):
        assert hw.torque_limit(1000, 0, hip, 0, .001)[0] == peak
        assert hw.torque_limit(-1000, 0, hip, hw.PEAK_WINDOW, .001)[0] == -peak
        assert abs(hw.torque_limit(1000, no_load*math.pi/30, hip, 0, .001)[0]) < 1e-12
        assert hw.torque_limit(1000, rated_rpm*math.pi/30, hip, hw.PEAK_WINDOW, .001)[0] == peak
    controller = SixStateController(model)
    rows = []
    heights = sorted([*controller.heights, *((controller.heights[:-1]+controller.heights[1:])/2)])
    for height in heights:
        ref = ml.equilibrium(model, height)
        a, b = ml.linearize(model, ref)
        if math.isclose(height, sim.L_STAND):
            ml.check_linearization(model, ref, a, b)
        i = int(np.clip(np.searchsorted(controller.heights, height)-1, 0, len(controller.heights)-2))
        t = (height-controller.heights[i])/(controller.heights[i+1]-controller.heights[i])
        k6, ff, theta = [(1-t)*x+t*y for x,y in zip(controller.table[i], controller.table[i+1])]
        assert k6.shape == (2, 6) and np.isfinite(k6).all()
        c, g, _ = ml.vmc_coordinates(ref)
        outer = sum(model.body_mass)/hw.BASELINE_MASS*(sim.KP_LEG*c[6]+sim.KD_LEG*c[7])
        full_gain = g[:,:2]@k6@c[:6] + g[:,2,None]*outer
        radius = float(max(abs(np.linalg.eigvals(a-b@full_gain))))
        assert radius < 1, (height, radius)
        rows.append(dict(height_m=float(height), K6=k6.tolist(), feedforward=ff.tolist(),
                         theta_equilibrium=float(theta), full_closed_radius=radius))
    out = Path(__file__).resolve().parent/'results/continuation_2026-09-09/dm8009_mf9025'
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='MODEL_AND_LOCAL_LQR_PASS', mass_kg=float(sum(model.body_mass)),
        thigh_m=sim.L1, shank_m=sim.L2, hip_spacing_m=sim.L5, track_width_m=hw.TRACK_WIDTH,
        chassis_width_m=float(2*model.geom('chassis_box').size[1]),
        platform_width_m=float(2*model.geom('chassis_lid').size[1]),
        tire_diameter_m=2*hw.WHEEL_RADIUS,
        hip_bus_V=hw.HIP_BUS_VOLTAGE, wheel_bus_V=hw.WHEEL_BUS_VOLTAGE,
        states=['theta','theta_dot','x','vx','pitch','pitch_dot'], gains=rows,
        note='Nominal geometry and local stability only; not jump, hardware or estimator acceptance.')
    root = Path(__file__).resolve().parents[1]
    files = ['xml/wheelleg_dm8009.xml','tools/hardware_profile.py','tools/wheelleg_sim.py',
             'tools/model_lqr.py','tools/rm_controller.py','tools/state_estimation.py','tools/test_dm8009_model.py']
    report['source_sha256'] = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    (out/'model_and_lqr.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('PASS: 150/150/270mm, two closed five-bars, mass/inertia, motor limits, 2x6 gains and 7 stable points')


if __name__ == '__main__':
    check()
