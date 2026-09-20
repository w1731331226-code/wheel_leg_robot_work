"""从项目根目录运行；只复测35V既有记录中的最大吸收功率/能量工况。"""
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()/'tools'))
import hardware_profile as hw
import model_lqr as ml
import wheelleg_sim as sim
import test_dm8009_dynamics as test
from test_jump_stop import run as landing

hw.HIP_BUS_VOLTAGE = 35.0
hw.HIP_RATED_RPM, hw.HIP_NO_LOAD_RPM = 100*35/24, 160*35/24
nominal, _ = sim.load_model(ml.XML, True)
configure, make_state = hw.configure_spec, sim.make_state
out = Path(__file__).resolve().parent
rows = []
for mass, mode, action in [(7, 'jump', None), (8, 'landing_1_0.5', partial(landing, 1, .5, True, None, True))]:
    def configure_payload(spec, track_width=None):
        configure(spec, track_width)
        spec.geom('chassis_lid').mass += mass-7

    def fixed_controller(model, hardware=False, six_state=False):
        assert abs(sum(model.body_mass)-mass) < 1e-9
        return make_state(nominal, hardware, six_state)

    with patch.object(hw, 'configure_spec', configure_payload), patch.object(sim, 'make_state', fixed_controller):
        row, log = test.run(mode, action)
    row['mass_kg'] = mass
    rows.append(row)
    (out/f'{mass}_{mode}.log').write_text(log)
    print(mass, mode, row['rails'], flush=True)
    assert row['action_pass'] and row['peak_window_pass']

files = [ml.XML, sim.__file__, hw.__file__, test.__file__, str(Path('tools/rm_controller.py')),
         str(Path('tools/state_estimation.py')), str(Path('tools/model_lqr.py')),
         str(Path('tools/test_jump_stop.py')), __file__]
report = dict(status='SIMULATED_LOAD_INPUTS_ONLY', hip_bus_V=35, wheel_bus_V=24,
    controller_mass_kg=7, runs=rows, hardware_acceptance=False,
    source_sha256={str(Path(f).resolve().relative_to(Path.cwd())):
                   hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
    note='Per-rail values are sums of motor mechanical powers, not measured DC bus powers. '
         'Absorbed mechanical energy is not battery recovered energy. Torque-squared integral '
         'is a thermal-load proxy, not temperature. Battery/BMS, DC/DC reverse-flow capability, '
         'motor loss/thermal parameters and voltage/current/temperature measurements are missing.')
(out/'load_inputs.json').write_text(json.dumps(report, indent=2)+'\n')
