import contextlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import test_robustness as test
import wheelleg_sim as sim
out=Path(__file__).parent
samples=[]
original=sim.control

def observe(model,data,state):
    original(model,data,state)
    wheels=[model.geom('wheel_collide_'+s).id for s in ('L','R')]
    bump=model.geom('single_wheel_bump').id
    samples.append(dict(t=float(data.time),yaw=float(sim.euler(data)[2]),yaw_rate=float(state.yw_f),
        torque=float(state.yaw_torque),ctrl=data.ctrl.tolist(),wheel_speed=[float(state.ws1),float(state.ws2)],
        bump_contact=[any(bump in (c.geom1,c.geom2) and w in (c.geom1,c.geom2) for c in data.contact) for w in wheels]))
with (out/'baseline.log').open('w') as log,contextlib.redirect_stdout(log),patch.object(sim,'control',observe):
    result=test.run_terrain(-1,0,.015,1,True,None,True)
peak=max(samples,key=lambda r:abs(r['yaw']))
contacts=[s for s in samples if any(s['bump_contact'])]
summary=dict(result=result,peak=peak,first_contact=contacts[0],last_contact=contacts[-1],
 max_requested_yaw_torque=max(abs(s['torque']) for s in samples),
 max_wheel_torque=max(max(abs(v) for v in s['ctrl'][4:]) for s in samples))
(out/'baseline.json').write_text(json.dumps(dict(summary=summary,samples=samples),indent=2)+'\n')
print(json.dumps(summary,indent=2))
assert not result['passed'] and result['traversed']
