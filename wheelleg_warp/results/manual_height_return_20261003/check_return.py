from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,'/home/wmt/wheel_leg_robot_work/wheelleg_warp')
from manual_demo import Demo,JumpDemo,sim
from training_contract import source_hashes
root=Path('/home/wmt/wheel_leg_robot_work');out=root/'wheelleg_warp/results/manual_height_return_20261003';frozen=source_hashes(root/'wheelleg_warp/manual_demo.py')
g=Demo((.115,))
for _ in range(100):g.step()
j=JumpDemo(g)
for ticks in range(500):
 j.step()
 if 'LAND' in j.phases and j.control.jp=='DRIVE':break
assert not j.finished and j.control.jp=='DRIVE'
q,v=j.pose();g=Demo.after_jump(j);q2,v2=g.pose()
np.testing.assert_array_equal(q,q2);np.testing.assert_array_equal(v,v2)
for _ in range(300):
 h=float(g.height[0]);g.set_height(max(.115,h-.0004));g.step();assert not g.finished
q3,_=g.pose();ids=g.env.ids.numpy();actual=float(np.mean([sim.fk_joints(q3[ids[2*s]],q3[ids[2*s+1]])['leg_len'] for s in range(2)]))
assert float(g.height[0])==.115 and abs(actual-.115)<.0005
state=g.env.state.numpy()[0];assert state[37]==state[0] and state[0]==12000;assert min(state[31:33])>=.1147044660616607 and state[33]>=0 and state[38]>=0 and max(state[35:37])<=1e-6
assert source_hashes(root/'wheelleg_warp/manual_demo.py')==frozen
(out/'verification.json').write_text(json.dumps(dict(passed=True,landing_phases=sorted(j.phases),landing_tick=ticks,
 position_velocity_unchanged_on_return=True,target_m=.115,actual_m=actual,physical_evidence_count_matches=True,min_actual_chain_m=float(min(state[31:33])),min_design_margin_rad=float(state[38]),source_sha256=frozen),indent=2)+'\n')
g.close();print('PASS landing handoff preserves q/v; ground range restored; actual115mm, physical/design gates pass')
