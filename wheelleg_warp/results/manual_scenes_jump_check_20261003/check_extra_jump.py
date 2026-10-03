from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,'/home/wmt/wheel_leg_robot_work/wheelleg_warp')
from manual_demo import Demo,JumpDemo
from training_contract import digest
rows=[]
for height,moving in [(.25,False),(.3,True)]:
 g=Demo((height,))
 for tick in range(200):g.step(speed=1. if moving and tick>=100 else 0.)
 j=JumpDemo(g)
 for _ in range(400):
  j.step(speed=1. if moving else 0.)
  assert not j.finished
 assert {'SQUAT','JUMP','FLY','LAND','DRIVE'}.issubset(j.phases) and j.control.jp=='DRIVE'
 rows.append(dict(height_m=height,moving=moving,phases=sorted(j.phases),device=str(j.physics.data.qpos.device),returned_to_drive=True))
 j.close()
root=Path('/home/wmt/wheel_leg_robot_work');out=root/'wheelleg_warp/results/manual_scenes_jump_check_20261003/extra_jump_check.json'
out.write_text(json.dumps(dict(passed=True,rows=rows,source_sha256=digest(root/'wheelleg_warp/manual_demo.py'),physics_sha256=digest(root/'wheelleg_warp/fast_physics.py')),indent=2)+'\n')
print('PASS250mm GPU jump and300mm moving GPU jump, flight/landing/drive')
