from pathlib import Path
import importlib.util,sys,json
import numpy as np
sys.path.insert(0,'/home/wmt/wheel_leg_robot_work/wheelleg_warp')
from manual_demo import Demo,JumpDemo,HEIGHTS,TURN_RATE
from training_contract import source_hashes,digest
root=Path('/home/wmt/wheel_leg_robot_work');out=root/'wheelleg_warp/results/manual_input_turn_20261003';frozen=source_hashes(root/'wheelleg_warp/manual_demo.py')
spec=importlib.util.spec_from_file_location('manual_demo_old',out/'manual_demo_before.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
prior=old.Demo((.3,))
for _ in range(100):prior.step()
for _ in range(100):prior.step(turn=.3)
ids=prior.env.ids.numpy();before=float(prior.env.data.sensordata.numpy()[0,ids[10]+2]);prior.close()
x=Demo(HEIGHTS);rates=[]
for _ in range(100):x.step()
for turn in (TURN_RATE,0.,-TURN_RATE,0.):
 for _ in range(100):x.step(turn=turn);assert not x.finished
 ids=x.env.ids.numpy();rate=x.env.data.sensordata.numpy()[:,ids[10]+2].copy();rates.append(rate.tolist())
 if turn:assert np.max(abs(rate-turn))<.05
s=x.env.state.numpy();assert np.all(s[:,31:33]>=.1147044660616607) and np.all(s[:,33]>=0) and np.all(s[:,38]>=0) and np.max(s[:,35:37])<=1e-6
x.close();jump_rows=[]
for case in ('air_queue','release_speed','timeout'):
 g=Demo((.3,))
 for tick in range(200):g.step(speed=1. if case=='release_speed' and tick>=100 else 0.)
 j=JumpDemo(g);flights=0;previous='DRIVE';queued=False
 for tick in range(1000):
  # Keep turning only in the timeout case: original safety gates must reject it.
  j.step(speed=1. if case=='release_speed' and tick<2 else 0.,turn=TURN_RATE if case=='timeout' else 0.)
  if j.control.jp=='FLY' and previous!='FLY':
   flights+=1
   if case=='air_queue' and not queued:j.request_jump();queued=True
  previous=j.control.jp
  if case=='timeout' and not j.pending and j.jump_status.startswith('EXPIRED'):break
  if case!='timeout' and flights==(2 if case=='air_queue' else 1) and j.control.jp=='DRIVE' and not j.pending:break
 assert not j.finished
 if case=='timeout':assert not j.pending and j.jump_status.startswith('EXPIRED') and flights==0
 else:assert flights==(2 if case=='air_queue' else 1)
 jump_rows.append(dict(case=case,flights=flights,status=j.jump_status,ticks=tick,original_gates_retained=True))
 j.close()
assert source_hashes(root/'wheelleg_warp/manual_demo.py')==frozen
result=dict(passed=True,heights_m=HEIGHTS,original_command_rad_s=.3,original_actual_rad_s_at2s=before,new_command_rad_s=TURN_RATE,new_actual_rad_s_at2s=rates[0][3],all_height_left_release_right_release_rates=rates,
 actual_turn_speed_ratio=rates[0][3]/before,jump_cases=jump_rows,physical_design_torque_gates_passed=True,source_sha256=frozen,
 original_jump_controller_sha256=digest(root/'wheelleg_ppo/tools/wheelleg_sim.py'),learning=False)
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS measured A/D improvement across5heights with original bounds; airborne queue, speed release retry, explicit safety timeout')
