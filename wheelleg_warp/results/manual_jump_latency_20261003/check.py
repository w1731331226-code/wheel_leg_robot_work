from pathlib import Path
import sys,time,json
import numpy as np
sys.path.insert(0,'wheelleg_warp')
from manual_demo import Demo,JumpDemo,HEIGHTS,sim
from training_contract import source_hashes
root=Path.cwd();out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent/'rerun';out.mkdir(parents=True,exist_ok=False);rows=[];frozen=source_hashes(root/'wheelleg_warp/manual_demo.py')
for h,moving in [(h,False) for h in HEIGHTS]+[(.3,True)]:
 g=Demo((h,));original=id(g);model_id=id(g.env.model);jump_id=id(g.jump_physics)
 for tick in range(180):g.step(speed=1. if moving and tick>=80 else 0.)
 t=time.perf_counter();j=JumpDemo(g);enter=time.perf_counter()-t;flight=None;phase_times={};joint_margin=1e9
 for tick in range(650):
  j.step(speed=1. if moving else 0.);assert not j.finished
  phase_times.setdefault(j.control.jp,(tick+1)*.02)
  if flight is None and j.control.jp=='FLY':flight=(tick+1)*.02
  if 'LAND' in j.phases and j.control.jp=='DRIVE' and not j.pending:break
 assert flight is not None and j.control.jp=='DRIVE',(h,phase_times,j.jump_status)
 q,v=j.pose();t=time.perf_counter();g=Demo.after_jump(j);leave=time.perf_counter()-t
 np.testing.assert_array_equal(q,g.pose()[0]);np.testing.assert_array_equal(v,g.pose()[1])
 assert id(g)==original and id(g.env.model)==model_id and id(g.jump_physics)==jump_id
 for _ in range(150):g.step(speed=1. if moving else 0.);assert not g.finished
 q,_=g.pose();ids=g.env.ids.numpy();actual=np.mean([sim.fk_joints(q[ids[2*s]],q[ids[2*s+1]])['leg_len'] for s in range(2)])
 state=g.env.state.numpy()[0]
 row=dict(height=h,moving=moving,flight_s=flight,phases=phase_times,entry_wall_s=enter,return_wall_s=leave,actual_restored_m=float(actual),reference_restored_m=float(g.height[0]),physical_min_m=float(min(state[31:33])),design_margin=float(state[38]),torque_excess=float(max(state[35:37])))
 rows.append(row);(out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print('CASE',row,flush=True)
 assert abs(actual-h)<.002 and abs(float(g.height[0])-h)<1e-8
 assert min(state[31:33])>=.1147044660616607 and state[33]>=0 and state[38]>=0 and max(state[35:37])<=1e-6
 g.close()
assert source_hashes(root/'wheelleg_warp/manual_demo.py')==frozen
(out/'verification.json').write_text(json.dumps(dict(passed=True,rows=rows,resources_reused=True,handoff_q_v_exact=True,source_sha256=frozen),indent=2)+'\n')
print('PASS all5heights plus moving jump; automatic original-height restore; exact q/v and cached resources')
