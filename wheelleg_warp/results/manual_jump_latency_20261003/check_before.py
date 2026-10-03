import sys,time,json
from pathlib import Path
sys.path.insert(0,'wheelleg_warp')
import importlib.util
source=Path(__file__).with_name('manual_demo_before.py')
spec=importlib.util.spec_from_file_location('demo_before_latency',source);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
Demo,JumpDemo=old.Demo,old.JumpDemo
x=Demo((.38,))
for _ in range(100):x.step()
t=time.perf_counter();j=JumpDemo(x);enter=time.perf_counter()-t;takeoff=None
for i in range(650):
 j.step()
 if takeoff is None and j.control.jp=='FLY':takeoff=(i+1)*.02
 if 'LAND' in j.phases and j.control.jp=='DRIVE':break
t=time.perf_counter();g=Demo.after_jump(j);leave=time.perf_counter()-t
r=dict(entry_wall_s=enter,request_to_flight_sim_s=takeoff,landing_switch_wall_s=leave)
Path('/tmp/wheelleg_jump_latency_baseline_rerun.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True);g.close()
