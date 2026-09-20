import contextlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import test_jump_stress as stress
import test_jump_height as height
import wheelleg_sim as sim
out=Path(__file__).parent
rows=[]
options=dict(hardware=True,six_state=True)
for mismatch,disturbance in ((0,False),(0,True),(.01,True),(-.01,True)):
 for speed,trigger in ((1,3.5),(-1,3.75),(0,4)):
  name=f'jump_{mismatch}_{disturbance}_{speed}'
  with (out/(name+'.log')).open('w') as log,contextlib.redirect_stdout(log):
   passed=stress.run(speed,trigger,4 if disturbance else 0,.1 if disturbance else 0,.05,actuator_mismatch=mismatch,**options)
  rows.append(dict(name=name,passed=bool(passed)))
  print(name,passed,flush=True)
  (out/'adjacent.json').write_text(json.dumps(rows,indent=2)+'\n')
for name,action in [('height_min',lambda:height.run_height(0,sim.L_SQUAT_MIN,**options)),('height_max',lambda:height.run_height(0,sim.L_MAX,**options)),('cancel',lambda:height.cancel_on_release(**options)),('commit',lambda:height.prepare_then_commit(**options))]:
 with (out/(name+'.log')).open('w') as log,contextlib.redirect_stdout(log):
  passed=action()
 rows.append(dict(name=name,passed=bool(passed)))
 print(name,passed,flush=True)
 (out/'adjacent.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
