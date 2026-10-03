"""Rendering visibility must preserve physical fields and the tested controller."""
from pathlib import Path
import sys,json,hashlib,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'wheelleg_warp'))
from manual_demo import SCENES,scenario,show_terrain
from native.terrain import model
OUT=Path(__file__).resolve().parent
old=OUT/'manual_demo_at_run.py';new=ROOT/'wheelleg_warp/manual_demo.py'
record=json.loads((OUT/'verification.json').read_text())
assert hashlib.sha256(old.read_bytes()).hexdigest()==record['source_sha256']['wheelleg_warp/manual_demo.py']
def nodes(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a=nodes(old);b=nodes(new);unchanged=['scenario','manual_command','Demo','JumpDemo','check']
assert all(a[n]==b[n] for n in unchanged)
fields=('geom_pos','geom_quat','geom_size','geom_friction','geom_contype','geom_conaffinity','geom_condim','geom_solmix','geom_solref','geom_solimp','body_mass','body_inertia','actuator_gainprm')
visible={}
for scene in SCENES:
    m=model(scenario(.3,1.,scene[0]));before={n:getattr(m,n).copy() for n in fields};show_terrain(m)
    for n,v in before.items():np.testing.assert_array_equal(v,getattr(m,n))
    selected=[i for i in range(m.ngeom) if m.geom(i).name.startswith(('bump_','terrain_')) and m.geom_pos[i,2]>-1.]
    assert all(m.geom_group[i]==0 and m.geom_matid[i]==-1 for i in selected)
    assert selected or scene[0]=='flat';visible[scene[0]]=len(selected)
result=dict(passed=True,rendering_only_fields=['geom_group','geom_matid','geom_rgba'],physical_fields_unchanged=list(fields),visible_collision_boxes=visible,
    physical_implementation_AST_unchanged=unchanged,physical_probe_producer_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),current_demo_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
    source_scope='Physical120case and4GPUjump evidence reused against exact archived producer and unchanged physical AST; current GUI separately checked')
path=OUT/'visual_scope_check.json'
if path.exists():assert json.loads(path.read_text())==result
else:path.write_text(json.dumps(result,indent=2)+'\n')
print('PASS display boxes for12scenes; physical model fields unchanged; physical producer archived and AST exact')
