"""Reuse the verified18-case execution recorder for native design-gate parity."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=OUT.parent/'nineteenth_round_motion_reference_20261002/production_panel.py'
spec=importlib.util.spec_from_file_location('registered_panel',SOURCE)
panel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=panel;spec.loader.exec_module(panel)
panel.OUT=OUT/'native_panel'


def check():
    panel.check()
    r=json.loads((panel.OUT/'verification.json').read_text())
    for row,margin,full_gate in zip(r['episodes'],r['active_design_margin_rad'],r['original_full_gates']):
        assert row['design_joint_contract']=='active-1p4-v1'
        assert row['min_active_design_margin_rad']==margin
        assert row['design_joint_passed']==(margin>=0)
        assert row['success']==full_gate
        assert row['baseline_version']=='height115-current-vmc-v5-full-design-nominal-boundary-candidate'
    print('PASS native v5 success matches independently recorded full-design gates in all18 episodes',flush=True)


if __name__=='__main__':
    if '--check' not in sys.argv:panel.run()
    check()
