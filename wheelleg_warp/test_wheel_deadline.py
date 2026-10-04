"""Single preregistered nominal-wheel/body coordinate correction, no gain search."""
from pathlib import Path
import argparse
import numpy as np
import test_deadline_reference as deadline

def reference(position,direction):
    remaining=np.maximum(1.5-direction*(position[:,0]+.075),.05)
    return np.clip(np.arctan2(-direction*position[:,1],remaining),-np.deg2rad(5),np.deg2rad(5))

def run(out):
    saved_reference=deadline.reference;saved_writer=deadline.base.atomic_json
    def write(path,value):
        value=dict(value);value['verifier_sha256']=deadline.base.digest(__file__)
        value['deadline_harness_sha256']=deadline.base.digest(Path(deadline.__file__))
        if path.name=='preregistration.json':
            value.update(formula='B1unclippedPD+(nominalyawKp0.4+B1kp)*arm*clip(atan2(-direction*y,max(1.5-direction*(x+0.075),0.05)),+-5deg) when abs(command)>0.05; total clipped to original0.3Nm',
                derivation='Nominal upright wheel center is +0.075m from body x; signed wheel progress=direction*(x+.075). Public step-front1.55 minus radius.05 gives1.5m. Constant nominal offset, not actual passive-state oracle; no fitted gains.',
                rejection='Single coordinate correction only: require all12 correct-reference cases to satisfy original gates in both directions; otherwise reject this reference branch, no gain/reference-bound sweep',
                nominal_wheel_body_offset_m=.075)
        if path.name=='result.json':value['conclusion_scope']='Single nominal coordinate correction under ideal odometry and known public obstacle geometry; no learning/independent-test/novelty claim'
        return saved_writer(path,value)
    position=np.array([[0.,.12],[0.,.12]]);directions=np.array([1,-1])
    assert abs(reference(position,directions)[0])>abs(reference(position,directions)[1])
    deadline.reference=reference;deadline.base.atomic_json=write
    try:deadline.run(out)
    finally:deadline.reference=saved_reference;deadline.base.atomic_json=saved_writer

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
