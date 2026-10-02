"""Recheck rejected causal allocation without replaying or overwriting physical evidence."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from native.terrain import model,HeightTerrainScenario
from select_braking_common_action import batch_constraint_margins


def check():
    out=Path(__file__).resolve().parent
    for folder in (out.parent/'twelfth_round_causal_allocation_20261002',out):
        r=json.loads((folder/'verification.json').read_text());z=np.load(folder/'failed_pair.npz');actual=z['actual'];pred=z['predicted']
        assert not r['completed'] and r['failure']['step']==0 and r['failure']['reason']=='original_predictor_pairing_gate_failed'
        assert np.isfinite(actual).all() and np.isfinite(pred).all()
        command=float(abs(actual[:,:,-12:-6]-pred[:,:,-12:-6]).max())
        assert command==r['failure']['command'] and command>r['original_pairing_gates']['command']
        np.testing.assert_array_equal(actual[0,:,-12:-6],pred[0,:,-12:-6])
        request=np.array(r['decisions'][0]['request']);assert abs(request).max()<=1 and np.sum(abs(request),axis=1).max()<=.1+1e-12
        assert all(min(row['nonlinear_margins'])>=0 for row in r['decisions'][0]['rows'])
        for name,h in r['source_sha256'].items():
            path=folder/'forecaster_at_run.py' if name=='wheelleg_warp/probe_braking_feedback.py' else ROOT/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==h,name
    # Actual five milliseconds remain inside the original physical/design gates.
    z=np.load(out/'failed_pair.npz');s=np.load(out/'current_pre.npz');m=model(HeightTerrainScenario(stand_height_m=.115));a=z['actual']
    g=batch_constraint_margins(m,a[:,:,:m.nq],a[:,:,m.nq:m.nq+m.nv],a[:,:,-12:-6],a[:,:,-6:],s['v'])
    assert np.all(g>=0)
    print('PASS rejection recheck: original command gate fails, initial commands equal, requests legal, actual5ms gates pass; no full episode/admission')


if __name__=='__main__':check()
