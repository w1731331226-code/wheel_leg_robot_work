"""Close original completed numeric work after its known JSON integer failure."""
import json
import numpy as np
from predict_contact_holdout import OUT,metrics
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run():
    p=json.loads((OUT/'registration.json').read_text());ledger=json.loads((OUT/'completed_jobs.json').read_text());fix=json.loads((OUT/'serialization_correction_registration.json').read_text())
    assert sha(__file__)==fix['finalizer_sha256'] and sha(OUT/'registration.json')==fix['prediction_registration_sha256']
    assert not (OUT/'completion.json').exists() and ledger['states']==576 and all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    records=[]
    for r in ledger['records']:
        assert sha(OUT/r['path'])==r['sha256'];batch=json.loads((OUT/r['path']).read_text())['records'];assert len(batch)==r['states']==144;records.extend(batch)
    assert len(records)==576
    groups=metrics(records,p['numerical_reserve_rad'])
    groups=json.loads(json.dumps(groups,default=lambda v:v.item() if isinstance(v,np.generic) else (_ for _ in ()).throw(TypeError(type(v).__name__))))
    correction=dict(reason='Only convert NumPy integer summary counts to JSON native values; unchanged predictions, descriptors, corners, gates and source. No new physics/model steps.',registration_sha256=sha(OUT/'serialization_correction_registration.json'),finalizer_sha256=sha(__file__))
    atomic_json(OUT/'completion.json',dict(states=576,cpu_corner_steps=9216,cpu_actual_geometry_steps=576,training_updates=0,records=ledger['records'],groups=groups,
        development_gate=groups['all/all']['gpu_misses']==groups['all/all']['oracle_misses']==0,registration_sha256=sha(OUT/'registration.json'),serialization_correction=correction))
    atomic_json(OUT/'progress.json',dict(status='complete',states=576));print(json.dumps(groups['all/all']))


if __name__=='__main__':run()
