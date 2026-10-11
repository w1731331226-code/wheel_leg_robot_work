"""Locate already accepted gain requests before target contact; no causal necessity claim."""
import json
import numpy as np
from analyze_complete_contact import I
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'
DATA=BASE/'reflection_retention_falsification_v1'


def complete_precontact_packets(trace,onset):
    starts=np.arange(0,len(trace),40);ends=np.minimum(starts+39,len(trace)-1)
    return starts,(starts+39 < len(trace)) & (trace[ends,I['post_s']] < onset)


def run():
    fake=np.zeros((81,len(I)));fake[:,I['post_s']]=np.arange(1,82)*.0005
    starts,keep=complete_precontact_packets(fake,.04)
    np.testing.assert_array_equal(starts,[0,40,80]);np.testing.assert_array_equal(keep,[True,False,False])
    np.testing.assert_array_equal(complete_precontact_packets(fake,.1)[1],[True,True,False])
    review=json.loads((DATA/'review.json').read_text());exposure=json.loads((DATA/'exposure_audit.json').read_text())
    assert review['verified'] and exposure['verified'] and exposure['review_sha256']==sha(DATA/'review.json')
    assert len(exposure['pairs'])==8
    result=[];hashes={}
    for pair in exposure['pairs']:
        condition=pair['conditions'][0];label=condition['condition'];assert label.endswith('_original')
        directory=DATA/label;file=directory/'result.json'
        assert sha(file)==review['raw_sha256'][str(file.relative_to(ROOT))]
        row=json.loads(file.read_text())['runs'][0];assert row['success'] and row['seed']==pair['case']
        files=[file]+[directory/row[k]['path'] for k in ('actor_trace','complete_trace')]
        for f in files:
            h=sha(f);assert h==review['raw_sha256'][str(f.relative_to(ROOT))];hashes[str(f.relative_to(ROOT))]=h
        with np.load(files[1]) as z:actor=z['trace']
        with np.load(files[2]) as z:trace=z['trace']
        onset=condition['first_target_contact_s'];assert onset is not None
        starts,keep=complete_precontact_packets(trace,onset);assert len(starts)==len(actor) and keep.any()
        action=actor[:,962:];time=trace[starts,I['pre_s']]
        active=np.flatnonzero(np.max(abs(action),axis=1)>1e-6)
        before=action[keep];last=int(np.flatnonzero(keep)[-1])
        result.append(dict(condition=label,case=pair['case'],model=pair['model'],first_target_contact_s=onset,
            complete_precontact_actor_packets=int(keep.sum()),first_nonzero_request_s=float(time[active[0]]) if len(active) else None,
            precontact_nonzero_packets=int((np.max(abs(before),axis=1)>1e-6).sum()),
            precontact_channel_rms=np.sqrt(np.mean(before.astype(float)**2,axis=0)).tolist(),
            last_complete_precontact_packet_start_s=float(time[last]),last_public_route_y_m=float(actor[last,389])))
    out=BASE/'round358_gain_precontact.json';assert not out.exists()
    atomic_json(out,dict(round=358,verified=True,records=result,input_sha256=hashes,review_sha256=sha(DATA/'review.json'),
        exposure_sha256=sha(DATA/'exposure_audit.json'),auditor_sha256=sha(__file__),new_physics_steps=0,new_model_forward_rows=0,new_learning_samples=0,
        timing='Actor packets count as precontact only if their last recorded post_s is strictly before the accepted first positive target-normal contact; endpoint equality is excluded.',
        limits='Target onset is private offline annotation, never policy input. Ground contacts already exist. Pre-target action/path changes are associations, not evidence of necessity, obstacle anticipation or a deployable contact trigger. Do not infer that zeroing the requests preserves or destroys success without intervention.'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':run()
