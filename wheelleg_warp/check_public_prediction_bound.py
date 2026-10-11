"""Exact public-input collisions bound point-prediction error; no model fitting."""
import json
import numpy as np
from review_yaw_sector import ROOT, sha
from analyze_complete_contact import I
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'
OUT=BASE/'public_history_prediction_qualification_v1'


def matched_bound(x,u,y,other_x,other_u,other_y):
    n=min(len(x),len(other_x));same=np.all(x[:n]==other_x[:n],axis=1)&np.all(u[:n]==other_u[:n],axis=1)
    indices=np.flatnonzero(same)
    return indices,abs(y[indices]-other_y[indices])/2


def self_check():
    x=np.zeros((3,481));u=np.zeros((3,6));y=np.zeros((3,3));other=x.copy();other[1,0]=1e-12
    other_u=u.copy();other_u[2,0]=.1;other_y=y.copy();other_y[:,0]=4
    indices,bound=matched_bound(x,u,y,other,other_u,other_y)
    np.testing.assert_array_equal(indices,[0]);np.testing.assert_array_equal(bound,[[2,0,0]])
    midpoint=(y+other_y)/2
    np.testing.assert_array_equal(abs(midpoint[indices]-y[indices]),bound)
    np.testing.assert_array_equal(abs(midpoint[indices]-other_y[indices]),bound)
    assert len(matched_bound(x,u,y,x+1,u,y)[0])==0
    print('PASS359 exact-match/unequal-action rejection and midpoint tightness',flush=True)


def run():
    self_check();p=json.loads((OUT/'proposal.json').read_text());assert not (OUT/'review.json').exists()
    for mapping in (p['source_sha256'],p['evidence_sha256']):
        assert all(sha(ROOT/f)==h for f,h in mapping.items())
    study=BASE/'fixed_force_prequalification_v1';accepted=json.loads((study/'baseline_raw_review.json').read_text())
    completion=json.loads((study/'baseline/completion.json').read_text())
    assert accepted['verified'] and accepted['completion_sha256']==sha(study/'baseline/completion.json')
    result_hashes={str((study/'baseline'/r['path']).relative_to(ROOT)):r['sha256'] for r in completion['records']}
    groups={};hashes={};loaded={};row_cache={};physical_rows=0
    for case in p['cases']:
        path=ROOT/case['result'];assert sha(path)==result_hashes[case['result']]
        if str(path) not in row_cache:row_cache[str(path)]=json.loads(path.read_text())['runs']
        row=next(r for r in row_cache[str(path)] if r['seed']==case['seed']);assert row['scenario']==case['scenario']
        arrays=[]
        for field in ('actor_trace','complete_trace'):
            file=path.parent/row[field]['path'];h=sha(file)
            assert h==row[field]['sha256']
            # The old physical receipt binds dense files; actor hashes are bound by the accepted result manifest.
            if field=='complete_trace':assert h==accepted['raw_sha256'][str(file.relative_to(ROOT))]
            hashes[str(file.relative_to(ROOT))]=h
            with np.load(file) as z:arrays.append(z['trace'])
        actor,t=arrays;assert len(t)==row['physical_steps'];physical_rows+=len(t)
        assert actor.shape==(int(np.ceil(len(t)/40)),968) and np.isfinite(actor).all()
        np.testing.assert_array_equal(actor[:,:481],actor[:,481:962])
        n=len(t)//40;start=np.arange(n)*40;end=start+39
        np.testing.assert_allclose(t[end,I['post_s']]-t[start,I['pre_s']],.02,rtol=0,atol=1e-12)
        y=np.rad2deg(t[end][:,[I['roll'],I['pitch'],I['yaw']]])
        assert np.max(abs(y))<90,'Euler wrap requires a different predeclared target'
        loaded[case['seed']]=dict(x=actor[:n,:481],u=actor[:n,962:],y=y,time=t[start,I['pre_s']])
        scene=dict(case['scenario']);left=scene.pop('height_l');right=scene.pop('height_r');scene.pop('terrain_seed')
        assert (left==0)!=(right==0);scene['obstacle_height']=max(left,right)
        key=json.dumps(scene,sort_keys=True);side='left' if left else 'right'
        assert side not in groups.setdefault(key,{})
        groups[key][side]=case['seed']
    assert len(loaded)==40 and len(groups)==20
    records=[];witnesses={};total=0;threshold=p['engineering_gate']['maximum_component_prediction_error_deg']
    for number,(key,pair) in enumerate(sorted(groups.items())):
        assert set(pair)=={'left','right'};left,right=loaded[pair['left']],loaded[pair['right']]
        n=min(len(left['x']),len(right['x']));np.testing.assert_array_equal(left['time'][:n],right['time'][:n])
        indices,bound=matched_bound(left['x'],left['u'],left['y'],right['x'],right['u'],right['y']);total+=len(indices)
        record=dict(pair=number,scenario=json.loads(key),cases=pair,matched_complete_packets=len(indices),falsifies_gate=bool(len(bound) and bound.max()>threshold))
        if len(indices):
            maxima=bound.max(axis=0);k=int(np.unravel_index(np.argmax(bound),bound.shape)[0]);index=int(indices[k])
            record.update(maximum_required_component_radius_deg=maxima.tolist(),witness_actor_index=index,witness_start_s=float(left['time'][index]),
                witness_left_next_deg=left['y'][index].tolist(),witness_right_next_deg=right['y'][index].tolist())
            witnesses[f'pair_{number}_public481']=left['x'][index]
            witnesses[f'pair_{number}_submitted6']=left['u'][index]
            witnesses[f'pair_{number}_left_next_deg']=left['y'][index]
            witnesses[f'pair_{number}_right_next_deg']=right['y'][index]
        records.append(record)
    np.savez_compressed(OUT/'witnesses.npz',**witnesses)
    falsified=[r['pair'] for r in records if r['falsifies_gate']]
    atomic_json(OUT/'review.json',dict(round=359,verified=True,episodes=40,scenario_pairs=20,matched_complete_packets=total,
        physical_rows_read=physical_rows,records=records,falsifying_pairs=falsified,point_prediction_gate_falsified=bool(falsified),
        proposal_sha256=sha(OUT/'proposal.json'),baseline_review_sha256=sha(study/'baseline_raw_review.json'),input_sha256=hashes,
        witness_sha256=sha(OUT/'witnesses.npz'),reviewer_sha256=sha(__file__),new_physics_steps=0,new_model_forward_rows=0,new_learning_samples=0,
        interpretation='Triangle-inequality lower bound for exact raw481/current-action equality under the same known closed-loop controller. Nominal microstep commands may diverge after hidden contact. No model fitting or proof of uncontrollability; the 1deg engineering gate does not alter task thresholds. No claim that wider/set-valued or additional-sensor prediction is impossible.',
        next='360 direction/framework review must use the bound to accept or reject further qualification; no controller/PPO admission from a non-falsified necessary condition.'))
    print('DONE359 pairs',len(records),'matched',total,'falsified',falsified,'max radius',max(max(r.get('maximum_required_component_radius_deg',[0])) for r in records),flush=True)


if __name__=='__main__':run()
