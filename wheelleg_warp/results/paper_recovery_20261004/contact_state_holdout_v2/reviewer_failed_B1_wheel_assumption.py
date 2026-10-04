"""Offline identity, event, discrete-state and artifact audit; no prediction scores."""
from pathlib import Path
import argparse,json,math
import numpy as np
from review_yaw_sector import ROOT,sha,success


def run(out):
    load=lambda name:json.loads((out/name).read_text())
    reg=load('registration.json');contract=load('collector_contract.json');done=load('collection_completion.json');ledger=load('collection_jobs.json')
    assert done['completed_episodes']==ledger['completed_episodes']==reg['rollout_budget']==192
    assert done['training_updates']==reg['training_updates']==contract['training_updates']==0
    assert done['records']==ledger['records'] and done['collector_contract_sha256']==sha(out/'collector_contract.json')
    assert contract['registration_sha256']==sha(out/'registration.json')
    assert all(sha(ROOT/n)==v for n,v in contract['source_sha256'].items())
    assert not reg['old_gate_or_final_used'] and len({c['seed'] for c in reg['cases']})==48
    assert [c['seed'] for c in reg['cases']]==list(range(5700000,5700048))
    assert load('collector_self_check.json')['graph_buffer_owners_alive']
    labels=['B1']+[str(m['seed']) for m in reg['models']]
    assert [r['label'] for r in done['records']]==labels
    for m in reg['models']:
        p=Path(m['prefix']);assert sha(p.with_suffix('.zip'))==m['checkpoint']['checkpoint_sha256'] and sha(p.with_suffix('.pkl'))==m['checkpoint']['normalization_sha256']
    counts={};total=0;missing_total=0
    for record in done['records']:
        label=record['label'];base=out/'collection'
        for suffix,key in [('states.npz','states_sha256'),('episodes.json','episodes_sha256'),('missing_events.json','missing_sha256')]:assert sha(base/f'{label}_{suffix}')==record[key]
        z=np.load(base/f'{label}_states.npz',allow_pickle=False);episodes=json.loads((base/f'{label}_episodes.json').read_text());missing=json.loads((base/f'{label}_missing_events.json').read_text())['events']
        rows=episodes['runs'];state=z['state'];world=z['world'];event=z['event'];ids=z['ids']
        assert len(rows)==48 and state.shape==(record['result']['snapshots'],103) and np.isfinite(state).all()
        for row,case in zip(rows,reg['cases']):
            assert row['seed']==case['seed'] and row['scenario']==case['scenario'] and row['success']==success(row)
            assert row['physical_evidence_steps']==row['physical_steps']>0 and math.isclose(row['duration_s'],row['physical_steps']*.0005,abs_tol=1e-10)
            physical=min(row['min_actual_A_leg_m'],row['min_actual_B_leg_m'])>=row['geometric_limit_m'] and row['min_eight_joint_margin_rad']>=0 and max(row['max_actual_torque_excess_Nm'],row['max_command_torque_excess_Nm'])<=1e-6
            assert row['physical_safety_passed']==physical and row['design_joint_passed']==(row['min_active_design_margin_rad']>=0)
        pairs=[(int(w),int(e)) for w,e in zip(world,event)]
        for s,w,e in zip(state,world,event):
            assert 0<=w<48 and 0<=e<3 and 0<=s[49]<rows[w]['duration_s']
            if e<2:assert [1.5,3.5][e]-1e-10<=s[49]<[1.5,3.5][e]+.0005+1e-10
            else:assert s[101]==0 and s[102]>0
            assert all(int(mask)==mask and 0<=mask<=15 for mask in s[101:103])
            np.testing.assert_allclose(s[50:56],(s[56:62]+s[62:68]).astype(np.float32),rtol=0,atol=1e-6)
            assert np.all(s[66:68]==0)  # L2 and registered B1 have no wheel residual.
            q=s[:17][ids[:4]].astype(np.float32);v=s[85:101][ids[4:8]].astype(np.float32)
            expected=(q+np.float32(.0005)*v).astype(np.float32)
            np.testing.assert_array_equal(s[68:85][ids[:4]],expected)
        for m in missing:
            w=m['world'];e=m['event'];assert m['reason']==rows[w]['reason'] and m['duration_s']==rows[w]['duration_s']
            if e<2:assert rows[w]['duration_s']<=[1.5,3.5][e]+.0005
            else:assert rows[w]['touched_contact_mask']|rows[w]['touched_terrain_contact_mask']==0
            pairs.append((w,e))
        assert len(pairs)==len(set(pairs))==144 and set(pairs)=={(w,e) for w in range(48) for e in range(3)}
        assert len(missing)==record['result']['missing_events']
        assert episodes['summary']['success_count']==sum(r['success'] for r in rows)
        assert episodes['physical']==sum(r['physical_safety_passed'] for r in rows) and episodes['design']==sum(r['design_joint_passed'] for r in rows)
        for k in ['summary','physical','design']:assert record['result'][k]==episodes[k]
        counts[label]=dict(snapshots=len(state),events=[int(np.sum(event==i)) for i in range(3)],missing=len(missing),success=episodes['summary']['success_count'],physical=episodes['physical'],design=episodes['design'])
        total+=len(state);missing_total+=len(missing)
    assert total+missing_total==reg['maximum_snapshots']==576
    report=dict(verified=True,completed_episodes=192,unique_scenarios=48,snapshots=total,missing_events=missing_total,training_updates=0,counts=counts,registration_sha256=sha(out/'registration.json'),completion_sha256=sha(out/'collection_completion.json'),reviewer_sha256=sha(__file__),limits='Collection integrity only. No predictor scores or control safety/generalization admission. Original v1 entirely rejected due freed graph buffers; these are unchanged-case corrective replays, not a fresh untouched namespace.')
    (out/'collection_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
