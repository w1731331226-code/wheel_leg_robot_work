"""Frozen 2x2 offline contact-position/frame-gap intervention, no controller."""
import json
import numpy as np
import mujoco
from contact_manifold_causal import OUT,BASE,TOL,FIELDS,entries,data,advance,contract
from probe_contact_uncertainty import construct
from native.terrain import model
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

ARMS={'unchanged_local_plane':(False,False),'replace_contact_positions_only':(True,False),
      'replace_contact_frames_and_gaps_only':(False,True),'replace_both':(True,True)}
FLIP=np.diag([-1.,-1.,1.])  # Fixed pair-order conversion; preserves right handedness.


def contacts(m,d):
    wheels={m.geom('wheel_collide_'+s).id:s for s in ['L','R']};records=[]
    for i in range(d.ncon):
        c=d.contact[i];a,b=map(int,c.geom)
        if a in wheels and m.geom_bodyid[b]==0:wheel=a;other=b;slot=0
        elif b in wheels and m.geom_bodyid[a]==0:wheel=b;other=a;slot=1
        else:raise AssertionError('Unexpected non-wheel/world contact')
        frame=c.frame.reshape(3,3).copy();canonical=FLIP@frame if slot==0 else frame.copy();assert np.linalg.det(canonical)>0
        name=m.geom(other).name;descriptor=int(name.removeprefix('current_support_')) if name.startswith('current_support_') else None
        records.append(dict(index=i,side=wheels[wheel],wheel_slot=slot,descriptor_index=descriptor,geom=list(map(int,c.geom)),pos=c.pos.tolist(),frame=frame.tolist(),canonical_frame=canonical.tolist(),gap=float(c.dist),
            dim=int(c.dim),friction=c.friction.tolist(),solref=c.solref.tolist(),solreffriction=c.solreffriction.tolist(),solimp=c.solimp.tolist()))
    return records


def trial(m,state,source,arm):
    pos,frame_gap=ARMS[arm];d=data(m,state);captured={}
    def override(m,d):
        before=contacts(m,d);assert len(before)==len(source);expected=[];used=set();q=d.qpos.copy();v=d.qvel.copy();ctrl=d.ctrl.copy()
        for b in before:
            j=b['descriptor_index'];assert j is not None and j not in used;used.add(j);a=source[j];assert a['side']==b['side'] and a['dim']==b['dim']
            for n in ['friction','solref','solreffriction','solimp']:np.testing.assert_allclose(a[n],b[n],rtol=0,atol=TOL)
            c=d.contact[b['index']];target=np.array(a['canonical_frame']);target=FLIP@target if b['wheel_slot']==0 else target
            want_pos=np.array(a['pos'] if pos else b['pos']);want_frame=target if frame_gap else np.array(b['frame']);want_gap=a['gap'] if frame_gap else b['gap']
            if pos:c.pos[:]=want_pos
            if frame_gap:c.frame[:]=want_frame.reshape(9);c.dist=want_gap
            expected.append(dict(index=b['index'],descriptor_index=j,pos=want_pos.tolist(),frame=want_frame.tolist(),gap=float(want_gap),target_frame_from_source=target.tolist()))
        assert used==set(range(len(source)))
        np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v);np.testing.assert_array_equal(d.ctrl,ctrl)
        captured.update(before=before,expected=expected)
    advance(m,d,override);after=contacts(m,d);assert len(after)==len(captured['expected'])
    for b,want in zip(after,captured['expected']):
        assert b['index']==want['index'] and b['descriptor_index']==want['descriptor_index']
        np.testing.assert_array_equal(b['pos'],want['pos']);np.testing.assert_array_equal(b['frame'],want['frame']);assert b['gap']==want['gap']
    return dict(before=captured['before'],expected=captured['expected'],after=after,next={n:getattr(d,n).copy().tolist() for n in FIELDS},time_s=float(d.time),ncon=int(d.ncon),nefc=int(d.nefc))


def freeze():
    p=contract();c=json.loads((OUT/'compatibility_review.json').read_text());assert c['verified'] and c['states']==603 and not (OUT/'intervention_contract.json').exists()
    reg=json.loads((OUT/'registration.json').read_text());assert list(ARMS)==reg['branches']
    atomic_json(OUT/'intervention_contract.json',dict(source_contract_sha256=sha(OUT/'source_contract.json'),compatibility_review_sha256=sha(OUT/'compatibility_review.json'),
        source_sha256={**p['source_sha256'],'wheelleg_warp/intervene_contact_manifold.py':sha(__file__)},arms={n:list(v) for n,v in ARMS.items()},cpu_producer_steps=2412,cpu_review_steps=2412,
        conversion='Canonical wheel-directed right-handed frame: if wheel is geom0, premultiply diag(-1,-1,1); inverse same transform for target order. No normalization or outcome-dependent normal flip.',
        persistence='Every point/frame/gap after implicit must equal requested values exactly; body state/command untouched by override. No later collision pass allowed.',
        information='All actual parameters/current full contact fields only offline counterfactual; no predictor/actor deployment, continuous bounds or safety qualification.',training_updates=0,new_gpu_steps=0))
    print('FROZEN 603 states x4 arms before intervention',flush=True)


def verify():
    p=json.loads((OUT/'intervention_contract.json').read_text());contract()
    assert p['source_contract_sha256']==sha(OUT/'source_contract.json') and p['compatibility_review_sha256']==sha(OUT/'compatibility_review.json')
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items());return p


def reference():
    r=json.loads((OUT/'compatibility.json').read_text());return {(x['cohort'],x['label'],x['index'],x['model']):x['standard'] for x in r['records']}


def summarize(rows):
    result={}
    for cohort in ['new_saved_states','old_critical_fixtures']:
        selected=[r for r in rows if r['cohort']==cohort];group={}
        for arm in ARMS:
            a=[r['arms'][arm] for r in selected]
            group[arm]=dict(states=len(a),max_active_q_error_rad=max(r['active_q_error_rad'] for r in a),mean_active_q_error_rad=float(np.mean([r['active_q_error_rad'] for r in a])),
                max_full_q_error_rad=max(r['full_q_error_rad'] for r in a),max_full_v_error=max(r['full_v_error'] for r in a),within_original_reconstruction_tolerance=sum(r['active_q_error_rad']<=TOL for r in a))
        result[cohort]=group
    return result


def run():
    verify();assert not (OUT/'intervention_progress.json').exists();refs=reference();rows=[];steps=0
    try:
        for cohort,label,i,state,ids,descriptor,scene,case,event in entries():
            actual=model(scene);d=data(actual,state);mujoco.mj_fwdPosition(actual,d);source=contacts(actual,d);assert len(source)==len(descriptor)
            parameters=[scene.mass,scene.mu_l,scene.mu_r,scene.drive_difference];plane=construct(state[:17],descriptor,parameters);oracle=refs[(cohort,label,i,'actual_geometry')];control=refs[(cohort,label,i,'local_plane')]
            row=dict(cohort=cohort,label=label,index=i,case=case,event=event,source_contacts=source,arms={})
            for arm in ARMS:
                result=trial(plane,state,source,arm);steps+=1
                if arm=='unchanged_local_plane':
                    for n in FIELDS:np.testing.assert_allclose(result['next'][n],control[n],rtol=0,atol=TOL)
                q=np.array(result['next']['qpos']);v=np.array(result['next']['qvel']);dq=q-np.array(oracle['qpos']);dv=v-np.array(oracle['qvel'])
                result.update(active_q_error_rad=float(np.max(np.abs(dq[ids[:4]]))),full_q_error_rad=float(np.max(np.abs(dq))),full_v_error=float(np.max(np.abs(dv))))
                row['arms'][arm]=result
            rows.append(row);atomic_json(OUT/'intervention_progress.json',dict(status='running',states=len(rows),cpu_steps=steps))
            if i==143:print('COMPLETED',cohort,label,len(rows),flush=True)
        assert len(rows)==603 and steps==2412;verify()
        atomic_json(OUT/'intervention.json',dict(states=603,cpu_steps=steps,records=rows,summary=summarize(rows),intervention_contract_sha256=sha(OUT/'intervention_contract.json'),training_updates=0,new_gpu_steps=0))
        atomic_json(OUT/'intervention_progress.json',dict(status='complete',states=603,cpu_steps=steps));print(json.dumps(summarize(rows)),flush=True)
    except BaseException as e:
        atomic_json(OUT/'intervention_failure.json',dict(error=str(e),completed_states=len(rows),completed_cpu_steps=steps,silently_resumable=False));atomic_json(OUT/'intervention_partial.json',dict(records=rows));raise


def review():
    verify();r=json.loads((OUT/'intervention.json').read_text());assert r['states']==603 and r['cpu_steps']==2412 and r['intervention_contract_sha256']==sha(OUT/'intervention_contract.json')
    stored={(x['cohort'],x['label'],x['index']):x for x in r['records']};assert len(stored)==603;seen=set();steps=0;refs=reference()
    for cohort,label,i,state,ids,descriptor,scene,case,event in entries():
        key=(cohort,label,i);assert key not in seen;seen.add(key);old=stored[key];assert old['case']==case and old['event']==event
        actual=model(scene);d=data(actual,state);mujoco.mj_fwdPosition(actual,d);source=contacts(actual,d);assert source==old['source_contacts']
        plane=construct(state[:17],descriptor,[scene.mass,scene.mu_l,scene.mu_r,scene.drive_difference])
        for arm in ARMS:
            result=trial(plane,state,source,arm);steps+=1;previous=old['arms'][arm]
            for n in ['before','expected','after','time_s','ncon','nefc']:assert result[n]==previous[n]
            for n in FIELDS:np.testing.assert_allclose(result['next'][n],previous['next'][n],rtol=0,atol=TOL)
            oracle=refs[(*key,'actual_geometry')];dq=np.array(result['next']['qpos'])-oracle['qpos'];dv=np.array(result['next']['qvel'])-oracle['qvel']
            for field,value in [('active_q_error_rad',np.max(np.abs(dq[ids[:4]]))),('full_q_error_rad',np.max(np.abs(dq))),('full_v_error',np.max(np.abs(dv)))]:assert np.isclose(value,previous[field],rtol=0,atol=TOL)
            if arm=='unchanged_local_plane':
                for n in FIELDS:np.testing.assert_allclose(result['next'][n],refs[(*key,'local_plane')][n],rtol=0,atol=TOL)
        if i==143:print('VERIFIED',cohort,label,len(seen),flush=True)
    assert seen==set(stored) and steps==2412 and summarize(r['records'])==r['summary'];verify()
    atomic_json(OUT/'intervention_review.json',dict(verified=True,states=603,reconstructed_cpu_steps=steps,summary=r['summary'],intervention_sha256=sha(OUT/'intervention.json'),intervention_contract_sha256=sha(OUT/'intervention_contract.json'),
        limits='Finite current-contact counterfactual at actual parameters; modification persistence verified, same implementation numerically replayed. Legacy cohort is static fixture only. No deployment, measurability/continuous-domain/CPU-GPU or safety admission.'))
    print('PASS603 four-arm replay and persistence checks',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','review']);globals()[p.parse_args().command]()
