"""Independent reconstruction of static-equivalent targets before/after correction."""
from pathlib import Path
import json
import numpy as np
from diagnostic import check, sha, ROOT, OUT
from probe_height_angle_envelope import inverse
from state_estimation import leg_kinematics
import wheelleg_sim as sim


def run():
    check()
    r=json.loads((OUT/'verification.json').read_text())
    z=np.load(OUT/'reference_trace.npz',allow_pickle=False)['trace']
    valid=z[:,:,0]>0
    t=np.load(OUT/'trace.npz',allow_pickle=False)['trace']
    np.testing.assert_array_equal(valid,t[:,:,4]>0)
    np.testing.assert_allclose(z[:,:,12:14][valid]-z[:,:,10:12][valid],
        (z[:,:,16:18]/z[:,:,4,None])[valid],rtol=0,atol=1e-14)
    angles=np.stack((z[:,:,8:10],z[:,:,10:12],z[:,:,12:14]),axis=-1)
    length=z[:,:,1:3,None]
    x=sim.L5/2+length*np.sin(angles);y=-length*np.cos(angles)
    ac=np.hypot(x,y);ce=np.hypot(x-sim.L5,y)
    ca=(sim.L1**2+ac**2-sim.L2**2)/(2*sim.L1*ac)
    cb=(sim.L4**2+ce**2-sim.L3**2)/(2*sim.L4*ce)
    assert np.all(abs(ca[valid])<=1) and np.all(abs(cb[valid])<=1)
    qa=sim.PHI1_STAND-np.arctan2(y,x)+np.arccos(np.clip(ca,-1,1))
    qb=sim.PHI4_STAND-np.arctan2(y,x-sim.L5)-np.arccos(np.clip(cb,-1,1))
    excess=np.maximum(abs(qa),abs(qb))-z[:,:,3,None,None]
    locations=np.argwhere(valid)
    errors=[]
    # Deterministic uniform checks of the independent scalar IK and current-J
    # reconstruction. All-frame target/margin statistics remain vectorized.
    for index in np.linspace(0,len(locations)-1,401,dtype=int):
        step,w=locations[index];row=z[step,w]
        for side in range(2):
            leg,_,_,jac=leg_kinematics(row[18+2*side:20+2*side],np.zeros(2))
            angle=np.arctan2(leg[1],leg[0])+np.pi/2
            for start,column in ((22,10),(28,12)):
                hub=np.linalg.solve(jac,row[start+2*side:start+2*side+2])[1]
                expected=angle+(hub-row[7])/row[4]
                errors.append(abs(expected-row[column+side]))
            for stage in range(3):
                joints=inverse(row[1+side],angles[step,w,side,stage])
                assert joints is not None
                np.testing.assert_allclose(joints,[qa[step,w,side,stage],qb[step,w,side,stage]],rtol=0,atol=1e-13)
    assert max(errors)<1e-10
    rows=[]
    for w,selected in enumerate(r['registration']['plan_by_world']):
        if selected<0:continue
        a=excess[:,w].max(axis=1)
        # This display cutoff separates float32 rounding from material target
        # changes; it never changes a physical or admission gate.
        new=valid[:,w] & (a[:,1]<=1e-6) & (a[:,2]>1e-6)
        hit=np.flatnonzero(new)
        rows.append(dict(world=w,group=r['registration']['parameter_groups'][w],
            peak_static_joint_excess_rad=dict(zip(('projected_reference','accepted_nom_before','accepted_nom_after'),a[valid[:,w]].max(axis=0).tolist())),
            materially_new_infeasible_target_steps=int(new.sum()),
            first_command_interval_start_after_arrival_s=None if not len(hit) else int(hit[0])*.0005-r['episodes'][w]['arrival_s'],
            first_command_interval_end_after_arrival_s=None if not len(hit) else (int(hit[0])+1)*.0005-r['episodes'][w]['arrival_s']))
    assert all(row['materially_new_infeasible_target_steps']>0 for row in rows)
    candidate=OUT/'static_candidate_checked'
    trial=json.loads((candidate/'verification.json').read_text())
    decisions=json.loads((candidate/'projection_decisions.json').read_text())
    actual=np.load(candidate/'trace.npz',allow_pickle=False)
    actions=actual['extra'];data=actual['trace'];scored=data[:,:,4]>0
    assert scored.sum(axis=0).tolist()==[e['physical_steps'] for e in trial['episodes']]
    assert all(e['physical_steps']==e['physical_evidence_steps'] and e['physical_safety_passed'] for e in trial['episodes'])
    margins=[float(np.min(1.4-abs(data[scored[:,w],w,:4]))) for w in range(2)]
    np.testing.assert_array_equal(margins,trial['active_design_margin_rad'])
    assert min(margins)>=0 and all(not e['success'] and e['stop_distance_m']>.6 for e in trial['episodes'])
    assert np.max(abs(actions))<=1 and np.max(abs(np.diff(np.concatenate([np.zeros_like(actions[:1]),actions]),axis=0)).sum(axis=2))<=.1+1e-12
    assert not data[:,:,5][scored].any() and np.max(data[:,:,7][scored])<=1e-6
    changed=updates=0
    for frame in decisions:
        for row in frame['rows']:
            np.testing.assert_array_equal(actions[frame['iteration'],row['world']],row['selected'])
            assert row['solver_status']==0 and row['maximum_linear_excess']<=1e-9
            updates+=1;changed+=not np.array_equal(row['wanted'],row['selected'])
    assert updates==800 and changed>0
    for name,digest in trial['source_sha256'].items():assert sha(ROOT/name)==digest,name
    candidate_result=dict(actual_request_updates=updates,changed_from_guide=changed,
        active_design_margin_rad=margins,physical_pass=2,task_pass=0,
        stop_distance_m=[e['stop_distance_m'] for e in trial['episodes']],
        tail_speed_m_s=[e['tail_speed_m_s'] for e in trial['episodes']],promoted=False)
    result=dict(role='static_reference_order_counterexample_not_dynamic_causality_proof',rows=rows,
        scalar_reconstruction_max_angle_error_rad=max(errors),scalar_checked_frames=401,
        display_rounding_cutoff_rad=1e-6,static_projection_candidate=candidate_result,full_admission=False,
        limitations='Static-equivalent target violation exists even in successful nominal episodes. It does not prove this is the unique dynamic failure cause, or that enforcing this proxy solves parking. Original actual gates remain unchanged.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'reference_trace.npz',OUT/'trace.npz',OUT/'verification.json',candidate/'verification.json',candidate/'trace.npz',candidate/'projection_decisions.json')},
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target=OUT/'analysis.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('CHECKED eight reference-order counterexamples; candidate physical/design2/2, task0/2; original domains; no promotion.')


if __name__=='__main__':run()
