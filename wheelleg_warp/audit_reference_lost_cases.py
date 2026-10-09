"""Saved80 reference episodes and six matched lost-case witnesses; no rollouts."""
import json
import numpy as np
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/nonzero_reference_mechanism_v1'


def run():
    assert not (OUT/'lost_case_audit.json').exists()
    c=json.loads((OUT/'completion.json').read_text());r=json.loads((OUT/'review.json').read_text())
    assert r['verified'] and not r['fixed_reference_utility_gate_passed']
    assert r['completion_sha256']==sha(OUT/'completion.json')
    rows={}
    for entry in c['records']:
        if entry['arm'] not in ('joint_yaw','joint_center','joint_lower'):
            continue
        file=OUT/entry['path'];assert sha(file)==entry['sha256']
        for row in json.loads(file.read_text())['runs']:
            rows[entry['arm'],row['seed']]=(file.parent,row)
    episodes=[];witnesses=[]
    for arm in ('joint_center','joint_lower'):
        lost=r['pairs'][arm]['joint_yaw']['lost']
        for key,(directory,row) in rows.items():
            if key[0]!=arm:continue
            reference=directory/row['joint_reference_trace']['path'];assert sha(reference)==row['joint_reference_trace']['sha256']
            with np.load(reference) as z:
                t=z['trace'];role=z['role']
                h=t[:,2];room=np.maximum(0,np.minimum(h-t[:,32],t[:,33]-h))
                d0=np.clip(role[:,6],-room,room)
                D=np.minimum(.035,np.minimum((t[:,33]-t[:,32])/2,np.minimum(h+.02-t[:,32],t[:,33]-h+.02)))
                ad=t[:,23]
                predicted=d0+ad*np.where(ad>=0,D-d0,D+d0)
                np.testing.assert_allclose(predicted,t[:,4],rtol=0,atol=1e-12)
                aligned=(d0*ad>0)
                assert np.all(abs(predicted[aligned])>=abs(d0[aligned])-1e-12)
                moving=t[:,31]!=0
                episodes.append(dict(arm=arm,seed=row['seed'],success=row['success'],physical_steps=len(t),
                    same_sign_additive_steps=int(aligned.sum()),same_sign_additive_moving_steps=int((aligned&moving).sum()),
                    max_half_difference_increment_m=float(abs(t[:,4]-d0).max()),
                    max_abs_mean_shift_m=float(abs(t[:,3]-h).max()),
                    room_original_geometric_binding_steps=int((abs(role[:,6])>room).sum()),
                    reference_sha256=sha(reference)))
                if row['seed'] not in lost:continue
                full=directory/row['complete_trace']['path'];assert sha(full)==row['complete_trace']['sha256']
                with np.load(full) as z:
                    dense=z['trace'];cols={str(n):i for i,n in enumerate(z['columns'])}
                    crossings=np.flatnonzero((abs(dense[:,[cols[n] for n in ('roll','pitch','yaw')]])>np.deg2rad(5.)).any(axis=1))
                    assert len(crossings);i=int(crossings[0])
                    attitude=dense[i,[cols[n] for n in ('roll','pitch','yaw')]]
                    axes=[n for n,v in zip(('roll','pitch','yaw'),attitude) if abs(v)>np.deg2rad(5.)]
                    assert axes==['yaw']
                    normal=dense[i,[cols['left_normal_N'],cols['right_normal_N']]]
                    omega=dense[i,[cols['pre_wheel_speed_left'],cols['pre_wheel_speed_right']]]
                    command=dense[i,[cols['command_left'],cols['command_right']]]
                    bounds=dense[i,[cols['pre_command_bound_left'],cols['pre_command_bound_right']]]
                    witness=dict(arm=arm,seed=row['seed'],height=row['scenario']['stand_height_m'],
                        first_exceeded_axes=axes,first_step=i+1,attitude_deg=np.rad2deg(attitude).tolist(),
                        moving_command=float(t[i,31]),nominal_mean=float(h[i]),tracking_mean=float(t[i,3]),
                        baseline_half_difference_m=float(d0[i]),selected_half_difference_m=float(t[i,4]),
                        filtered_difference_action=float(ad[i]),same_sign_additive=bool(aligned[i]),
                        original_mean_room_m=float(room[i]),baseline_requested_offset_m=float(role[i,6]),
                        pre_normal_N=normal.tolist(),pre_wheel_speed_rad_s=omega.tolist(),
                        wheel_command_Nm=command.tolist(),wheel_command_bounds_Nm=bounds.tolist(),
                        public_reference_sha256=sha(reference),dense_sha256=sha(full))
                matched_dir,matched=rows['joint_yaw',row['seed']];assert matched['success'] and matched['scenario']==row['scenario']
                mf=matched_dir/matched['complete_trace']['path'];assert sha(mf)==matched['complete_trace']['sha256']
                with np.load(mf) as z:
                    md=z['trace'];mc={str(n):j for j,n in enumerate(z['columns'])};assert i<len(md)
                    witness['matched_yaw_at_same_step_deg']=float(np.rad2deg(md[i,mc['yaw']]))
                    witness['matched_peak_deg']=matched['peak_deg']
                    witness['matched_dense_sha256']=sha(mf)
                witnesses.append(witness)
    assert len(episodes)==80 and len(witnesses)==6
    atomic_json(OUT/'lost_case_audit.json',dict(verified=True,round=308,episodes=episodes,witnesses=witnesses,
        review_sha256=sha(OUT/'review.json'),auditor_sha256=sha(__file__),
        new_simulation=0,new_controller_queries=0,new_training_samples=0,
        interpretation='All six newly lost cases firstcrossyaw atmiddle heights .24/.3,notendpointheight failures. Same-sign normalized difference requests addtoexistingbaseline offset,so canamplify roll compensation evenwhere oldworkspace clipping doesnotbind. Actualcontact/omega snapshots arediagnostic only,not actorinputs orunique causalproof.',
        fixed_feedback_remains_closed=True,formal_PPO_admitted=False,
        next='309 assess whether actionsemantic double compensation versus necessaryworkspace isdistinguishable frompriorfeedback designs usingtheseexistingrecords andalgebra; no gainfit orphysicalrescue.310deepreview/cleanup.'))
    print('PASS30880reference episodes/sixmatchedfirstyaw witnesses;0physics',flush=True)


if __name__=='__main__':run()
