"""Saved19-trajectory timing and request-feedback witnesses; no dynamics or inference."""
import json
import pickle
import numpy as np
from run_frozen_reflection import OUT,inputs
from review_yaw_sector import ROOT,sha
from audit_fixed_force_failures import first
from audit_policy_reflection import reflect_raw
from dashboard.live_env import atomic_json


def phases(command):
    moving=command!=0;seen=np.maximum.accumulate(moving)
    return dict(standing=~seen,moving=moving,parking=seen & ~moving)


def run():
    sample=phases(np.array([0,0,-.1,-.5,0,0.]));assert [int(x.sum()) for x in sample.values()]==[2,2,2]
    p=inputs();assert not (OUT/'timing_audit.json').exists()
    review=json.loads((OUT/'review.json').read_text());assert review['verified'] and review['reproduction_gate_passed']
    c=json.loads((OUT/'completion.json').read_text());assert review['completion_sha256']==sha(OUT/'completion.json')
    records={};source={};total=0
    for entry in c['records']:
        directory=OUT/entry['condition'];f=directory/'result.json';assert sha(f)==entry['sha256'];row=json.loads(f.read_text())['runs'][0]
        arrays=[]
        for field in ('complete_trace','joint_guard_trace','actor_trace','parking_trace','gyro_trace'):
            file=directory/row[field]['path'];digest=sha(file);assert digest==review['raw_sha256'][str(file.relative_to(ROOT))]
            source[str(file.relative_to(ROOT))]=digest
            with np.load(file,allow_pickle=False) as z:arrays.append(z['trace'])
        t,g,actor,parking,gyro=arrays;n=len(t);total+=n;time=t[:,3];yaw=np.rad2deg(t[:,9]);command=t[:,43]
        masks=phases(command);assert sum(int(m.sum()) for m in masks.values())==n
        nominal=(g[:,6]-g[:,7])/2;residual=(g[:,24]-g[:,18]-g[:,25]+g[:,19])/2
        def point(i):
            if i is None:return None
            k=i//40
            return dict(step=i+1,pre_s=float(t[i,2]),post_s=float(time[i]),yaw_deg=float(yaw[i]),body_y_m=float(t[i,5]),
                command_m_s=float(command[i]),Nom_wheel_halfdiff_Nm=float(nominal[i]),residual_wheel_halfdiff_Nm=float(residual[i]),
                Nom_filtered_gyro_z_rad_s=float(gyro[i,4]),original_lambda=float(t[i,53]),guard_lambda=float(g[i,27]),
                policy_row=k,submitted6=actor[k,962:].tolist(),effective6=parking[i,11:17].tolist(),
                estimated_route_y_m=float(actor[k,389]),packet_yaw_deg=float(np.rad2deg(actor[k,353])))
        events=dict(first_motion=point(first(masks['moving'])),parking_start=point(first(masks['parking'])),yaw_peak=point(int(abs(yaw).argmax())),
            first_Nom_wheel_halfdiff_above_1e_minus6_Nm=point(first(abs(nominal)>1e-6)),first_residual_wheel_halfdiff_above_1e_minus6_Nm=point(first(abs(residual)>1e-6)))
        for threshold in (.01,.1,.5,1.,3.,5.):events[f'first_yaw_above_{threshold}_deg']=point(first(abs(yaw)>threshold))
        for sec in (1.,2.,4.,6.,7.2,9.):
            i=int(np.searchsorted(time,sec));events[f'at_{sec}_s']=point(i) if i<n else None
        segments={}
        for name,mask in masks.items():
            active=mask & (abs(nominal)>1e-6) & (abs(residual)>1e-6)
            segments[name]=dict(physical_steps=int(mask.sum()),yaw_peak_deg=float(abs(yaw[mask]).max()) if mask.any() else None,
                body_y_peak_m=float(abs(t[mask,5]).max()) if mask.any() else None,
                opposing_Nom_steps=int((active & (nominal*residual<0)).sum()),both_nonzero_steps=int(active.sum()),
                residual_nonzero_Nom_zero_steps=int((mask & (abs(nominal)<=1e-6) & (abs(residual)>1e-6)).sum()),
                wheel_residual_RMS_Nm=float(np.sqrt(np.mean(residual[mask]**2))) if mask.any() else None,
                original_lambda_limited_steps=int((mask & (t[:,53]<1-1e-9)).sum()))
        index=first(np.max(abs(actor[:,962:]),axis=1)>1e-6)
        records[entry['condition']]=dict(events=events,phases=segments,reset_submitted6=actor[0,962:].tolist(),
            first_submitted_above_1e_minus6_normalized=None if index is None else dict(policy_row=index,actor_time_s=index*.02,submitted6=actor[index,962:].tolist()),
            guard_nominal_changed=int((g[:,60]>1e-12).sum()),guard_residual_changed=int((g[:,27]<1-1e-9).sum()))
        if entry['condition'].endswith('_reflection'):
            label=entry['condition'].removesuffix('_reflection');model=next(m for m in p['models'] if m['label']==label)
            with (ROOT/model['normalization']).open('rb') as stream:norm=pickle.load(stream)
            raw=actor[:,:481];delta=abs(norm.normalize_obs(raw.copy())-norm.normalize_obs(reflect_raw(raw)))
            groups={'physical_packet':np.array([39*k+j for k in range(10) for j in range(26)]),
                'accepted_residual':np.array([39*k+j for k in range(10) for j in range(26,32)]),
                'filtered_request':np.array([39*k+j for k in range(10) for j in range(32,38)]),
                'route':np.array([39*k+38 for k in range(10)]),'total_command_history':np.array([390+8*k+j for k in range(9) for j in range(6)])}
            records[entry['condition']]['normalized_feature_asymmetry_at_fixed_times']={str(sec):{name:float(delta[min(int(sec/.02),len(actor)-1),indices].max()) for name,indices in groups.items()} for sec in (0.,.2,1.,2.,4.,6.)}
    assert total==review['first_episode_world_steps']==351340
    atomic_json(OUT/'timing_audit.json',dict(round=348,verified=True,evaluations=19,first_episode_world_steps=total,records=records,input_sha256=source,
        review_sha256=sha(OUT/'review.json'),auditor_sha256=sha(__file__),new_physics_steps=0,new_policy_forward_rows=0,new_learning_samples=0,formal5_admitted=False,
        measurement_scope='1e-6 thresholds identify numeric-sized requests/commandhalf-differences only;not controlgates. Phase partitions followactualcurrentcommand andseenmotion. Snapshotpose ispoststep,command/gyroprestep;packetdelayed. Normalizedfeaturedifferences arefrozenRMS units,notphysicaldistances orcausalfeatureimportance.',
        interpretation='Azero resetmean/algebraic equivariance doesnot keep futureinput histories symmetric orprove closed-loopstability. Nominalwheelcommand opposition ismeasured,not a ground-yaw work/energycertificate. No unique cause assigned tosolvernoise,deadzone,historyfeedback orlegchannels.',
        next='349 consolidate finite hypothesis and stopping decision using theseaccepted traces;no further sweeps/learning bydefault.350direction/cleanup andwhole-framework review.'))
    print('PASS348all19phase/timing witnesses,351340savedsteps,0physics/inference/learning',flush=True)


if __name__=='__main__':run()
