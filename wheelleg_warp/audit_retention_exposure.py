"""Complete-contact exposure at paired common-time prefixes; preserves original task gates."""
import json
import numpy as np
from run_reflection_retention import OUT,inputs
from analyze_complete_contact import moments,integrate,first,unit,I
from complete_contact_recorder import CONTACT_COL
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run():
    unit();p=inputs();assert not (OUT/'exposure_audit.json').exists()
    review=json.loads((OUT/'review.json').read_text());assert review['verified'] and review['uniform_reflection_expansion_closed']
    assert review['completion_sha256']==sha(OUT/'completion.json')
    entries={e['condition']:e for e in json.loads((OUT/'completion.json').read_text())['records']}
    data={};hashes={};steps_total=contacts_total=0
    for job in p['conditions']:
        directory=OUT/job['label'];row=json.loads((directory/'result.json').read_text())['runs'][0]
        assert sha(directory/'result.json')==review['raw_sha256'][str((directory/'result.json').relative_to(ROOT))]
        file=directory/row['complete_trace']['path'];assert sha(file)==review['raw_sha256'][str(file.relative_to(ROOT))];hashes[str(file.relative_to(ROOT))]=sha(file)
        with np.load(file,allow_pickle=False) as z:
            assert z['contact_columns'].tolist()==CONTACT_COL;t=z['trace'];meta=z['meta'];contact=z['contacts'];step=z['contact_steps'].astype(int)
        assert len(t)==row['physical_steps'];steps_total+=len(t);contacts_total+=len(contact)
        gf=directory/'geometry.json';assert sha(gf)==entries[job['label']]['checked']['geometry_sha256'];hashes[str(gf.relative_to(ROOT))]=sha(gf)
        geometry=json.loads(gf.read_text());assert len(geometry['boxes'][0])==1
        box=geometry['boxes'][0][0];wheel=np.asarray(geometry['wheel_geom_ids']);body=np.asarray(geometry['geom_body_ids'])
        a,b=contact[:,2].astype(int),contact[:,3].astype(int);wa,wb=np.isin(a,wheel),np.isin(b,wheel)
        external=(wa&(body[b]==0))|(wb&(body[a]==0));assert not np.any(wa&wb&external)
        full,normal,rest,vertical=moments(contact,meta[step-1,3:6],np.where(wb,1.,-1.))
        target=external&((a==box['geom'])|(b==box['geom']))
        def reduce(values,mask):return np.bincount(step[mask]-1,weights=values[mask],minlength=len(t))
        signals=dict(full=reduce(full,external),normal=reduce(normal,external),tangential_and_contact_torque=reduce(rest,external),
            target_full=reduce(full,target),target_normal=reduce(normal,target),target_load=reduce(np.maximum(contact[:,23],0),target),target_upward=reduce(np.maximum(vertical,0),target))
        np.testing.assert_allclose(signals['full'],signals['normal']+signals['tangential_and_contact_torque'],rtol=1e-10,atol=1e-10)
        assert not row['scenario']['relative_attitude']
        time=t[:,I['pre_s']];post=t[:,I['post_s']]
        attitude=np.rad2deg(t[:,[I[n] for n in ('roll','pitch','yaw')]])
        crossing=first(post,np.max(abs(attitude),axis=1)>5)
        contact_time=first(time,signals['target_load']>1e-6)
        side='left' if row['scenario']['height_l'] else 'right'
        local=(t[:,[I[f'pre_{side}_{k}'] for k in 'xyz']]-box['position'])@np.asarray(box['rotation'])
        data[job['label']]=dict(row=row,time=time,post=post,t=t,signals=signals,first_attitude5_s=crossing,first_target_contact_s=contact_time,
            central=abs(local[:,0])<=box['size'][0]/2,centre_inside=(abs(local[:,0])<=box['size'][0]+1e-7)&(abs(local[:,1])<=box['size'][1]+1e-7))
    records=[]
    for pair in p['pairs']:
        jobs=[j for j in p['conditions'] if j['pair']==pair['index']];left,right=(data[j['label']] for j in jobs)
        cutoff=right['first_attitude5_s'];assert cutoff is not None and left['first_attitude5_s'] is None
        compared=[]
        for job,d in zip(jobs,(left,right)):
            windows=dict(whole=np.ones(len(d['time']),bool),common_prefix_through_first_reflection_failure=d['time']<cutoff)
            summaries={}
            for name,mask in windows.items():
                signal=d['signals'];summaries[name]=dict(samples=int(mask.sum()),
                    target_normal_impulse_Ns=float(signal['target_load'][mask].sum()*.0005),upward_normal_impulse_Ns=float(signal['target_upward'][mask].sum()*.0005),
                    target_positive_duration_s=float((mask&(signal['target_load']>1e-6)).sum()*.0005),
                    target_positive_central_duration_s=float((mask&d['central']&(signal['target_upward']>1e-6)).sum()*.0005),
                    yaw_moments={key:integrate(signal[key],mask) for key in ('full','normal','tangential_and_contact_torque','target_full','target_normal')})
            k=int(np.searchsorted(d['post'],cutoff));assert k<len(d['t'])
            compared.append(dict(condition=job['label'],first_target_contact_s=d['first_target_contact_s'],first_attitude5_s=d['first_attitude5_s'],
                attitude_at_common_cutoff_deg=np.rad2deg(d['t'][k,7:10]).tolist(),body_y_at_common_cutoff_m=float(d['t'][k,5]),
                windows=summaries))
        a,b=[r['windows']['common_prefix_through_first_reflection_failure'] for r in compared]
        records.append(dict(pair=pair['index'],case=pair['case']['seed'],model=pair['model'],common_cutoff_s=cutoff,conditions=compared,
            reflection_minus_original_prefix_target_normal_Ns=b['target_normal_impulse_Ns']-a['target_normal_impulse_Ns'],
            reflection_minus_original_prefix_upward_Ns=b['upward_normal_impulse_Ns']-a['upward_normal_impulse_Ns']))
    assert steps_total==review['first_episode_world_steps'] and len(records)==8
    atomic_json(OUT/'exposure_audit.json',dict(round=353,verified=True,trajectories=16,first_episode_world_steps=steps_total,raw_contact_rows=contacts_total,pairs=records,input_sha256=hashes,
        review_sha256=sha(OUT/'review.json'),auditor_sha256=sha(__file__),moment_helper_sha256=sha(ROOT/'wheelleg_warp/analyze_complete_contact.py'),new_physics_steps=0,new_model_forward_rows=0,new_learning_samples=0,
        uniform_reflection_expansion_closed=True,formal5_admitted=False,
        timing='Allsolver contacts read. Force signed onrobot (geom1 positive/geom0 negative),world force=local@frame;momentaboutsolver-preCOM. Pairedprefix usesSAMEelapsedtime endingatfirstreflectionpost-attitudefailure,includingprecedingsolverstep.',
        limits='Pairedpolicy intervention changespath/otheractions/contact simultaneously. Contactimpulse/moment differences areobserved mediators,notunique cause oryawacceleration proof. Tangential includesdriveforces. Positivevertical force doesnotcertify centredloadedtraversal. No retroactive gate change ornewpopulation rate.',
        next='354 synthesize whether existingtask comparisons conflate disturbanceexposure withrejection;no additionalreflection runs orPPO.355direction/cleanup remains due.'))
    print('PASS35316fullcontact traces/8common-time prefixes',contacts_total,'contactrows;0physics/learning',flush=True)


if __name__=='__main__':run()
