"""Solver-time external wheel-contact yaw moments; observational, no rollout."""
import json

import numpy as np
import complete_contact_recorder as rec
from score_reference_learning import load_rows

I = {name:i for i,name in enumerate(rec.old.COL)}


def moments(contact, com, sign):
    frame = contact[:,9:18].reshape(-1,3,3)
    full_force = np.einsum('ni,nij->nj',contact[:,23:26],frame)*sign[:,None]
    full_torque = np.einsum('ni,nij->nj',contact[:,26:29],frame)*sign[:,None]
    normal_force = contact[:,23,None]*frame[:,0,:]*sign[:,None]
    arm = contact[:,6:9]-com
    full = np.cross(arm,full_force)[:,2]+full_torque[:,2]
    normal = np.cross(arm,normal_force)[:,2]
    return full, normal, full-normal, normal_force[:,2]


def first(time, mask):
    indices = np.flatnonzero(mask)
    return float(time[indices[0]]) if len(indices) else None


def integrate(values, mask):
    return dict(signed_Nms=float(values[mask].sum()*.0005),
                absolute_Nms=float(np.abs(values[mask]).sum()*.0005),
                peak_abs_Nm=float(np.max(np.abs(values[mask]))) if mask.any() else None)


def unit():
    c=np.zeros((1,29));c[:,6:9]=[1,0,0];c[:,9:18]=np.eye(3).reshape(1,9)
    c[:,23:29]=[3,2,0,0,0,4]
    full,normal,rest,vertical=moments(c,np.zeros((1,3)),np.ones(1))
    np.testing.assert_array_equal([full[0],normal[0],rest[0],vertical[0]],[6,0,6,0])
    neg=moments(c,np.zeros((1,3)),-np.ones(1))
    np.testing.assert_array_equal(neg[0],-full)
    c[:,9:18]=np.array([[0,1,0],[-1,0,0],[0,0,1]]).reshape(1,9)
    np.testing.assert_array_equal(moments(c,np.zeros((1,3)),np.ones(1))[0],[7])
    assert first(np.array([0,.0005]),np.array([False,True]))==.0005
    assert first(np.array([0]),np.array([False])) is None


def run():
    unit()
    out=rec.OUT;completion=json.loads((out/'completion.json').read_text())
    review=json.loads((out/'round172_data_review.json').read_text())
    assert completion['verified'] and review['verified'] and completion['completed_evaluations']==40
    contract=json.loads((out/'source_contract.json').read_text())
    assert all(rec.sha(rec.ROOT/name)==value for name,value in contract['source_sha256'].items())
    results=[];inputs={}
    for job in completion['records']:
        file=out/job['path'];assert rec.sha(file)==job['sha256']
        inputs[str(file.relative_to(out))]=rec.sha(file)
        data=load_rows(file,json.loads((out/'proposal.json').read_text())['cases'])
        gf=file.parent/'geometry.json';assert rec.sha(gf)==job['geometry_sha256'];g=json.loads(gf.read_text())
        inputs[str(gf.relative_to(out))]=rec.sha(gf)
        wheel=np.asarray(g['wheel_geom_ids']);body=np.asarray(g['geom_body_ids'])
        for w,row in enumerate(data['runs']):
            tf=file.parent/row['complete_trace']['path'];assert rec.sha(tf)==row['complete_trace']['sha256']
            inputs[str(tf.relative_to(out))]=rec.sha(tf)
            with np.load(tf,allow_pickle=False) as z:
                assert z['contact_columns'].tolist()==rec.CONTACT_COL
                t,meta,c,steps=z['trace'],z['meta'],z['contacts'],z['contact_steps'].astype(int)
            assert len(t)==row['physical_steps'] and np.isfinite(c).all()
            a,b=c[:,2].astype(int),c[:,3].astype(int)
            wa,wb=np.isin(a,wheel),np.isin(b,wheel)
            external=(wa & (body[b]==0)) | (wb & (body[a]==0))
            assert not np.any(wa & wb & external)
            sign=np.where(wb,1.,-1.)
            full,normal,rest,vertical=moments(c,meta[steps-1,3:6],sign)
            totals=[np.bincount(steps[external]-1,weights=x[external],minlength=len(t)) for x in (full,normal,rest)]
            np.testing.assert_allclose(totals[0],totals[1]+totals[2],atol=1e-10,rtol=1e-10)
            box=g['boxes'][w][0];assert len(g['boxes'][w])==1
            target=(a==box['geom']) | (b==box['geom'])
            target &= external
            load=np.bincount(steps[target]-1,weights=np.maximum(c[target,23],0),minlength=len(t))
            upward=np.bincount(steps[target]-1,weights=np.maximum(vertical[target],0),minlength=len(t))
            side='left' if row['scenario']['height_l'] else 'right'
            p=(t[:,[I[f'pre_{side}_{k}'] for k in 'xyz']]-box['position'])@np.asarray(box['rotation'])
            span=np.abs(p[:,0])<=box['size'][0]+1e-7
            central=np.abs(p[:,0])<=box['size'][0]/2
            time=t[:,I['pre_s']];yaw_time=t[:,I['post_s']]
            contact_time=first(time,load>1e-6)
            yaw5=first(yaw_time,np.abs(t[:,I['yaw']])>np.deg2rad(5))
            attitude5=first(yaw_time,np.max(np.abs(t[:,[I['roll'],I['pitch'],I['yaw']]]),axis=1)>np.deg2rad(5))
            clip=np.abs(t[:,I['roll_offset_requested']]-t[:,I['roll_offset_selected']])>1e-9
            windows=dict(whole_episode=np.ones(len(t),bool),target_centre_span=span)
            if contact_time is not None and yaw5 is not None and yaw5>contact_time:
                windows['first_target_contact_to_first_yaw5']=(time>=contact_time)&(time<yaw5)
            summary={name:{kind:integrate(values,mask) for kind,values in zip(('full','normal_only','friction_and_contact_torque'),totals)} for name,mask in windows.items()}
            results.append(dict(controller=job['label'],case=row['seed'],height_m=row['target_leg_m'],success=row['success'],original_peak_deg=row['peak_deg'],
                first_positive_target_contact_s=contact_time,first_yaw5_s=yaw5,first_attitude5_s=attitude5,
                yaw5_after_target_contact=(yaw5>contact_time) if yaw5 is not None and contact_time is not None else None,
                reference_clip_observed=bool(clip.any()),first_reference_clip_s=first(time,clip),
                target_span_steps=int(span.sum()),central_steps=int(central.sum()),central_positive_upward_normal_steps=int((central & (upward>1e-6)).sum()),
                target_centre_inside_entire_sampled_span=bool(span.any() and np.all(np.abs(p[span,1])<=box['size'][1]+1e-7)),
                moment_windows=summary,all_contacts=len(c),wheel_static_contacts=int(external.sum())))
    assert len(results)==40
    failures=[r for r in results if not r['success']]
    yaw_failures=[r for r in failures if r['first_yaw5_s'] is not None]
    counts=dict(original_failures=len(failures),yaw_failures=len(yaw_failures),yaw5_after_target_contact=sum(r['yaw5_after_target_contact'] is True for r in yaw_failures),
                yaw_failures_without_reference_clip=sum(not r['reference_clip_observed'] for r in yaw_failures))
    rec.atomic_json(out/'complete_contact_analysis.json',dict(verified=True,records=40,unique_development_cases=20,results=results,counts=counts,
        source_sha256=rec.sha(__file__),input_sha256=inputs,completion_sha256=rec.sha(out/'completion.json'),data_review_sha256=rec.sha(out/'round172_data_review.json'),new_evaluations=0,training_updates=0,
        formula='tau_z=((point-COM) cross signed(local_force @ frame) + signed(local_torque @ frame))_z; normal-only excludes all local contact torque. geom1 positive/geom0 negative. Sum external wheel-to-static contacts by solver-pre timestep,dt0.0005.',
        limits='External wheel-contact moment decomposition and timing are observational, not yaw-acceleration or causal-effect proof. Tangential force includes commanded traction. Upward normal is not top-face/full-load/full-passage certificate.20 paired development cases are not40 independent cases. Original task scores unchanged.'))
    print('PASS40 full-contact moment decomposition',counts,flush=True)
    for label in ('B0','B1-route'):
        for h in (.115,.16,.24,.30,.38):
            selected=[r for r in results if r['controller']==label and r['height_m']==h]
            print(label,h,[(r['case'],r['success'],r['first_positive_target_contact_s'],r['first_yaw5_s'],
                {k:round(v['signed_Nms'],6) for k,v in r['moment_windows']['target_centre_span'].items()}) for r in selected],flush=True)


if __name__=='__main__':
    run()
