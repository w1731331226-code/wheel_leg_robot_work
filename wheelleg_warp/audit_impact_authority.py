"""Optimistic post-contact 50Hz action opportunity, not a dynamics counterfactual."""
import json
import numpy as np
from analyze_complete_contact import moments, I
from complete_contact_recorder import ROOT, OUT, sha, atomic_json


def next_decision(time):
    # Contact is computed after the action at the current boundary was selected.
    return (int(np.floor(time/.02+1e-8))+1)*.02


def common_envelope(time, decision):
    ticks=np.maximum(0,np.floor((time-decision)/.0005+1e-7).astype(int)+1)
    return .3*np.minimum(1,.01*ticks)


def unit():
    assert abs(next_decision(.04)-.06)<1e-12
    assert abs(next_decision(.041)-.06)<1e-12
    np.testing.assert_allclose(common_envelope(np.array([.0195,.02,.0205,.0695]),.02),[0,.003,.006,.3])


def run():
    unit()
    analysis=json.loads((OUT/'complete_contact_analysis.json').read_text())
    assert analysis['verified'] and analysis['source_sha256']==sha(ROOT/'wheelleg_warp/analyze_complete_contact.py')
    src=ROOT/'wheelleg_warp/native/controller.py';text=src.read_text()
    assert 'state[w,16+j]+wp.clamp(D(targets[w,j])-state[w,16+j],D(-.01),D(.01))' in text
    assert 'state[w,20]*yaw_cfg[3],state[w,21]*yaw_cfg[3]' in text
    assert 'state[w,8]=state[w,8]+(D(sensor[w,gyro+2])-state[w,8])*D(.025)' in text
    assert 'yaw_config=(.4,2.,.24,.3)' in (ROOT/'wheelleg_warp/native/environment.py').read_text()
    rows=[]
    for r in analysis['results']:
        file=OUT/'runs'/r['controller']/f'case_{r["case"]}.npz'
        assert sha(file)==analysis['input_sha256'][str(file.relative_to(OUT))]
        with np.load(file,allow_pickle=False) as z:
            t,c,meta,steps=z['trace'],z['contacts'],z['meta'],z['contact_steps'].astype(int)
        g=json.loads((file.parent/'geometry.json').read_text());wheel=np.asarray(g['wheel_geom_ids']);body=np.asarray(g['geom_body_ids'])
        a,b=c[:,2].astype(int),c[:,3].astype(int);wa,wb=np.isin(a,wheel),np.isin(b,wheel)
        ext=(wa & (body[b]==0)) | (wb & (body[a]==0))
        normal=moments(c,meta[steps-1,3:6],np.where(wb,1.,-1.))[1]
        normal=np.bincount(steps[ext]-1,weights=normal[ext],minlength=len(t))
        time=t[:,I['pre_s']];start=r['first_positive_target_contact_s'];assert start is not None
        decision=next_decision(start)
        window=(time>=start-1e-9)&(time<start+.1-1e-9)
        absolute=np.abs(normal)
        total=float(absolute[window].sum()*.0005)
        before=float(absolute[window & (time<decision-1e-9)].sum()*.0005)
        env=common_envelope(time,decision)
        # Integral of hypothetical commanded per-wheel common residual only.
        # Ignores differential competition, clipping and Nom interaction; not delta-speed/yaw.
        rows.append(dict(controller=r['controller'],case=r['case'],height_m=r['height_m'],success=r['success'],
            contact_s=start,next_possible_contact_triggered_policy_s=decision,wait_ms=(decision-start)*1000,
            fixed100ms_abs_normal_impulse_Nms=total,before_first_decision_Nms=before,
            fraction_before_first_decision=before/total if total>1e-12 else None,
            commanded_common_residual_per_wheel_integral_100ms_Nms=float(env[window].sum()*.0005),
            yaw_failure=r['first_yaw5_s'] is not None))
    low=[r for r in rows if r['height_m'] in (.115,.16)]
    fractions=[r['fraction_before_first_decision'] for r in low]
    result=dict(verified=True,rows=rows,records=40,low_height_records=16,
        low_height_fraction_before_first_decision=dict(min=min(fractions),max=max(fractions),median=float(np.median(fractions))),
        constants=dict(policy_period_s=.02,physics_period_s=.0005,request_slew_per_physics_step=.01,wheel_residual_scale_Nm=.3,full_scale_from_zero_after_substeps=100,elapsed_to_100th_command_s=.0495,Nom_yaw_gyro_filter_alpha=.025,Nom_yaw_gyro_filter_efold_s=-.0005/np.log(.975)),
        source_sha256=sha(__file__),controller_sha256=sha(src),analysis_sha256=sha(OUT/'complete_contact_analysis.json'),
        verdict='Reactive contact-triggered policy cannot prevent the moment portion occurring before its first available action. Common authority exists in virtual6, but merely exposing it is not a new method. Post-impact recovery/anticipatory policies are not ruled out.',
        limits='Optimistically grants immediate perfect contact detection at next50Hz boundary, unavailable ground-truth contact information in real39packet. Actual delay/proxy/diamond bounds/global lambda may reduce opportunity. Hypothetical command integral is not applied force, achievable speed reduction, causal benefit or infeasibility of all learned policies.100ms window is exploratory and cannot silently become primary gate.',
        new_physics_evaluations=0,new_training_updates=0)
    atomic_json(OUT/'impact_authority_audit.json',result)
    print('PASS authority/slew/timing checks;',result['low_height_fraction_before_first_decision'],flush=True)


if __name__=='__main__':run()
