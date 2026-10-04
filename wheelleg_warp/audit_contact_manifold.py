"""Offline pre-state contact representation audit; no time integration/control."""
import json
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
from probe_holdout_factors import inputs,verify_contract,OUT as FACTORS
from probe_contact_uncertainty import construct
from native.terrain import model
from dashboard.live_env import atomic_json

OUT=FACTORS.parent/'contact_manifold_audit_v1'
TOL=1e-12  # Diagnostic equality tolerance only; no control/prediction gate change.


def snapshot(m,state,ids):
    d=mujoco.MjData(m);d.qpos[:]=state[:17];d.qvel[:]=state[17:33];d.ctrl[:]=state[50:56];d.qacc_warmstart[:]=0.;d.time=0.
    q=d.qpos.copy();v=d.qvel.copy();mujoco.mj_forward(m,d)
    np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v)
    wheels={m.geom('wheel_collide_'+side).id:side for side in ['L','R']};contacts=[]
    for i in range(d.ncon):
        c=d.contact[i];a,b=map(int,c.geom);side=None;normal=None;support=None;descriptor=None
        if a in wheels and m.geom_bodyid[b]==0:side=wheels[a];normal=-c.frame[:3];support=m.geom(b).name
        elif b in wheels and m.geom_bodyid[a]==0:side=wheels[b];normal=c.frame[:3];support=m.geom(a).name
        if support and support.startswith('current_support_'):descriptor=int(support.removeprefix('current_support_'))
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        contacts.append(dict(index=i,geom_names=[m.geom(a).name,m.geom(b).name],side=side,support=support,descriptor_index=descriptor,
            normal_toward_wheel=None if normal is None else normal.tolist(),pos=c.pos.tolist(),signed_gap_m=float(c.dist),frame=c.frame.tolist(),dim=int(c.dim),exclude=int(c.exclude),efc_address=int(c.efc_address),includemargin=float(c.includemargin),
            friction=c.friction.tolist(),solref=c.solref.tolist(),solreffriction=c.solreffriction.tolist(),solimp=c.solimp.tolist(),local_contact_force6=force.tolist()))
    options={n:int(getattr(m.opt,n)) for n in ['integrator','solver','cone','iterations','noslip_iterations','disableflags','enableflags']}
    options.update(timestep=float(m.opt.timestep),tolerance=float(m.opt.tolerance),meaninertia=float(m.stat.meaninertia))
    return dict(contacts=contacts,active_qacc=d.qacc[ids[4:8]].tolist(),options=options,nefc=int(d.nefc))


def audit(label,i,state,ids,old,scene):
    parameters=[scene.mass,scene.mu_l,scene.mu_r,scene.drive_difference]
    actual=snapshot(model(scene),state,ids);plane=snapshot(construct(state[:17],old['contacts'],parameters),state,ids)
    wheel=[c for c in actual['contacts'] if c['side'] is not None];assert len(wheel)==len(old['contacts']);pairs=[]
    for j,(a,descriptor) in enumerate(zip(wheel,old['contacts'])):
        assert a['side']==descriptor['side'];np.testing.assert_allclose(a['normal_toward_wheel'],descriptor['normal_toward_wheel'],rtol=0,atol=TOL)
        assert abs(a['signed_gap_m']-descriptor['signed_gap_m'])<=TOL
        matches=[b for b in plane['contacts'] if b['descriptor_index']==j and b['side']==a['side']]
        pair=dict(descriptor_index=j,side=a['side'],actual_contact_index=a['index'],plane_contact_indices=[b['index'] for b in matches],unique_match=len(matches)==1)
        if len(matches)==1:
            b=matches[0];delta=np.array(b['pos'])-a['pos'];normal=np.array(a['normal_toward_wheel'])
            pair.update(position_delta_m=delta.tolist(),position_distance_m=float(np.linalg.norm(delta)),tangential_position_distance_m=float(np.linalg.norm(delta-normal*np.dot(normal,delta))),
                normal_distance=float(np.linalg.norm(np.array(b['normal_toward_wheel'])-normal)),gap_difference_m=float(b['signed_gap_m']-a['signed_gap_m']),
                dim_changed=a['dim']!=b['dim'],constraint_active_changed=(a['efc_address']>=0)!=(b['efc_address']>=0),
                parameter_differences={name:float(np.max(np.abs(np.array(a[name])-b[name]))) for name in ['friction','solref','solreffriction','solimp']},
                contact_force_norm_difference_N=float(np.linalg.norm(b['local_contact_force6'][:3])-np.linalg.norm(a['local_contact_force6'][:3])))
        pairs.append(pair)
    return dict(label=label,index=i,case=old['case'],event=old['event'],actual=actual,plane=plane,pairs=pairs,
        active_qacc_delta=(np.array(plane['active_qacc'])-actual['active_qacc']).tolist())


def summary(rows):
    groups={}
    for event in [-1,0,1,2]:
        selected=[r for r in rows if event==-1 or r['event']==event];pairs=[p for r in selected for p in r['pairs']];matched=[p for p in pairs if p['unique_match']]
        groups[str(event)]=dict(states=len(selected),actual_wheel_contacts=len(pairs),matched_contacts=len(matched),unmatched_contacts=len(pairs)-len(matched),
            nonwheel_actual_contacts=sum(c['side'] is None for r in selected for c in r['actual']['contacts']),
            max_position_distance_m=max((p['position_distance_m'] for p in matched),default=0.),max_tangential_position_distance_m=max((p['tangential_position_distance_m'] for p in matched),default=0.),
            max_normal_distance=max((p['normal_distance'] for p in matched),default=0.),max_gap_difference_m=max((abs(p['gap_difference_m']) for p in matched),default=0.),
            dim_changes=sum(p['dim_changed'] for p in matched),active_changes=sum(p['constraint_active_changed'] for p in matched),
            parameter_mismatches={k:sum(p['parameter_differences'][k]>TOL for p in matched) for k in ['friction','solref','solreffriction','solimp']},
            max_active_qacc_delta_rad_s2=max((max(abs(v) for v in r['active_qacc_delta']) for r in selected),default=0.))
    return groups


def register():
    prior=verify_contract();review=json.loads((FACTORS/'review.json').read_text());assert review['verified'] and review['states']==576 and not OUT.exists();OUT.mkdir()
    sources={**prior['source_sha256'],'wheelleg_warp/audit_contact_manifold.py':sha(__file__)}
    atomic_json(OUT/'registration.json',dict(version='contact-manifold-audit-v1',states=576,cpu_forwards=1152,new_physics_steps=0,new_gpu_steps=0,training_updates=0,source_sha256=sources,factor_review_sha256=sha(FACTORS/'review.json'),diagnostic_tolerance=TOL,
        selection='All576 existing states, no failure selection. Compare actual CPU geometry versus unchanged current-support-plane model at actual parameters, same q/v/known command/warm0/time0.',
        matching='Explicit current_support_i maps to original descriptor order and same wheel. Record missing/multiple matches instead of nearest-neighbor selection.',
        outputs='Contact positions/frame/normal/gap, dimension/active status/effective friction/solref/solimp, solved local force and active joint acceleration. mj_forward must leave q/v identical.',
        privileges='Full scene, actual parameters, forces and contact positions are offline oracle observations; not granted to actor/supervisor. No modification or promotion of failed predictor.',
        limits='Representation comparison is descriptive, not a contact-position causal intervention or a CPU-GPU contact audit. Equal fields at one pose do not imply equal future response or a robust bound.'))
    print('REGISTERED 576 states,1152 forward calls,0 integrated steps',flush=True)


def contract():
    p=json.loads((OUT/'registration.json').read_text());assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items()) and sha(FACTORS/'review.json')==p['factor_review_sha256'];return p


def run():
    p=contract();assert not (OUT/'completion.json').exists();rows=[];batch=[];ledger=[]
    for values in inputs():
        r=audit(*values);rows.append(r);batch.append(r)
        if r['index']==143:
            path=OUT/f'{r["label"]}_contacts.json';atomic_json(path,dict(records=batch));ledger.append(dict(label=r['label'],path=path.name,sha256=sha(path),states=144));batch=[]
            atomic_json(OUT/'progress.json',dict(status='running',states=len(rows)));print('COMPLETED',r['label'],144,flush=True)
    assert len(rows)==576 and not batch;contract()
    atomic_json(OUT/'completion.json',dict(states=576,cpu_forwards=1152,new_physics_steps=0,training_updates=0,records=ledger,summary=summary(rows),registration_sha256=sha(OUT/'registration.json')))
    atomic_json(OUT/'progress.json',dict(status='complete',states=576));print(json.dumps(summary(rows)['-1']),flush=True)


def review():
    p=contract();c=json.loads((OUT/'completion.json').read_text());assert c['states']==p['states']==576 and c['cpu_forwards']==1152 and c['new_physics_steps']==c['training_updates']==0 and c['registration_sha256']==sha(OUT/'registration.json')
    stored={}
    for job in c['records']:
        assert sha(OUT/job['path'])==job['sha256'];data=json.loads((OUT/job['path']).read_text())['records'];assert len(data)==job['states']==144
        for r in data:
            key=(r['label'],r['index']);assert key not in stored;stored[key]=r
    seen=set();rows=[]
    for values in inputs():
        key=values[:2];assert key not in seen;seen.add(key);r=audit(*values);assert r==stored[key];rows.append(r)
        if key[1]==143:print('VERIFIED',key[0],144,flush=True)
    assert seen==set(stored) and len(rows)==576 and summary(rows)==c['summary'];contract()
    atomic_json(OUT/'review.json',dict(verified=True,states=576,reconstructed_cpu_forwards=1152,new_physics_steps=0,summary=c['summary'],completion_sha256=sha(OUT/'completion.json'),source_sha256=sha(__file__),
        limits='Full forward/manifold payload rebuilt; q/v identity verified. Descriptive audit only; no unique causal attribution, new input admission, controller safety or continuous-domain certificate.'))
    print('PASS',json.dumps(c['summary']['-1']),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run','review']);globals()[p.parse_args().command]()
