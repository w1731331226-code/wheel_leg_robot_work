"""All-state offline support/parameter-envelope/warmstart response diagnosis."""
from pathlib import Path
import json
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
from probe_contact_uncertainty import construct,step
from native.terrain import HeightTerrainScenario,model
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004'
DATA=BASE/'contact_state_holdout_v2'
PRIOR=BASE/'contact_state_prediction_v1'
OUT=BASE/'contact_holdout_factors_v1'


def inputs():
    registration=json.loads((DATA/'registration.json').read_text())
    completion=json.loads((DATA/'collection_completion.json').read_text())
    prediction=json.loads((PRIOR/'completion.json').read_text())
    for job,old in zip(completion['records'],prediction['records']):
        label=job['label'];assert old['label']==label and sha(PRIOR/old['path'])==old['sha256']
        path=DATA/'collection'/f'{label}_states.npz';assert sha(path)==job['states_sha256']
        rows=json.loads((PRIOR/old['path']).read_text())['records']
        with np.load(path,allow_pickle=False) as z:
            assert len(z['state'])==len(rows)==144
            for i,(state,row) in enumerate(zip(z['state'],rows)):
                world=int(z['world'][i]);assert row['world']==world and row['index']==i and row['event']==int(z['event'][i])
                yield label,i,state,z['ids'],row,HeightTerrainScenario(**registration['cases'][world]['scenario'])


def summarize(records):
    summary={}
    for event in [-1,0,1,2]:
        rows=[r for r in records if event==-1 or r['event']==event]
        if not rows:continue
        summary[str(event)]=dict(states=len(rows),contact_point_raw_corner_misses=sum(not r['contact_point_inside_raw'] for r in rows),contact_point_reserved_corner_misses=sum(not r['contact_point_inside_reserved'] for r in rows),
            max_contact_point_excess_rad=max(r['contact_point_excess_rad'] for r in rows),max_support_representation_delta_rad=max(max(abs(x) for x in r['support_delta']) for r in rows),
            max_warm_delta_rad=max(max(abs(x) for x in r['warm_delta']) for r in rows),max_gpu_warm_cpu_residual_rad=max(max(abs(x) for x in r['backend_residual']) for r in rows))
    return summary


def register():
    assert not OUT.exists();review=json.loads((PRIOR/'review.json').read_text());assert review['verified'] and review['states']==576
    p=json.loads((PRIOR/'registration.json').read_text());sources={**p['source_sha256'],'wheelleg_warp/probe_holdout_factors.py':sha(__file__)}
    assert all(sha(ROOT/n)==v for n,v in sources.items());OUT.mkdir()
    atomic_json(OUT/'registration.json',dict(version='holdout-response-factors-v1',states=576,cpu_contact_point_steps=576,cpu_actual_warm_steps=576,training_updates=0,new_gpu_steps=0,
        selection='All576 verified saved states; no failure filtering or new episodes.',source_sha256=sources,prior_review_sha256=sha(PRIOR/'review.json'),prior_completion_sha256=sha(PRIOR/'completion.json'),collection_completion_sha256=sha(DATA/'collection_completion.json'),
        domain=p['bounds'],reserve_rad=p['numerical_reserve_rad'],
        branches='CPU current-contact model with actual mass/mu/drive and warm0; CPU actual geometry/parameters with saved warm. Reuse previously verified actual-geometry CPU warm0 and observed GPU responses.',
        decomposition='GPU-contact_point = (CPU_geometry_warm0-contact_point)+(CPU_geometry_savedwarm-CPU_geometry_warm0)+(GPU-CPU_geometry_savedwarm). Report signed vectors and maxima, not a percentage causal attribution.',
        privileges='Actual parameters, full actual scene and saved simulator warmstate are offline diagnostic truth only, never a predictor/controller input. Raw and original-reserved corner inclusion of actual-parameter contact point tests finite corner coverage. No enlargement/refit/gate reopening.',
        limits='Not a full factorial for all interactions; support delta includes contact representation and constitutive differences, residual includes cross-backend/representation differences. Warm-only intervention is controlled. This diagnostic cannot certify continuous-domain bounds, safety or method advantage.',old_gate_or_final_used=False))
    print('REGISTERED all576;1152 CPU steps, no learning',flush=True)


def verify_contract():
    p=json.loads((OUT/'registration.json').read_text())
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    for directory,key,name in [(PRIOR,'prior_review_sha256','review.json'),(PRIOR,'prior_completion_sha256','completion.json'),(DATA,'collection_completion_sha256','collection_completion.json')]:assert sha(directory/name)==p[key]
    assert p['states']==576 and p['training_updates']==p['new_gpu_steps']==0 and not p['old_gate_or_final_used']
    return p


def run():
    p=verify_contract();assert not (OUT/'progress.json').exists();records=[];batch=[];ledger=[]
    for label,i,state,ids,previous,scene in inputs():
        parameters=[scene.mass,scene.mu_l,scene.mu_r,scene.drive_difference];bounds=np.array(p['domain']);assert np.all(np.array(parameters)>=bounds[:,0]) and np.all(np.array(parameters)<=bounds[:,1])
        contact=step(construct(state[:17],previous['contacts'],parameters),state,ids)
        m=model(scene);d=mujoco.MjData(m);d.qpos[:]=state[:17];d.qvel[:]=state[17:33];d.ctrl[:]=state[50:56];d.qacc_warmstart[:]=state[33:49];d.time=0.
        mujoco.mj_step(m,d);warm=d.qpos[ids[:4]].copy();zero=np.array(previous['oracle_q']);gpu=np.array(previous['gpu_q']);lower=np.array(previous['lower_q']);upper=np.array(previous['upper_q']);reserve=p['reserve_rad']
        support=zero-contact;warm_delta=warm-zero;residual=gpu-warm
        np.testing.assert_allclose(support+warm_delta+residual,gpu-contact,rtol=0,atol=1e-15)
        row=dict(label=label,index=i,case=previous['case'],event=previous['event'],parameter_vector=parameters,contact_point_q=contact.tolist(),actual_warm_q=warm.tolist(),support_delta=support.tolist(),warm_delta=warm_delta.tolist(),backend_residual=residual.tolist(),
            contact_point_inside_raw=bool(np.all(contact>=lower)&np.all(contact<=upper)),contact_point_inside_reserved=bool(np.all(contact>=lower-reserve)&np.all(contact<=upper+reserve)),
            contact_point_excess_rad=float(max(0.,float(np.max(lower-reserve-contact)),float(np.max(contact-upper-reserve)))))
        records.append(row);batch.append(row);atomic_json(OUT/'progress.json',dict(status='running',states=len(records),pending_label=label))
        if i==143:
            path=OUT/f'{label}_factors.json';atomic_json(path,dict(records=batch));ledger.append(dict(label=label,path=path.name,sha256=sha(path),states=144));batch=[]
            atomic_json(OUT/'completed_jobs.json',dict(states=len(records),records=ledger));print('COMPLETED',label,144,flush=True)
    assert len(records)==576 and not batch;verify_contract()
    atomic_json(OUT/'completion.json',dict(states=576,cpu_contact_point_steps=576,cpu_actual_warm_steps=576,training_updates=0,records=ledger,summary=summarize(records),registration_sha256=sha(OUT/'registration.json')))
    atomic_json(OUT/'progress.json',dict(status='complete',states=576));print(json.dumps(summarize(records)['-1']),flush=True)


def review():
    p=verify_contract();done=json.loads((OUT/'completion.json').read_text());assert done['registration_sha256']==sha(OUT/'registration.json') and done['states']==576 and done['training_updates']==0
    assert done['cpu_contact_point_steps']==done['cpu_actual_warm_steps']==576 and done['records']==json.loads((OUT/'completed_jobs.json').read_text())['records']
    stored={}
    for job in done['records']:
        assert sha(OUT/job['path'])==job['sha256'] and job['states']==144
        rows=json.loads((OUT/job['path']).read_text())['records'];assert len(rows)==144
        for r in rows:
            key=(r['label'],r['index']);assert key not in stored;stored[key]=r
    records=[];seen=set()
    for label,i,s,ids,old,scene in inputs():
        key=(label,i);assert key not in seen;seen.add(key);r=stored[key]
        assert r['case']==old['case'] and r['event']==old['event'] and r['parameter_vector']==[scene.mass,scene.mu_l,scene.mu_r,scene.drive_difference]
        # Rebuild both new branches independently from the original saved arrays.
        outputs=[]
        for m,warm in [(construct(s[:17],old['contacts'],r['parameter_vector']),np.zeros(16)),(model(scene),s[33:49])]:
            d=mujoco.MjData(m);d.qpos[:]=s[:17];d.qvel[:]=s[17:33];d.ctrl[:]=s[50:56];d.qacc_warmstart[:]=warm;d.time=0.;mujoco.mj_step(m,d);outputs.append(d.qpos[ids[:4]].copy())
        for values,name in zip(outputs,['contact_point_q','actual_warm_q']):np.testing.assert_allclose(values,r[name],rtol=0,atol=1e-12)
        contact,warm=outputs;zero=np.array(old['oracle_q']);gpu=np.array(old['gpu_q']);lower=np.array(old['lower_q']);upper=np.array(old['upper_q']);reserve=p['reserve_rad']
        for a,name in [(zero-contact,'support_delta'),(warm-zero,'warm_delta'),(gpu-warm,'backend_residual')]:np.testing.assert_allclose(a,r[name],rtol=0,atol=1e-12)
        np.testing.assert_allclose(np.array(r['support_delta'])+r['warm_delta']+r['backend_residual'],gpu-contact,rtol=0,atol=1e-15)
        assert r['contact_point_inside_raw']==bool(np.all(contact>=lower)&np.all(contact<=upper)) and r['contact_point_inside_reserved']==bool(np.all(contact>=lower-reserve)&np.all(contact<=upper+reserve))
        assert np.isclose(r['contact_point_excess_rad'],max(0.,float(np.max(lower-reserve-contact)),float(np.max(contact-upper-reserve))),rtol=0,atol=1e-15)
        records.append(r)
        if i==143:print('VERIFIED',label,144,flush=True)
    assert seen==set(stored) and len(seen)==576 and summarize(records)==done['summary'];verify_contract()
    atomic_json(OUT/'review.json',dict(verified=True,states=576,reconstructed_cpu_steps=1152,summary=done['summary'],registration_sha256=sha(OUT/'registration.json'),completion_sha256=sha(OUT/'completion.json'),source_sha256=sha(__file__),training_updates=0,
        limits='All-state finite controlled diagnostic only. Actual parameters/warm/full geometry never granted to controller. Decomposition identity is arithmetic, not proof of unique physical cause; prior failed envelope stays rejected.'))
    print('PASS',done['summary']['-1'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run','review']);globals()[p.parse_args().command]()
