"""Frozen current-contact/public-corner response check on all new event states."""
from pathlib import Path
import json
import numpy as np
from review_yaw_sector import ROOT,sha
from probe_contact_uncertainty import construct,step
from check_contact_response import extract_current
from native.terrain import HeightTerrainScenario,model
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004'
DATA=BASE/'contact_state_holdout_v2'
OUT=BASE/'contact_state_prediction_v1'


def metrics(records,reserve):
    groups={}
    for label in sorted({r['label'] for r in records})+['all']:
        for kind in ['scheduled','contact_creation','all']:
            selected=[r for r in records if (label=='all' or r['label']==label) and (kind=='all' or (r['event']==2)==(kind=='contact_creation'))]
            if not selected:continue
            groups[f'{label}/{kind}']=dict(states=len(selected),gpu_misses=sum(not r['gpu_inside'] for r in selected),oracle_misses=sum(not r['oracle_inside'] for r in selected),
                gpu_false_safe=sum(r['interval_safe'] and np.max(np.abs(r['gpu_q']))>1.4 for r in selected),oracle_false_safe=sum(r['interval_safe'] and np.max(np.abs(r['oracle_q']))>1.4 for r in selected),
                max_gpu_excess_rad=max(r['gpu_excess_rad'] for r in selected),max_oracle_excess_rad=max(r['oracle_excess_rad'] for r in selected),
                max_corner_width_rad=max(float(np.max(np.asarray(r['upper_q'])-r['lower_q'])) for r in selected))
    return groups


def freeze():
    assert not OUT.exists();OUT.mkdir()
    prior=json.loads((BASE/'contact_uncertainty_v1/registration.json').read_text());contract=json.loads((DATA/'collector_contract.json').read_text());review=json.loads((DATA/'collection_review.json').read_text())
    assert review['verified'] and review['snapshots']==576
    sources={**contract['source_sha256'],**{f'wheelleg_warp/{n}':sha(ROOT/'wheelleg_warp'/n) for n in ['predict_contact_holdout.py','review_contact_prediction.py','probe_contact_uncertainty.py','check_contact_response.py','native/models.py','native/terrain.py']}}
    assert all(sha(ROOT/n)==v for n,v in sources.items())
    p=dict(version='contact-state-prediction-v1',states=576,corners=prior['corners'],bounds=prior['bounds'],numerical_reserve_rad=prior['numerical_reserve_rad'],
        cpu_corner_steps=9216,cpu_actual_geometry_steps=576,training_updates=0,new_gpu_steps=0,source_sha256=sources,
        collection_review_sha256=sha(DATA/'collection_review.json'),collection_completion_sha256=sha(DATA/'collection_completion.json'),
        prior_parameter_registration_sha256=sha(BASE/'contact_uncertainty_v1/registration.json'),
        information='Ideal pre-state fullq/v plus known held command and wheel/world current normal/gap only. Actual scene used offline solely to emulate that measurement and validate actual geometry; actual parameters never select predictor corners. CPU models warm0. Descriptor extracted at saved pre-q, not post-q.',
        scores='Strict component inclusion in all16-corner min/max plus unchanged one float32 position ULP at1.4. Interval safe only if entire reserved interval lies in[-1.4,1.4]. Report every miss/excess and false-safe, scheduled/contact_creation separately.',
        limits='Finite diagnostic envelope, not a proven continuous-domain bound, deployable sensor model, or recursive controller. No refit, enlargement, new learning or old/final test use.',old_gate_or_final_used=False)
    atomic_json(OUT/'registration.json',p);print('FROZEN 576 states;9216 corner steps+576 offline oracle steps')


def run():
    p=json.loads((OUT/'registration.json').read_text());assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    assert sha(DATA/'collection_review.json')==p['collection_review_sha256'] and sha(DATA/'collection_completion.json')==p['collection_completion_sha256']
    reg=json.loads((DATA/'registration.json').read_text());done=json.loads((DATA/'collection_completion.json').read_text());reserve=p['numerical_reserve_rad'];records=[];ledger=[]
    assert not (OUT/'progress.json').exists()
    for job in done['records']:
        label=job['label'];sp=DATA/'collection'/f'{label}_states.npz';assert sha(sp)==job['states_sha256'];z=np.load(sp,allow_pickle=False);ids=z['ids'];batch=[]
        for i,state in enumerate(z['state']):
            world=int(z['world'][i]);event=int(z['event'][i]);scene=HeightTerrainScenario(**reg['cases'][world]['scenario'])
            contacts=extract_current(scene,state[:17])
            values=np.array([step(construct(state[:17],contacts,vector),state,ids) for vector in p['corners']]);lower=values.min(0);upper=values.max(0)
            oracle=step(model(scene),state,ids);gpu=state[68:85][ids[:4]]
            lo=lower-reserve;hi=upper+reserve
            excess=lambda q:float(max(0.,float(np.max(lo-q)),float(np.max(q-hi))))
            row=dict(label=label,index=i,world=world,case=reg['cases'][world]['seed'],event=event,time_s=float(state[49]),contacts=contacts,corner_q=values.tolist(),lower_q=lower.tolist(),upper_q=upper.tolist(),gpu_q=gpu.tolist(),oracle_q=oracle.tolist(),
                gpu_inside=bool(np.all(gpu>=lo)&np.all(gpu<=hi)),oracle_inside=bool(np.all(oracle>=lo)&np.all(oracle<=hi)),interval_safe=bool(np.all(lo>=-1.4)&np.all(hi<=1.4)),gpu_excess_rad=excess(gpu),oracle_excess_rad=excess(oracle))
            batch.append(row);records.append(row)
            atomic_json(OUT/'progress.json',dict(status='running',states=len(records),pending_job=label))
        path=OUT/f'{label}_prediction.json';atomic_json(path,dict(records=batch));ledger.append(dict(label=label,path=path.name,sha256=sha(path),states=len(batch)))
        atomic_json(OUT/'completed_jobs.json',dict(records=ledger,states=len(records)));print('COMPLETED',label,len(batch),flush=True)
    assert len(records)==p['states']==576 and all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    groups=metrics(records,reserve)
    atomic_json(OUT/'completion.json',dict(states=576,cpu_corner_steps=9216,cpu_actual_geometry_steps=576,training_updates=0,records=ledger,groups=groups,
        development_gate=groups['all/all']['gpu_misses']==groups['all/all']['oracle_misses']==0,registration_sha256=sha(OUT/'registration.json')))
    atomic_json(OUT/'progress.json',dict(status='complete',states=576));print(json.dumps(groups['all/all']),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);globals()[p.parse_args().command]()
