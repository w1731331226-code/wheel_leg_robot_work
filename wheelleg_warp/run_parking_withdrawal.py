"""Fixed78 frozen-policy causal diagnostic; no training or implicit replay."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
import nominal_mean_main as main
import nominal_mean_evaluation as evaluation
import parking_withdrawal_probe as probe
from review_nom_yaw_filter import full_flags

ROOT,sha,write=main.ROOT,main.sha,main.write
OUT=main.OUT.parent/'parking_withdrawal_v1'


def jobs(p):
    return [dict(seed=m['seed'],condition=c,batch=b,indices=[i for i,_ in group],cases=[x for _,x in group])
            for m in p['models'] for c in p['conditions'] for b,group in enumerate(probe.phase.broad.batches(p['cases']))]


def admitted():
    main.verify();a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['scientific_evaluations_consumed']==a['training_updates']==0
    for path,key in [('parent_direction_review','parent_direction_review_sha256'),('parent_trainer_contract','parent_trainer_contract_sha256'),('source_unit','source_unit_sha256')]:
        assert sha(ROOT/a[path])==a[key]
    assert all(sha(ROOT/n)==s for n,s in a['new_source_sha256'].items())
    p=a['diagnostic'];assert p['total_first_episode_evaluation_budget']==78 and sum(len(j['cases']) for j in jobs(p))==78
    for m in p['models']:
        assert sha(ROOT/m['model'])==m['model_sha256'] and sha(ROOT/m['normalization'])==m['normalization_sha256']
    return p


def verify():
    p=admitted();c=json.loads((OUT/'runner_contract.json').read_text())
    assert c['source_sha256']==sha(__file__) and c['jobs']==jobs(p) and c['budget']==78
    assert all(sha(ROOT/n)==s for n,s in c['input_sha256'].items())
    return p


def freeze():
    assert not (OUT/'runner_contract.json').exists();p=admitted()
    files=[OUT/'source_admission.json',main.OUT/'round210_direction_review.json',main.OUT/'round211_parking_source_unit.json']
    files += [ROOT/'wheelleg_warp'/n for n in ('parking_withdrawal_probe.py','test_parking_withdrawal_probe.py','nominal_mean_evaluation.py','review_nom_yaw_filter.py')]
    write(OUT/'runner_contract.json',dict(source_sha256=sha(__file__),input_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},
        jobs=jobs(p),budget=78,updates=0,selection='All3 fixed200000 anchored models/all13 common cases/both conditions;no dropout or alternative checkpoint.',
        comparability='Record original replay full-flag differences versus old study,never discard/retry. Causal review uses the new matched original/withdrawal pair;exact pre-trigger prefix is required to interpret a case causally. Differences limit claims,not extra budget.',
        trigger='Private seen-nonzero-current-command AND exact currentcommand0,not startup/arrival/goal/contact truth.',
        scientific_gates='Original full flags/design/physical unchanged;casewise crossing/parking/request/filter and yaw/roll/velocity/stop retained. This diagnostic cannot revive the closed benefit or qualifyformal5.',
        worker='wheelleg-parking-withdrawal-v1.service;exclusive project-write.lock',interruption='Save all consumed prefixes/buffers/counters/model state hashes;no implicit restart or resume.'))
    verify();print('FROZEN78 diagnostic,0 consumed/0training',flush=True)


def parking_logs(directory,condition):
    rows=json.loads((directory/'result.json').read_text())['runs'];total=0
    for r in rows:
        f=directory/r['parking_trace']['path'];assert sha(f)==r['parking_trace']['sha256']
        with np.load(f,allow_pickle=False) as z:
            assert z['columns'].tolist()==probe.COL;data=z['trace'];probe.check_log(data,condition)
            assert len(data)==r['physical_steps']==r['parking_trace']['rows']
        with np.load(directory/r['phase_trace']['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(data[:,:3],z['trace'][:,:3])
        with np.load(directory/r['complete_trace']['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(data[:,:2],z['trace'][:,:2])
            np.testing.assert_array_equal(data[:,23:29],z['trace'][:,47:53])
        total+=f.stat().st_size
    return dict(parking_trace_files=len(rows),parking_trace_bytes=total)


def prior_comparison(seed,rows):
    v=json.loads((main.OUT/'runs/anchored'/str(seed)/'verification.json').read_text());old={};inputs={}
    for ref in v['evaluation_records']:
        if ref['panel'] not in ('regular','controlled'):continue
        f=main.OUT/ref['path'];assert sha(f)==ref['sha256'];inputs[str(f.relative_to(ROOT))]=ref['sha256']
        old.update({r['seed']:r for r in json.loads(f.read_text())['runs']})
    differences=[]
    for r in rows:
        o=old[r['seed']];assert r['scenario']==o['scenario']
        before,after=full_flags(o),full_flags(r)
        changed={k:dict(old=before[k],replay=after[k]) for k in before if before[k]!=after[k]}
        if changed:differences.append(dict(case=r['seed'],changed=changed))
    return dict(full_flag_differences=differences,old_input_sha256=inputs,all_old_labels_preserved=not differences,
                limits='13-world diagnostic batches differ from prior20-world panels;no cross-run bitwise claim. New matched pair reviewed separately.')


def run():
    torch.set_num_threads(1);p=verify();(OUT/'runs').mkdir(exist_ok=False)
    ledger=[];reserved=finished=0;raw=directory=current=model=None;start=time.perf_counter()
    try:
        for current in jobs(p):
            verify();raw=None;seed=current['seed'];condition=current['condition'];cases=current['cases']
            ref=next(m for m in p['models'] if m['seed']==seed)
            model=PPO.load(ROOT/ref['model'],device='cuda');main.training.check_agent(model,8000)
            directory=OUT/'runs'/str(seed)/condition/f'batch_{current["batch"]}';directory.mkdir(parents=True,exist_ok=False)
            reserved+=len(cases);write(OUT/'progress.json',dict(status='evaluating',current_job=current,reserved_evaluations=reserved,completed_evaluations=finished))
            factory=evaluation.phase.instrument;assert factory is probe.BASE_FACTORY
            def wrapped(group,mode,directory=None):
                nonlocal raw
                assert group==cases;raw=probe.instrument(group,mode,condition,directory)
                wait=raw.step_wait;batches=0
                def observed():
                    nonlocal batches
                    result=wait();batches+=1
                    if batches%25==0:write(OUT/'progress.json',dict(status='evaluating',current_job=current,reserved_evaluations=reserved,
                        completed_evaluations=finished,first_episodes_saved=len(raw._parking_frozen),first_physics_steps=int(raw._complete_buffers[2].numpy()[:,0].sum())))
                    return result
                raw.step_wait=observed;return raw
            evaluation.phase.instrument=wrapped
            try:result,checked=evaluation.evaluate(cases,'regular',model,ROOT/ref['normalization'],directory)
            finally:evaluation.phase.instrument=factory
            assert sha(ROOT/ref['model'])==ref['model_sha256'] and sha(ROOT/ref['normalization'])==ref['normalization_sha256']
            checked.update(parking_logs(directory,condition));finished+=len(result['runs'])
            comparison=prior_comparison(seed,result['runs']) if condition=='original' else None
            ledger.append(dict(seed=seed,condition=condition,batch=current['batch'],indices=current['indices'],path=str((directory/'result.json').relative_to(OUT)),
                sha256=sha(directory/'result.json'),original_replay_comparison=comparison,model_RMS_immutable=True,**checked))
            write(OUT/'completed_jobs.json',dict(completed_evaluations=finished,records=ledger));verify()
            print('COMPLETED',finished,seed,condition,'success',result['summary']['success_count'],'design',result['design'],flush=True)
        assert reserved==finished==78
        write(OUT/'completion.json',dict(verified=True,evaluations=78,training_updates=0,records=ledger,runner_contract_sha256=sha(OUT/'runner_contract.json'),
            evaluation_queue_wall_s=time.perf_counter()-start,first_episode_physics_steps=sum(r['physics_steps'] for r in ledger),
            scope='Existing13 development cases/3 frozen models,not fresh trained seeds/generalization. Paired causal interpretation still requires213 prefix/dynamic/fullflag review. Benefit remainsclosed.'))
        write(OUT/'progress.json',dict(status='complete',completed_evaluations=78,training_updates=0))
    except BaseException as error:
        partial=None
        if raw is not None:
            partial=evaluation.phase_runner.preserve(raw,directory,current['cases']);partial['parking']=probe.preserve(raw,directory)
        write(OUT/'interruption.json',dict(error=repr(error),reserved_evaluations=reserved,completed_evaluations=finished,current_job=current,records=ledger,partial=partial,
            model_state_hash=main.training.weight_digest(model) if model is not None else None,silently_resumable=False))
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);globals()[parser.parse_args().command]()
