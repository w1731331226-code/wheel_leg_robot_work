"""Single fixed492 analytic qualification;all original task gates retained."""
import json
import time
from pathlib import Path
import numpy as np
import coordination_probe as probe
import nominal_mean_main as main
import run_phase_support_qualification as prior
import probe_route_feedback as feedback
from review_nom_yaw_filter import full_flags

ROOT,sha,write=main.ROOT,main.sha,main.write
OUT=main.OUT.parent/'speed_yaw_coordination_v1'


def zero6(obs,candidate,arm):
    old,ref=ORIGINAL_ACTIONS(obs,candidate,arm)
    assert not old.any() and not ref.any()
    return np.zeros((len(obs),6),np.float32),ref


ORIGINAL_ACTIONS=feedback.actions


def admitted():
    p=main.verify();a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['scientific_evaluations_consumed']==a['training_updates']==0
    for name,key in [('parent_proposal','parent_proposal_sha256'),('source_unit','source_unit_sha256'),('parent_runtime_contract','parent_runtime_contract_sha256')]:assert sha(ROOT/a[name])==a[key]
    assert all(sha(ROOT/n)==s for n,s in a['source_sha256'].items())
    u=json.loads((ROOT/a['source_unit']).read_text());assert u['synthetic_terminal_repeat_autoreset_and_explicit_reset_checked'] and u['packet20ms_hold_and_raw_input_unchanged']
    return p,a


def jobs(p):
    return [dict(condition=c,panel=panel,batch=b,indices=[i for i,_ in group],cases=[x for _,x in group])
            for c in ('B0','Bomega','Cgamma') for panel in ('regular','controlled','legacy')
            for b,group in enumerate(probe.phase.broad.batches(p[panel]))]


def verify():
    p,a=admitted();c=json.loads((OUT/'runner_contract.json').read_text())
    assert c['source_sha256']==sha(__file__) and c['jobs']==jobs(p) and c['budget']==492
    assert all(sha(ROOT/n)==s for n,s in c['input_sha256'].items())
    return p,c


def freeze():
    assert not (OUT/'runner_contract.json').exists();p,a=admitted();j=jobs(p);assert sum(len(x['cases']) for x in j)==492
    files=[OUT/'source_admission.json',ROOT/a['source_unit'],ROOT/a['parent_proposal']]
    files += [ROOT/'wheelleg_warp'/n for n in ('speed_yaw_coordination.py','wheel_motion_proxy.py','route_pilot_env.py','probe_route_feedback.py')]
    refs={r['path']:r['sha256'] for r in p['references']['phase_result_refs']};cpu=p['references']['CPU_legacy'];refs[cpu['path']]=cpu['sha256']
    assert all(sha(ROOT/n)==s for n,s in refs.items())
    write(OUT/'runner_contract.json',dict(source_sha256=sha(__file__),input_sha256={**{str(f.relative_to(ROOT)):sha(f) for f in files},**refs},jobs=j,budget=492,updates=0,
        panels=dict(regular=96,controlled=40,legacy=28),conditions=['B0','Bomega','Cgamma'],classical=dict(candidate=dict(kp=0,kd=0,roll_gain=0),arm=0),
        selection='All164 registered development/regression cases/all3 conditions;no checkpoint/seed selection. No model or PPO updates.',
        gates='Every original fullflag/axis/physical/design gate retained;candidate must preserve strongB0/B1 successes without new component failure,all164 physical/design,CPUlegacy28 velocity1.05+.005/rollpitch+.1;controlled>=34/40 and paired allpanels J15percent+.05 versus B0/B1;Cgamma mechanism versus Bomega separately. No default baseline replacement on source/evalcompletion alone.',
        replay='B0 old-to-new fullflag differences recorded,not silently overwritten,retried or used to weaken reference. Actual newB0 paired candidate review separately;exact per-case prefix limits mechanistic claims.',
        worker='wheelleg-coordination-qualification-v1.service;exclusive project-write.lock',interruption='Full/gyro/role/phase/coordination prefixes and last buffers plus reserved/completed counters;no implicit resume/restart/budget increase.'))
    verify();print('FROZEN492 allpanel/3conditions,0 consumed/0learning',flush=True)


def original_map(p,panel):
    rows=[]
    for ref in p['references']['phase_result_refs']:
        if ref['label']=='B0' and ref['panel']==panel:
            f=ROOT/ref['path'];assert sha(f)==ref['sha256'];rows+=json.loads(f.read_text())['runs']
    return {r['seed']:r for r in rows}


def coordination_logs(directory,condition):
    rows=json.loads((directory/'result.json').read_text())['runs'];size=0
    for r in rows:
        f=directory/r['coordination_trace']['path'];assert sha(f)==r['coordination_trace']['sha256']
        with np.load(f,allow_pickle=False) as z:
            assert z['columns'].tolist()==probe.COL;data=z['trace'];probe.check_log(data,condition)
            assert len(data)==r['physical_steps']==r['coordination_trace']['rows']
        with np.load(directory/r['phase_trace']['path'],allow_pickle=False) as z:np.testing.assert_array_equal(data[:,:3],z['trace'][:,:3])
        size+=f.stat().st_size
    return dict(coordination_trace_files=len(rows),coordination_trace_bytes=size)


def run():
    p,c=verify();(OUT/'runs').mkdir(exist_ok=False);ledger=[];reserved=finished=0;raw=directory=current=None;start=time.perf_counter()
    try:
        for current in c['jobs']:
            verify();cases=current['cases'];condition=current['condition'];panel=current['panel'];raw=None
            directory=OUT/'runs'/condition/panel/f'batch_{current["batch"]}';directory.mkdir(parents=True,exist_ok=False)
            reserved+=len(cases);write(OUT/'progress.json',dict(status='evaluating',condition=condition,panel=panel,batch=current['batch'],reserved_evaluations=reserved,completed_evaluations=finished))
            factory=probe.phase.rec.old.evaluator.raw_env;actions=feedback.actions
            def wrapped(group,mode):
                nonlocal raw
                assert group==cases and mode=='virtual6';raw=probe.instrument(group,mode,condition,directory);wait=raw.step_wait;count=0
                def observed():
                    nonlocal count
                    result=wait();count+=1
                    if count%25==0:write(OUT/'progress.json',dict(status='evaluating',condition=condition,panel=panel,batch=current['batch'],reserved_evaluations=reserved,
                        completed_evaluations=finished,first_episodes_saved=len(raw._coord_frozen),first_physics_steps=int(raw._complete_buffers[2].numpy()[:,0].sum())))
                    return result
                raw.step_wait=observed;return raw
            probe.phase.rec.old.evaluator.raw_env=wrapped;feedback.actions=zero6
            try:result=probe.phase.rec.old.evaluator.evaluate(cases,'V6-route',classical=c['classical'])
            finally:probe.phase.rec.old.evaluator.raw_env=factory;feedback.actions=actions
            result=json.loads(json.dumps(result,allow_nan=False));write(directory/'result.json',result)
            checked=prior.prior.prior.check_files(directory,cases,{r['seed']:r for r in result['runs']});checked.pop('original_label_differences')
            assert prior.logs(directory,cases)==checked['physics_steps'];checked.update(coordination_logs(directory,condition));finished+=len(cases)
            old=original_map(p,panel);differences=[]
            if condition=='B0':
                for r in result['runs']:
                    assert r['scenario']==old[r['seed']]['scenario'];a,b=full_flags(old[r['seed']]),full_flags(r)
                    changes={k:dict(old=a[k],new=b[k]) for k in a if a[k]!=b[k]}
                    if changes:differences.append(dict(case=r['seed'],changes=changes))
            ledger.append(dict(condition=condition,panel=panel,batch=current['batch'],indices=current['indices'],path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),oldB0_flag_differences=differences,**checked))
            write(OUT/'completed_jobs.json',dict(completed_evaluations=finished,records=ledger));verify()
            print('COMPLETED',finished,condition,panel,'success',result['summary']['success_count'],'physical',result['physical'],'design',result['design'],flush=True)
        assert finished==reserved==492
        write(OUT/'completion.json',dict(verified=True,evaluations=492,training_updates=0,records=ledger,runner_contract_sha256=sha(OUT/'runner_contract.json'),evaluation_queue_wall_s=time.perf_counter()-start,
            first_episode_physics_steps=sum(r['physics_steps'] for r in ledger),scope='One deterministic analytic qualification on164 seen cases,not trainingseeds/independent generalization. Full gates/CPUlegacy/causal comparison still require review.'))
        write(OUT/'progress.json',dict(status='complete',completed_evaluations=492,training_updates=0))
    except BaseException as error:
        partial=None
        if raw is not None:partial=prior.preserve(raw,directory,current['cases']);partial['coordination']=probe.preserve(raw,directory)
        write(OUT/'interruption.json',dict(error=repr(error),reserved_evaluations=reserved,completed_evaluations=finished,current_job=current,records=ledger,partial=partial,silently_resumable=False));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);globals()[parser.parse_args().command]()
