"""Original full gates;baseline delivery can be audited while the worker runs."""
import json
import sys
from pathlib import Path
import numpy as np
import execution_input_study as runner
from review_nominal_mean_study import paired
from review_floor_broad_qualification import ordered,wrap
from review_nom_yaw_filter import full_flags
from score_reference_learning import load_rows,yaw_gate

OUT,ROOT,sha,write=runner.OUT,runner.ROOT,runner.sha,runner.write
KINDS=('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace','actor_trace')


def gate(value,reference):
    return value is not None and reference is not None and bool(yaw_gate([value],reference))


def complete():
    f=OUT/'study_completion.json'
    if not f.exists():raise RuntimeError('Incomplete6runs/1312:benefit conclusions refused')
    d=json.loads(f.read_text())
    if not d.get('verified') or d.get('policy_samples')!=1200000 or d.get('evaluations')!=1312:
        raise RuntimeError('Incomplete6runs/1312:benefit conclusions refused')
    return d


def collect(p,records,arm,seed,inputs,raw):
    panels={}
    for panel in ('regular','controlled','legacy'):
        selected=[r for r in records if (r['arm'],r['seed'],r['panel'])==(arm,seed,panel)]
        wanted=[j for j in p['eval_jobs'] if j['panel']==panel]
        assert [r['batch'] for r in selected]==[j['batch'] for j in wanted]
        rows=[];indices=[]
        for job,expected in zip(selected,wanted):
            assert job['indices']==expected['indices']
            f=OUT/job['path'];assert sha(f)==job['sha256'];inputs[str(f.relative_to(ROOT))]=job['sha256']
            d=load_rows(f,expected['cases']);rows+=d['runs'];indices+=job['indices']
            for r in d['runs']:
                for kind in KINDS:
                    path=f.parent/r[kind]['path'];assert sha(path)==r[kind]['sha256']
                    raw['files']+=1;raw['bytes']+=path.stat().st_size
                with np.load(f.parent/r['actor_trace']['path'],allow_pickle=False) as z:a=z['trace']
                assert a.shape==(int(np.ceil(r['physical_steps']/40)),968) and np.isfinite(a).all()
                if seed is None:
                    np.testing.assert_array_equal(a[:,:481],a[:,481:962])
                    if arm=='B0':np.testing.assert_array_equal(a[:,962:],0)
                if arm=='H0':
                    from execution_history_env import COMMAND_INDICES
                    np.testing.assert_array_equal(a[:,COMMAND_INDICES],0)
                    np.testing.assert_array_equal(a[:,481+COMMAND_INDICES],0)
        panels[panel]=wrap(ordered(rows,indices,len(p[panel])))
    return panels


def old_refs(p,inputs):
    result={}
    for label in ('B0','B1-route'):
        for panel in ('regular','controlled','legacy'):
            rows=[];indices=[]
            for ref in p['references']['phase_result_refs']:
                if (ref['label'],ref['panel'])!=(label,panel):continue
                f=ROOT/ref['path'];assert sha(f)==ref['sha256'];inputs[ref['path']]=ref['sha256']
                rows+=load_rows(f,[p[panel][i] for i in ref['indices']])['runs'];indices+=ref['indices']
            result[(label,panel)]=wrap(ordered(rows,indices,len(p[panel])))
    return result


def cpu(p,candidate):
    ref=p['references']['CPU_legacy'];f=ROOT/ref['path'];assert sha(f)==ref['sha256'];old=json.loads(f.read_text())['runs'];fail=[]
    for r,b in zip(candidate['runs'],old):
        assert r['seed']==b['name'];flags=[]
        if not r['success']:flags.append('fulltask')
        if r['velocity_rmse']>b['velocity_rmse']*1.05+.005+1e-12:flags.append('velocity')
        for i,name in enumerate(('roll','pitch')):
            if r['peak_deg'][i]>b['peak_deg'][i]+.1+1e-12:flags.append(name)
        if flags:fail.append(dict(case=r['seed'],flags=flags))
    return dict(passed=not fail,failures=fail)


def baseline():
    p,c=runner.verify();ledger=json.loads((OUT/'study_evaluation_ledger.json').read_text());inputs={};raw=dict(files=0,bytes=0)
    data=collect(p,ledger['records'],'B0',None,inputs,raw);old=old_refs(p,inputs);changes=[]
    for panel,d in data.items():
        for a,b in zip(old[('B0',panel)]['runs'],d['runs']):
            fa,fb=full_flags(a),full_flags(b);diff={k:dict(old=fa[k],new=fb[k]) for k in fa if fa[k]!=fb[k]}
            if diff:changes.append(dict(panel=panel,case=a['seed'],changes=diff))
    counts={k:dict(total=len(v['runs']),success=v['summary']['success_count'],physical=v['physical'],design=v['design'],J=v['summary']['mean_yaw_score_deg']) for k,v in data.items()}
    write(OUT/'round236_B0_delivery.json',dict(verified=True,counts=counts,old_B0_flag_changes=changes,
        CPUlegacy=cpu(p,data['legacy']),raw=raw,input_sha256=inputs,contract_sha256=sha(OUT/'study_contract.json'),
        source_sha256=sha(__file__),new_training_or_evaluations=0,limits='All164B0 delivery/calibration only;no H0/H1 benefit conclusion. Raw remains pending terminal archive.'))
    print('PASS236 fullB0',counts,'raw',raw,'oldflagchanges',len(changes),flush=True)


def full():
    finished=complete();p,c=runner.verify()
    assert finished['contract_sha256']==sha(OUT/'study_contract.json')
    expected=[dict(arm=a,seed=None) for a in p['new_classical']]+p['jobs'];assert finished['completed']==expected
    inputs={};raw=dict(files=0,bytes=0);old=old_refs(p,inputs);data={};comparisons={};checks={};cpus={}
    for job in expected:
        data[(job['arm'],job['seed'])]=collect(p,finished['records'],job['arm'],job['seed'],inputs,raw)
        if job['seed'] is not None:
            directory=OUT/'study_runs'/job['arm']/str(job['seed']);v=json.loads((directory/'training_verification.json').read_text())
            assert v['verified'] and v['policy_samples']==200000 and v['epochs']==400 and v['Adam']==8000 and not v['engineering_warmstart']
            inputs[str((directory/'training_verification.json').relative_to(ROOT))]=sha(directory/'training_verification.json')
    assert raw['files']==1312*6
    for seed in p['seeds']:
        a,b=[json.loads((OUT/'study_runs'/arm/str(seed)/'initialization.json').read_text()) for arm in ('H0','H1')]
        assert a['initial_weights']==b['initial_weights'] and a['initial_world']==b['initial_world']
        row={};checks[str(seed)]=row
        for panel in ('regular','controlled','legacy'):
            candidate=data[('H1',seed)][panel];base=data[('H0',seed)][panel]
            references=[('new'+label,data[(label,None)][panel]) for label in p['new_classical']]+[('old'+label,old[(label,panel)]) for label in p['new_classical']]
            for label,ref in references:
                pair=paired(ref,candidate);comparisons[f'{seed}/{panel}/{label}']=pair
                row[f'preserve/{panel}/{label}']=pair['preserved']
                row[f'J/{panel}/{label}']=gate(candidate['summary']['mean_yaw_score_deg'],ref['summary']['mean_yaw_score_deg'])
            pair=paired(base,candidate);comparisons[f'{seed}/{panel}/H0']=pair;row[f'mechanism_preserve/{panel}']=pair['preserved']
            x,y=candidate['summary']['mean_yaw_score_deg'],base['summary']['mean_yaw_score_deg']
            row[f'mechanism_J_lower/{panel}']=x is not None and y is not None and x<y
        row['controlled_atleast34']=data[('H1',seed)]['controlled']['summary']['success_count']>=34
        cpus[str(seed)]=cpu(p,data[('H1',seed)]['legacy']);row['CPUlegacy']=cpus[str(seed)]['passed']
    means={}
    for panel in ('regular','controlled','legacy'):
        values={arm:[data[(arm,s)][panel]['summary']['mean_yaw_score_deg'] for s in p['seeds']] for arm in ('H0','H1')}
        means[panel]=dict(values=values,passed=all(x is not None for arr in values.values() for x in arr) and gate(np.mean(values['H1']),np.mean(values['H0'])))
    passed=all(v for row in checks.values() for v in row.values()) and all(r['passed'] for r in means.values())
    counts={f'{a}/{s}/{panel}':dict(success=v['summary']['success_count'],physical=v['physical'],design=v['design'],J=v['summary']['mean_yaw_score_deg']) for (a,s),panels in data.items() for panel,v in panels.items()}
    write(OUT/'study_review.json',dict(verified=True,passed=passed,checks=checks,mechanism_means=means,counts=counts,
        comparisons=comparisons,CPUlegacy=cpus,raw=raw,input_sha256=inputs,completion_sha256=sha(OUT/'study_completion.json'),
        source_sha256=sha(__file__),formal5_admitted=False,entire_goal_complete=False,
        limits='Full information-necessity study,genericPPO vs informationablation;not novelalgorithm/independentgeneralization or publicationguarantee. Full manuscript/stats/newformal5 still separate.'))
    print('FULL REVIEW',passed,counts,flush=True)


if __name__=='__main__':{'baseline':baseline,'full':full}[sys.argv[1]]()
