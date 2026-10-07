"""Full492 original-gate qualification;partial results cannot select a winner."""
import json
import subprocess
from score_reference_learning import load_rows,yaw_gate
from review_floor_broad_qualification import ordered,wrap
from review_nominal_mean_study import paired
import run_coordination_qualification as runner


def fixed_score(value,reference):
    return value is not None and reference is not None and yaw_gate([value],reference)


def complete():
    f=runner.OUT/'completion.json'
    if not f.exists():return False
    c=json.loads(f.read_text())
    return c.get('verified') is True and c.get('evaluations')==492 and c.get('training_updates')==0


def run():
    p,contract=runner.verify();out=runner.OUT;root=runner.ROOT
    if not complete():raise RuntimeError('Incomplete492:qualification and winner selection refused')
    c=json.loads((out/'completion.json').read_text());assert c['runner_contract_sha256']==runner.sha(out/'runner_contract.json')
    assert [(r['condition'],r['panel'],r['batch']) for r in c['records']]==[(j['condition'],j['panel'],j['batch']) for j in contract['jobs']]
    state=subprocess.check_output(['systemctl','--user','show','wheelleg-coordination-qualification-v1.service','-p','MainPID','-p','SubState','-p','Result','-p','ExecMainStatus'],text=True)
    assert 'MainPID=0' in state and 'SubState=exited' in state and 'Result=success' in state and 'ExecMainStatus=0' in state
    inputs={};data={};reference={};files=bytes_=0
    for condition in contract['conditions']:
        for panel in ('regular','controlled','legacy'):
            rows=[];indices=[]
            for job in c['records']:
                if (job['condition'],job['panel'])!=(condition,panel):continue
                f=out/job['path'];assert runner.sha(f)==job['sha256'];inputs[str(f.relative_to(root))]=job['sha256']
                d=load_rows(f,[p[panel][i] for i in job['indices']]);assert job['verified'];rows+=d['runs'];indices+=job['indices']
                assert runner.sha(f.parent/'geometry.json')==job['geometry_sha256']
                runner.coordination_logs(f.parent,condition)
                for row in d['runs']:
                    for kind in ('complete_trace','gyro_trace','role_trace','phase_trace','coordination_trace'):
                        z=f.parent/row[kind]['path'];assert runner.sha(z)==row[kind]['sha256'];files+=1;bytes_+=z.stat().st_size
            data[(condition,panel)]=wrap(ordered(rows,indices,len(p[panel])))
    assert files==2460
    for label in ('B0','B1-route'):
        for panel in ('regular','controlled','legacy'):
            rows=[];indices=[]
            for ref in p['references']['phase_result_refs']:
                if (ref['label'],ref['panel'])!=(label,panel):continue
                f=root/ref['path'];assert runner.sha(f)==ref['sha256'];inputs[ref['path']]=ref['sha256']
                rows+=load_rows(f,[p[panel][i] for i in ref['indices']])['runs'];indices+=ref['indices']
            reference[(label,panel)]=wrap(ordered(rows,indices,len(p[panel])))
    pairs={};gates={};cpu_checks={};scores={}
    for condition in ('Bomega','Cgamma'):
        checks={}
        for panel in ('regular','controlled','legacy'):
            candidate=data[(condition,panel)]
            for label,before in [('newB0',data[('B0',panel)]),('oldB0',reference[('B0',panel)]),('oldB1-route',reference[('B1-route',panel)])]:
                q=paired(before,candidate);pairs[f'{condition}/{panel}/{label}']=q;checks[f'preserve/{panel}/{label}']=q['preserved']
            values=candidate['summary']['mean_yaw_score_deg'];scores[f'{condition}/{panel}']=values
            checks[f'J/{panel}/newB0']=fixed_score(values,data[('B0',panel)]['summary']['mean_yaw_score_deg'])
            checks[f'J/{panel}/oldB0']=fixed_score(values,reference[('B0',panel)]['summary']['mean_yaw_score_deg'])
            checks[f'J/{panel}/oldB1-route']=fixed_score(values,reference[('B1-route',panel)]['summary']['mean_yaw_score_deg'])
            if condition=='Cgamma':
                q=paired(data[('Bomega',panel)],candidate);pairs[f'Cgamma/{panel}/Bomega']=q
                checks[f'mechanism_preserve/{panel}']=q['preserved']
                checks[f'mechanism_J/{panel}']=fixed_score(values,data[('Bomega',panel)]['summary']['mean_yaw_score_deg'])
        checks['controlled_atleast34']=data[(condition,'controlled')]['summary']['success_count']>=34
        ref=p['references']['CPU_legacy'];f=root/ref['path'];assert runner.sha(f)==ref['sha256'];inputs[ref['path']]=ref['sha256'];cpu=json.loads(f.read_text())['runs'];fail=[]
        for row,old in zip(data[(condition,'legacy')]['runs'],cpu):
            assert row['seed']==old['name'];flags=[]
            if not row['success']:flags.append('fulltask')
            if row['velocity_rmse']>old['velocity_rmse']*1.05+.005+1e-12:flags.append('velocity')
            for i,axis in enumerate(('roll','pitch')):
                if row['peak_deg'][i]>old['peak_deg'][i]+.1+1e-12:flags.append(axis)
            if flags:fail.append(dict(case=row['seed'],flags=flags))
        checks['CPUlegacy']=not fail;cpu_checks[condition]=dict(passed=not fail,failures=fail)
        gates[condition]=dict(checks=checks,passed=all(checks.values()))
    counts={f'{a}/{b}':dict(success=d['summary']['success_count'],total=len(d['runs']),physical=d['physical'],design=d['design'],J=d['summary']['mean_yaw_score_deg']) for (a,b),d in data.items()}
    runner.write(out/'qualification_review.json',dict(verified=True,counts=counts,gates=gates,pairs=pairs,CPUlegacy=cpu_checks,
        raw_trajectory_files=files,raw_trajectory_bytes=bytes_,first_episode_physics_steps=c['first_episode_physics_steps'],evaluation_queue_wall_s=c['evaluation_queue_wall_s'],terminal_unit=state,
        input_sha256=inputs,source_sha256=runner.sha(__file__),completion_sha256=runner.sha(out/'completion.json'),runner_contract_sha256=runner.sha(out/'runner_contract.json'),
        new_training_or_evaluations=0,formal5_admitted=False,entire_goal_complete=False,
        limits='All492 analytic development/regression qualification,not independent generalization/trainingseeds/newalgorithm. Original literal allpanel J thresholds retained. Gate pass only permits contribution/learning-necessity planning;failure closes fixedcandidate without pole/gain/threshold/budget rescue. Causal prefix and requestimplementation evidence separate.'))
    print('PASS delivery492',files,bytes_,'gates',{k:v['passed'] for k,v in gates.items()},'counts',counts,flush=True)


if __name__=='__main__':run()
