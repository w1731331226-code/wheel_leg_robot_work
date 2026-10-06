"""Full six-run qualification; refuses partial study and retains every original flag."""
import json
import numpy as np
import nominal_mean_main as main
from review_floor_broad_qualification import ordered, wrap
from review_nom_yaw_filter import full_flags, added_failures, unit
from score_reference_learning import load_rows, yaw_gate

OUT,ROOT,sha,write=main.OUT,main.ROOT,main.sha,main.write


def paired(before, after):
    assert len(before['runs'])==len(after['runs'])
    lost=[];gained=[];regressions=[];rows=[]
    for a,b in zip(before['runs'],after['runs']):
        assert a['seed']==b['seed'] and a['scenario']==b['scenario']
        fa,fb=full_flags(a),full_flags(b)
        assert a['success']==(not any(fa.values())) and b['success']==(not any(fb.values()))
        new=added_failures(fa,fb)
        if a['success'] and not b['success']:lost.append(a['seed'])
        if not a['success'] and b['success']:gained.append(a['seed'])
        if new:regressions.append(dict(case=a['seed'],flags=new))
        rows.append(dict(case=a['seed'],height_m=a['target_leg_m'],original_flags=fa,candidate_flags=fb,new_flags=new))
    clean=all(r['physical_safety_passed'] and r['design_joint_passed'] for r in after['runs'])
    return dict(original_success=before['summary']['success_count'],candidate_success=after['summary']['success_count'],
        lost=lost,gained=gained,new_component_failures=regressions,physical_design_pass=clean,
        preserved=clean and not lost and not regressions,
        original_J=before['summary']['mean_yaw_score_deg'],candidate_J=after['summary']['mean_yaw_score_deg'],rows=rows)


def study_complete(p):
    file=OUT/'main_completion.json'
    if not file.exists():return False
    c=json.loads(file.read_text())
    return c.get('verified') is True and c.get('completed_runs')==main.jobs(p) and c.get('training_policy_steps')==1200000 and c.get('evaluations')==1254


def run():
    unit();p=main.verify()
    if not study_complete(p):
        raise RuntimeError('Incomplete six-run study: no qualification conclusion or partial-seed selection allowed')
    inputs={};data={};references={};pairs={};gates={};cpu_checks={};aux={}
    def read(file,cases,expected=None):
        digest=sha(file)
        if expected is not None:assert digest==expected
        inputs[str(file.relative_to(ROOT))]=digest
        return load_rows(file,cases)['runs']
    def restore(rows,cases):
        byid={c['seed']:i for i,c in enumerate(cases)}
        return wrap(ordered(rows,[byid[r['seed']] for r in rows],len(cases)))
    panels={k:p[k] for k in ['regular','controlled','legacy']};panels['auxiliary']=p['auxiliary_cases']
    for label in ['B0','B1-route']:
        for panel,cases in panels.items():
            refs=p['references']['aux_result_refs'] if panel=='auxiliary' else p['references']['phase_result_refs']
            rows=[]
            for ref in refs:
                if ref['label']!=label or (panel!='auxiliary' and ref['panel']!=panel):continue
                rows+=read(ROOT/ref['path'],[cases[i] for i in ref['indices']],ref['sha256'])
            references[(label,panel)]=restore(rows,cases)
    for job in main.jobs(p):
        arm,seed=job['arm'],job['seed'];d=OUT/'runs'/arm/str(seed)
        v=json.loads((d/'verification.json').read_text());assert v['verified'] and v['evaluations']==209 and v['policy_steps']==200000
        inputs[str((d/'verification.json').relative_to(ROOT))]=sha(d/'verification.json')
        init=json.loads((d/'initialization.json').read_text());assert init['engineering_warmstart'] is False
        inputs[str((d/'initialization.json').relative_to(ROOT))]=sha(d/'initialization.json')
        for panel,cases in panels.items():
            rows=[]
            for ref in v['evaluation_records']:
                if ref['panel']==panel:rows+=read(OUT/ref['path'],[cases[i] for i in ref['indices']],ref['sha256'])
            data[(arm,seed,panel)]=restore(rows,cases)
    for seed in p['seeds']:
        a=json.loads((OUT/'runs/anchored'/str(seed)/'initialization.json').read_text())
        b=json.loads((OUT/'runs/plain'/str(seed)/'initialization.json').read_text())
        assert a['initial_weight_sha256']==b['initial_weight_sha256'] and a['initial_world_sha256']==b['initial_world_sha256']
        for panel in ['regular','controlled','legacy']:
            candidate=data[('anchored',seed,panel)]
            for label in ['B0','B1-route']:
                result=paired(references[(label,panel)],candidate);pairs[f'anchored/{seed}/{panel}/vs_{label}']=result
                gates[f'preserve/{seed}/{panel}/{label}']=result['preserved']
            if panel!='legacy':
                result=paired(data[('plain',seed,panel)],candidate);pairs[f'anchored/{seed}/{panel}/vs_plain']=result
                gates[f'mechanism_preserve/{seed}/{panel}']=result['preserved']
        cpu_ref=p['references']['CPU_legacy'];cpu_file=ROOT/cpu_ref['path'];assert sha(cpu_file)==cpu_ref['sha256'];inputs[cpu_ref['path']]=cpu_ref['sha256']
        cpu=json.loads(cpu_file.read_text())['runs'];failures=[]
        for row,old in zip(data[('anchored',seed,'legacy')]['runs'],cpu):
            assert row['seed']==old['name'];flags=[]
            if not row['success']:flags.append('full_task')
            if row['velocity_rmse']>old['velocity_rmse']*1.05+.005+1e-12:flags.append('velocity')
            for k,axis in enumerate(['roll','pitch']):
                if row['peak_deg'][k]>old['peak_deg'][k]+.1+1e-12:flags.append(axis)
            if flags:failures.append(dict(case=row['seed'],flags=flags))
        cpu_checks[str(seed)]=dict(passed=not failures,failures=failures);gates[f'CPU_legacy/{seed}']=not failures
    scores={}
    for panel in ['regular','controlled']:
        values=[data[('anchored',s,panel)]['summary']['mean_yaw_score_deg'] for s in p['seeds']]
        plain=[data[('plain',s,panel)]['summary']['mean_yaw_score_deg'] for s in p['seeds']]
        reference=references[('B1-route',panel)]['summary']['mean_yaw_score_deg']
        direction=all(a is not None and b is not None and a<b for a,b in zip(values,plain))
        mechanism=(direction and float(np.mean(values))<=min(.85*float(np.mean(plain)),float(np.mean(plain))-.05)) if direction else False
        gates['advantage/'+panel]=yaw_gate(values,reference);gates['mechanism_score/'+panel]=mechanism
        scores[panel]=dict(reference_J=reference,anchored_J=values,plain_J=plain,reference_gate=gates['advantage/'+panel],mechanism_gate=mechanism)
    controlled_success=[data[('anchored',s,'controlled')]['summary']['success_count'] for s in p['seeds']]
    gates['controlled_mean_at_least34']=float(np.mean(controlled_success))>=34
    for seed in p['seeds']:
        rows=data[('anchored',seed,'auxiliary')]['runs'];zero=[r for r in rows if r['profile']=='zero'];pulse=[r for r in rows if r['profile']!='zero']
        aux[str(seed)]=dict(zero_success=sum(r['success'] for r in zero),pulse_success=sum(r['success'] for r in pulse),
            physical=sum(r['physical_safety_passed'] for r in rows),design=sum(r['design_joint_passed'] for r in rows),
            delivery_complete=all(r['force_delivery']['complete'] for r in rows),
            original_flags=[dict(case=r['seed'],flags=[k for k,v in full_flags(r).items() if v]) for r in rows if not r['success']])
    qualified=all(gates.values())
    counts={f'{arm}/{seed}/{panel}':dict(success=d['summary']['success_count'],total=len(d['runs']),physical=d['physical'],design=d['design'])
            for (arm,seed,panel),d in data.items()}
    write(OUT/'study_pair_review.json',dict(verified=True,gates=gates,formal_expansion_gate_passed=qualified,counts=counts,pairs=pairs,scores=scores,
        controlled_anchored_successes=controlled_success,CPU_legacy=cpu_checks,auxiliary_secondary=aux,
        source_sha256=sha(__file__),input_sha256=inputs,main_completion_sha256=sha(OUT/'main_completion.json'),trainer_contract_sha256=sha(OUT/'trainer_contract.json'),
        decision='Candidatebenefit qualified for contribution/freshformalplanning only' if qualified else 'Close candidatebenefit expansion at unchangedprimary/mechanism gates;no budget/reference/threshold/seed rescue',
        limits='Matched finite3seed qualification; used136 development+28regression+45operational cases notindependentgeneralization,kinematicmean constraint notdynamicsafety/newalgorithm byitself.Auxiliary cannot rescueprimary.',
        new_training_or_evaluations=0,entire_goal_complete=False))
    print('ALL ORIGINAL GATES',gates,'FORMAL EXPANSION',qualified,flush=True)
    print('SCORES',scores,'CONTROLLED',controlled_success,flush=True)


if __name__=='__main__':run()
