"""Restore registered order and enforce broad/legacy gates, no new rollout."""
import json
from collections import Counter
from floor_broad_adapter import OUT,rec
from train_height_comparison import summary
from review_nom_yaw_filter import compare,unit
from review_reference_role import candidate_names
from score_reference_learning import load_rows


def ordered(records,indices,count):
    out=[None]*count
    assert len(records)==len(indices)
    for row,i in zip(records,indices):
        assert 0<=i<count and out[i] is None;out[i]=row
    assert all(row is not None for row in out)
    return out


def wrap(rows):
    return dict(runs=rows,summary=summary(rows),physical=sum(r['physical_safety_passed'] for r in rows),design=sum(r['design_joint_passed'] for r in rows))


def run():
    unit();assert ordered(['b','a'],[1,0],2)==['a','b']
    try:ordered(['a','b'],[0,0],2)
    except AssertionError:pass
    else:raise AssertionError('Duplicate panel index accepted')
    p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'completion.json').read_text())
    previous=json.loads((OUT/'round187_data_review.json').read_text());assert c['verified'] and previous['verified'] and c['completed_new_evaluations']==344
    source=json.loads((OUT/'source_contract.json').read_text());assert all(rec.sha(rec.ROOT/n)==v for n,v in source['source_sha256'].items())
    inputs={};panels={};pairs={};calibration={}
    for label in p['classical']:
        for panel in ('regular','controlled'):
            ref=p['original_development_references'][label+'/'+panel];file=rec.ROOT/ref['path'];assert rec.sha(file)==ref['sha256']
            original=json.loads(file.read_text());inputs[str(file.relative_to(rec.ROOT))]=rec.sha(file)
            cases=[dict(seed=r['seed'],scenario=r['scenario']) for r in original['runs']]
            load_rows(file,cases)
            candidate=[];indices=[]
            key='regular' if panel=='regular' else 'controlled_remaining'
            wanted={r['seed']:i for i,r in enumerate(original['runs'])}
            for job in c['records']:
                if job['panel']==key and job['law']==label and job['arm']=='floor_only':
                    f=OUT/job['path'];assert rec.sha(f)==job['sha256'];data=json.loads(f.read_text());inputs[str(f.relative_to(rec.ROOT))]=rec.sha(f)
                    candidate.extend(data['runs']);indices.extend(wanted[r['seed']] for r in data['runs'])
            if panel=='controlled':
                ref=p['covered_candidate_references'][label];file=rec.ROOT/ref['path'];assert rec.sha(file)==ref['sha256']
                data=json.loads(file.read_text());inputs[str(file.relative_to(rec.ROOT))]=rec.sha(file)
                candidate.extend(data['runs']);indices.extend(wanted[r['seed']] for r in data['runs'])
            candidate=wrap(ordered(candidate,indices,len(cases)))
            assert [r['scenario'] for r in candidate['runs']]==[r['scenario'] for r in original['runs']]
            pair=candidate_names(compare(original,candidate));pairs[panel+'/'+label]=pair
            panels[panel+'/'+label]=dict(original_success=original['summary']['success_count'],candidate_success=candidate['summary']['success_count'],
                physical=candidate['physical'],design=candidate['design'],lost=pair['lost'],gained=pair['gained'],new_flags=pair['new_axis_or_task_failures'],qualified=pair['fixed_variant_qualification'])
        legacy={}
        for arm in ('original','floor_only'):
            rows=[];idx=[]
            for job in c['records']:
                if job['panel']=='legacy' and job['law']==label and job['arm']==arm:
                    f=OUT/job['path'];assert rec.sha(f)==job['sha256'];rows.extend(json.loads(f.read_text())['runs']);idx.extend(job['indices']);inputs[str(f.relative_to(rec.ROOT))]=rec.sha(f)
            legacy[arm]=wrap(ordered(rows,idx,28))
            assert [r['seed'] for r in legacy[arm]['runs']]==[r['seed'] for r in p['panels']['legacy']]
        pairs['legacy/'+label]=candidate_names(compare(legacy['original'],legacy['floor_only']))
        cpu_file=rec.ROOT/p['cpu_legacy_reference']['path'];assert rec.sha(cpu_file)==p['cpu_legacy_reference']['sha256'];cpu=json.loads(cpu_file.read_text())['runs']
        for arm,data in legacy.items():
            failures=[]
            for row,old in zip(data['runs'],cpu):
                assert row['seed']==old['name']
                flags=[]
                if not row['success']:flags.append('current_full_task')
                if row['velocity_rmse']>old['velocity_rmse']*1.05+.005+1e-12:flags.append('historic_velocity')
                for axis,name in ((0,'roll'),(1,'pitch')):
                    if row['peak_deg'][axis]>old['peak_deg'][axis]+.1+1e-12:flags.append('historic_'+name)
                if flags:failures.append(dict(case=row['seed'],flags=flags,originalCPUpeak=old['peak_deg'],currentGPUpeak=row['peak_deg'],
                    originalCPUvelocity=old['velocity_rmse'],currentGPUvelocity=row['velocity_rmse']))
            calibration[arm+'/'+label]=dict(current_success=data['summary']['success_count'],physical=data['physical'],design=data['design'],historic_gate_passed=not failures,failures=failures)
    inputs[str(cpu_file.relative_to(rec.ROOT))]=rec.sha(cpu_file)
    regular_clean=all(x['qualified'] for k,x in panels.items() if k.startswith('regular/'))
    qualified=regular_clean and all(x['qualified'] for k,x in panels.items() if k.startswith('controlled/')) and all(x['historic_gate_passed'] for x in calibration.values())
    assert not qualified
    rec.atomic_json(OUT/'round188_broad_pair_review.json',dict(verified=True,round=188,panels=panels,pairs=pairs,legacy_CPU_calibration=calibration,
        overall_engineering_qualification=qualified,source_sha256=rec.sha(__file__),input_sha256=inputs,
        completion_sha256=rec.sha(OUT/'completion.json'),data_review_sha256=rec.sha(OUT/'round187_data_review.json'),new_evaluations=0,training_updates=0,
        decision='Fixedfloor broad promotion rejected: aggregate success cannot hide lost previously successful cases or design violation. Legacy all28 success is separate from historical velocity/attitude calibration. No default baseline replacement/PPO/gain orfloor rescue sweep.',
        limits='Registered development/regression comparisons,notfresh independenttest/trainingseeds. OldCPU model/control timing distinct fromcurrentGPU; recorded calibration failures do not exclusively attribute tofloor intervention.',
        next='189inspect minimal actual design-crossing evidence/remaining method-contract need,190 direction review/cleanup. Full method/formal5seeds/independenttests/new manuscript remain unmet.'))
    print('PANELS',panels,flush=True)
    print('CPU calibration',{k:dict(passed=v['historic_gate_passed'],failures=len(v['failures'])) for k,v in calibration.items()},flush=True)
    print('ALL ENGINEERING QUALIFICATION',qualified,flush=True)


if __name__=='__main__':run()
