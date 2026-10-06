"""Original full-gate role comparisons and declared identity-region audit, no rollout."""
import json
import numpy as np
from reference_role_probe import OUT,rec
from review_nom_yaw_filter import compare,unit
from score_reference_learning import load_rows


def candidate_names(value):
    if isinstance(value,dict):return {k.replace('latest','candidate'):candidate_names(v) for k,v in value.items()}
    if isinstance(value,list):return [candidate_names(v) for v in value]
    return value


def run():
    unit();p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'completion.json').read_text())
    previous=json.loads((OUT/'round182_data_review.json').read_text());assert c['verified'] and previous['verified'] and c['comparison_records']==240
    source=json.loads((OUT/'source_contract.json').read_text());assert all(rec.sha(rec.ROOT/n)==v for n,v in source['source_sha256'].items())
    jobs={(j['arm'],j['noise'],j['classical']):j for j in c['records']};pairs={};inputs={};identity=[]
    for noise in ('clean','noisy'):
        for label in ('B0','B1-route'):
            ref=p['reuse_references'][noise+'/'+label];base=rec.ROOT/ref['path'];assert rec.sha(base)==ref['sha256']
            files={'original':base}
            for arm in ('floor_only','consistent_pair'):
                j=jobs[(arm,noise,label)];files[arm]=OUT/j['path'];assert rec.sha(files[arm])==j['sha256']
            loaded={name:load_rows(file,p['cases']) for name,file in files.items()}
            for file in files.values():inputs[str(file.relative_to(rec.ROOT))]=rec.sha(file)
            for arm,against in (('floor_only','original'),('consistent_pair','original'),('consistent_pair','floor_only')):
                pairs[arm+'/'+against+'/'+noise+'/'+label]=candidate_names(compare(loaded[against],loaded[arm]))
            for row in loaded['consistent_pair']['runs']:
                if row['target_leg_m'] not in (.16,.24,.30):continue
                file=files['consistent_pair'].parent/row['role_trace']['path'];assert rec.sha(file)==row['role_trace']['sha256']
                inputs[str(file.relative_to(rec.ROOT))]=rec.sha(file)
                with np.load(file,allow_pickle=False) as z:
                    data=z['trace'];delta=data[:,3]-data[:,2]
                    assert abs(delta).max()<1e-12
                    identity.append(dict(noise=noise,classical=label,case=row['seed'],height_m=row['target_leg_m'],
                        steps=len(data),nonzero_float_mean_rows=int(np.count_nonzero(delta)),max_abs_float_mean_difference_m=float(abs(delta).max())))
    floors=[v for k,v in pairs.items() if k.startswith('floor_only/original/')]
    governed=[v for k,v in pairs.items() if k.startswith('consistent_pair/')]
    assert len(floors)==4 and all(v['fixed_variant_qualification'] for v in floors)
    assert all(v['gained']==[6301009,6301011,6301013,6301015] and not v['lost'] for v in floors)
    assert len(governed)==8 and not any(v['fixed_variant_qualification'] for v in governed)
    added_records=sum(len(v['new_axis_or_task_failures']) for k,v in pairs.items() if k.startswith('consistent_pair/original/'))
    assert added_records==7 and len(identity)==48
    result=dict(verified=True,round=183,pairs=pairs,identity_region=identity,comparison_rows=240,unique_development_cases=20,
        floor_only_engineering_gate_passed=True,floor_only_gain_unique_case_ids=[6301009,6301011,6301013,6301015],
        consistent_pair_gate_passed=False,added_axis_comparison_records_vs_original=added_records,
        added_axis_unique_cases=[6301033,6301035],source_sha256=rec.sha(__file__),input_sha256=inputs,
        completion_sha256=rec.sha(OUT/'completion.json'),data_review_sha256=rec.sha(OUT/'round182_data_review.json'),new_evaluations=0,training_updates=0,
        decisions={'floor_only':'Retain as stronger conventional Nom candidate on this development panel; all4 controller/noise original-success and component gates preserved, four0.16m cases gained. Not deployed: original regression/regular and independent qualification remain required.',
                   'consistent_pair':'Close fixed combined projection variant: adds upper0.38m yaw violations in allfour comparisons vsoriginal/floor_only. Extra single-noise success6301033 does not rescue lost axis contract or justify conditional oracle/gain tuning.',
                   'identity':'Middle0.16/0.24/0.30 means numerically within1e-12 ofbase, with nonzero rounding counts reported. Gain6301033 is upper0.38m with genuine projection, not middle identity; no roundoff-induced methodological benefit claimed.',
                   'learning':'Engineering role correction and projection are not new algorithms/PPO superiority; no new learning budget or baseline replacement.'},
        limits='Paired20 existing cases under two fixedlaws/one per-case artificial noise realization. No independent test/seed population or globaldynamic safety. Fixedfloor changes support intent but original physical/design/task gates intact; reference projection effects not separated as a fullfactorial experiment.',
        next='184 determine broader strong-reference qualification and remaining mechanism/method/learning need from these results;185 deep review/cleanup. Full contribution/qualified3/formal5seeds/fresh independent tests/new manuscript unmet.')
    rec.atomic_json(OUT/'round183_pair_review.json',result)
    for key,v in pairs.items():print(key,'success',v['original_success'],v['candidate_success'],'Jreduction',v['yaw_score_reduction_deg'],
        'lost',v['lost'],'gained',v['gained'],'added',v['new_axis_or_task_failures'],'qualified',v['fixed_variant_qualification'],flush=True)
    print('IDENTITY48 streams, max mean rounding',max(x['max_abs_float_mean_difference_m'] for x in identity),flush=True)


if __name__=='__main__':run()
