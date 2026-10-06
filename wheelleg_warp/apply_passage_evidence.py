"""Apply explicit secondary evidence rules without relabeling original successes."""
import json
from collections import Counter
from passage_evidence_contract import OUT, REQUIRED, qualify
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json


def evidence(row):
    e={key:None for key in set(k for keys in REQUIRED.values() for k in keys)}
    e['original_task']=row['original_success']
    if 'geometry_prerequisite' in row:
        e['prescribed_path']=row['sampled_lane_bound_passed'] if row['interior_samples']>=2 else None
        e['ordered_event_passage']=row['before_entry_and_after_exit']
        e['adequate_sampling']=row['interior_density_passed'] if row['interior_samples']>=2 else None
    # Maximum-contact/late-pose summaries cannot certify complete support or flight.
    return e


def unit():
    base=dict(original_success=True,geometry_prerequisite=True,sampled_lane_bound_passed=True,interior_samples=10,
              before_entry_and_after_exit=True,interior_density_passed=True)
    assert qualify('ground_rollover',evidence(base))['status']=='insufficient_evidence'
    assert qualify('airborne_passage',evidence(base))['status']=='insufficient_evidence'
    bad={**base,'sampled_lane_bound_passed':False}; assert qualify('ground_rollover',evidence(bad))['status']=='failed'
    bad={**base,'original_success':False}; assert qualify('airborne_passage',evidence(bad))['status']=='failed'
    missing=dict(original_success=True,status='mixed mapping missing')
    assert qualify('ground_rollover',evidence(missing))['status']=='insufficient_evidence'


def run():
    unit(); draft=OUT/'path_qualification_draft_audit.json'; contract=OUT/'parallel_passage_evidence_contract.json'
    d=json.loads(draft.read_text()); c=json.loads(contract.read_text()); assert d['verified'] and c['verified']
    assert d['source_sha256']==sha(ROOT/'wheelleg_warp/path_qualification_draft.py')
    assert c['source_sha256']==sha(ROOT/'wheelleg_warp/passage_evidence_contract.py')
    for name,value in d['input_sha256'].items(): assert sha(OUT/name)==value
    result=[]
    for row in d['rows']:
        e=evidence(row); result.append(dict(controller=row['controller'],case=row['case'],original_success=row['original_success'],
            evidence=e,ground_rollover=qualify('ground_rollover',e),airborne_passage=qualify('airborne_passage',e)))
    assert len(result)==205
    summary={}
    for label in d['summary']:
        selected=[r for r in result if r['controller']==label]
        summary[label]=dict(original_success=sum(r['original_success'] for r in selected),
            ground=dict(Counter(r['ground_rollover']['status'] for r in selected)),air=dict(Counter(r['airborne_passage']['status'] for r in selected)))
        assert summary[label]['original_success']==d['summary'][label]['original_success']
    assert not any(r[k]['status']=='supported_at_recorded_resolution' for r in result for k in REQUIRED)
    atomic_json(OUT/'passage_evidence_application.json',dict(verified=True,records=205,unique_development_cases=41,summary=summary,rows=result,
        draft_sha256=sha(draft),contract_sha256=sha(contract),source_sha256=sha(__file__),
        no_original_relabel=True,new_evaluations=0,training_updates=0,
        interpretation='Failed refers only to this independently declared secondary definition when an observed original-task/path prerequisite is false; it is not a new original policy score. Insufficient evidence remains distinct. Max normal summaries never imply complete per-target top support or takeoff/landing.',
        next='169 choose one concrete data/task/reference hypothesis from these prerequisites and missing evidence; no automaticPPO or retrospective threshold changes;170 deep review/cleanup.'))
    print('PASS205 ternary applications, original labels unchanged;',summary,flush=True)


if __name__=='__main__': run()
