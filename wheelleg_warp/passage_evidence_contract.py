"""Three-valued qualification contract, separate from archived task success."""
import json
from pathlib import Path
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/task_mode_witness_v1'
COMMON=('original_task','prescribed_path','ordered_event_passage','same_time_complete_geometry','adequate_sampling')
REQUIRED={
    'ground_rollover':COMMON+('complete_per_target_contacts','positive_top_normal_support_every_central_sample','no_airborne_central_interval'),
    'airborne_passage':COMMON+('documented_takeoff','whole_obstacle_clearance','documented_landing','complete_air_interval_pose'),
}


def qualify(kind,evidence):
    if kind not in REQUIRED: raise ValueError('Unknown passage definition')
    if any(value is not None and type(value) is not bool for value in evidence.values()): raise ValueError('Evidence must be bool or unknown')
    conditions={key:evidence.get(key) for key in REQUIRED[kind]}
    failed=[key for key,value in conditions.items() if value is False]
    unknown=[key for key,value in conditions.items() if value is None]
    return dict(status='failed' if failed else ('insufficient_evidence' if unknown else 'supported_at_recorded_resolution'),failed=failed,unknown=unknown,
        limit='Evidence contract at recorded resolution, not continuous dynamics or a changed original task score.')


def unit():
    for kind,keys in REQUIRED.items():
        all_true={key:True for key in keys}; assert qualify(kind,all_true)['status']=='supported_at_recorded_resolution'
        absent=dict(all_true); absent.pop(keys[-1]); assert qualify(kind,absent)['status']=='insufficient_evidence'
        fail={**absent,keys[0]:False}; assert qualify(kind,fail)['status']=='failed'
        try: qualify(kind,{keys[0]:1})
        except ValueError: pass
        else: raise AssertionError('Numeric true accepted')


def run():
    unit(); geometry=OUT/'wheel_geometry_evidence_audit.json'; d=json.loads(geometry.read_text()); assert d['verified']
    result=dict(version='parallel-passage-evidence-contract-v1',verified=True,status='secondary operational evidence definitions; no primary training-task switch',
        original_contact='Original complete task/physics/design success remains authoritative for original task. No loaded/centred interpretation added retrospectively.',
        parallel_definitions={key:list(value) for key,value in REQUIRED.items()},
        rule='Any measured false fails this separate definition; otherwise missing evidence is unknown, never passed. All true only supports recorded-resolution definition.',
        ground_definition='Conservative continuously observed ground rolling: designated centre route, ordered target event, exact same-time whole tyre geometry/clearance and complete per-target contact list; every recorded central sample has positive upward top normal support and no airborne central interval.',
        ground_scope='This strict continuously observed diagnostic has zero permitted recorded unloading samples. It is not an accepted general uneven-terrain task; natural bouncing may fail it. Any tolerant duty/gap rule needs separate preregistration, not threshold selection from model scores.',
        airborne_definition='Separately document takeoff, same-time full-shape obstacle clearance and landing with adequate interval pose coverage; no ground-support certificate claimed.',
        numeric_scope='Existing1e-6N normal-load noise distinction only; no minimum-load fraction/duty/gap or performance threshold newly tuned. Spatial/sampling tolerances must be registered with evaluator before use.',
        current112_capability='Current max-contact witnesses, absent pre orientation and incomplete mixed-event mapping cannot make all required conditions true. Keep full passage unknown where necessary.',
        assumptions='Prescribed-lane ground and airborne definitions are secondary examples while optional user task choice is pending. No scope approval inferred from silence.',
        geometry_audit_sha256=sha(geometry),source_sha256=sha(__file__),new_physics_rollouts=0,training_updates=0,
        next='168 boundary fixtures and existing205 three-valued application, original task labels separately retained; no new training/repair;170 deep review/cleanup.',goal_status='active')
    atomic_json(OUT/'parallel_passage_evidence_contract.json',result); print('PASS8 ternary/invalid checks; parallel definitions, original task intact;0rollout',flush=True)


if __name__=='__main__': run()
