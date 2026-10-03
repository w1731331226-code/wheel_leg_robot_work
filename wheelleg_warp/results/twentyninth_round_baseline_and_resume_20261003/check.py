"""Round29 controlled regressions and current-source fix evidence, not admission."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from training_contract import digest
from train_height_comparison import protocol
HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/name).read_text())
def run():
    current=protocol(HERE/'protocol_v8')
    cpu=read('cpu_current/verification.json');static=read('gpu_static5/verification.json');bare=read('gpu_bare_static/verification.json');coordinated=read('gpu_coordinated_static/verification.json')
    assert (cpu['success'],static['success'],bare['success'],coordinated['success'])==(28,19,28,28)
    fixed=read('fixed_capped_legacy.json');assert fixed['summary']['success_count']==28 and fixed['old_velocity_passes']==fixed['old_roll_pitch_passes']==28
    assert all(r['physical_safety_passed'] and r['design_joint_passed'] for r in fixed['runs'])
    bank=HERE.parent/'twentyninth_round_capped_support_20261003';collection=json.loads((bank/'state_collection.json').read_text())
    assert len(collection['episodes'])==148 and sum(r['success'] for r in collection['episodes'])==147
    assert all(r['physical_safety_passed'] and r['design_joint_passed'] for r in collection['episodes'])
    cal=json.loads((bank/'verification.json').read_text());assert cal['passed'] and not cal['learning']
    for name,value in cal['source_sha256'].items():assert digest(ROOT/name)==value,name
    assert cal['native_check']['native_host_command_max_difference_Nm']<1e-5
    assert 'PASS frozen conditional RMS replay' in (bank/'check.log').read_text()
    assert read('curriculum_check_v8.json')['stage3_physical_delay_effective'] and read('curriculum_check_v8.json')['all_reporting_contract_fields_match']
    assert read('budget_check.json')['resumed_exact_steps']==2000 and read('budget_check.json')['invalid_budget_rejected']
    resume=read('formal_resume_check.json');assert resume['passed'] and resume['actual_train_entry']
    assert resume['source_sha256']==digest(ROOT/'wheelleg_warp/train_height_comparison.py')
    assert resume['unsaved_budget_rejected'] and resume['invalid_checkpoint_hash_rejected'] and resume['fresh_initialization_calls']==1
    smoke=read('protocol_v8/engineering/M3/verification.json');assert smoke['passed'] and smoke['policy_steps']==12000
    for name,value in smoke['source_sha256'].items():assert digest(ROOT/name)==value,name
    rejected=HERE.parent/'twentyninth_round_initial_actions_20261003/state_collection.json';bad=json.loads(rejected.read_text())
    assert sum(not r['design_joint_passed'] for r in bad['episodes'])==2
    assert not (HERE/'protocol_v8/readiness.json').exists() and not (HERE/'protocol_v8/gate_opened.json').exists() and not (HERE/'protocol_v8/runs').exists()
    report=dict(passed=True,round=29,version=current['protocol'],active_protocol='protocol_v8',
        baseline_root_cause='Radial support used every target height as barrier, injecting >100N at high safe heights. Capped reference at existing original design minimum preserves low-height support.',
        original28_success=28,original28_velocity_passes=28,original28_roll_pitch_passes=28,
        public148_physical=148,public148_design=148,public148_success=147,
        rejected_fixed115_new_design_failures=2,curriculum_delay_and_reporting_fixed=True,
        real_standard_PPO_partial_budget_check=True,formal_resume_full_entry_checked=True,
        formal_resume_fixture='CPU synthetic environment; real train function and SB3 updates; native physical state restarts explicitly',
        current_source_M3_entry_probe_steps=12000,
        formal_training_admitted=False,gate_or_final_simulated=False,
        remaining=['current-v8 B2-V/B2 entry probes','B1 selection on v8 using unchanged selection cases','boundary/nearcut and current-source full admission audit','round30 direction review'],
        source_sha256=current['source_sha256'],verifier_sha256=digest(__file__))
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS ROUND29: controlled root cause, capped original28 restored, public148 safety/design, true stage delay/reporting and exact budget helper; full admission remains')
if __name__=='__main__':run()
