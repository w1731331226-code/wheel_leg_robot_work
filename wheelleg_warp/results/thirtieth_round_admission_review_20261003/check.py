"""Finite engineering admission; learning effects and invariant safety are unproved."""
from pathlib import Path
import ast,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from train_height_comparison import protocol,METHODS,summary
from training_contract import digest,verify_checkpoint
from pretrain_yaw import selection_key
HERE=Path(__file__).resolve().parent
OUT=HERE/'protocol_gpu_v3'
PREVIOUS=HERE.parent/'twentyninth_round_baseline_and_resume_20261003'

def read(path):return json.loads(Path(path).read_text())
def matching(hashes):
    for name,value in hashes.items():assert digest(ROOT/name)==value,name
def node(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if getattr(n,'name',None)==name),include_attributes=False)

def run():
    p=protocol(OUT);assert p['device']=='cuda'
    assert p['environments']*p['ppo']['n_steps']==5000
    assert p['policy_steps_per_seed']==2000000 and p['evaluation_interval']==20000
    assert len(p['selection'])==32 and len(p['gate'])==64 and len(p['regression'])==28
    assert set(r['seed'] for r in p['selection']).isdisjoint(r['seed'] for r in p['gate'])
    old=read(PREVIOUS/'protocol_v8/protocol.json')
    for name in ('training_banks','selection','gate','regression','original_regression','b1_candidates','ppo','normalization','initial_log_std','nondegradation'):
        assert p[name]==old[name],name
    for name,value in old['source_sha256'].items():
        if name!='wheelleg_warp/train_height_comparison.py':assert digest(ROOT/name)==value,name
    assert digest(HERE/'comparison_before_cuda.py')==old['source_sha256']['wheelleg_warp/train_height_comparison.py']
    current=(ROOT/'wheelleg_warp/train_height_comparison.py').read_text();previous=(HERE/'comparison_before_cuda.py').read_text()
    for name in ('CurriculumEnv','evaluate','summary','assess','lock_gate','learn_exact'):
        assert node(current,name)==node(previous,name),name
    assert read(PREVIOUS/'curriculum_check_v8.json')['passed']
    assert read(PREVIOUS/'curriculum_check_v8.json')['stage3_physical_delay_effective']
    fixed=read(PREVIOUS/'fixed_capped_legacy.json');matching(fixed['source_sha256'])
    assert fixed['summary']['success_count']==fixed['old_velocity_passes']==fixed['old_roll_pitch_passes']==28
    assert [r['scenario'] for r in fixed['runs']]==[r['scenario'] for r in p['regression']]
    bank=HERE.parent/'twentyninth_round_capped_support_20261003'
    collection=read(bank/'state_collection.json');matching(collection['source_sha256'])
    assert len(collection['episodes'])==148 and all(r['physical_safety_passed'] and r['design_joint_passed'] for r in collection['episodes'])
    init=read(bank/'initial_action_config.json');matching(init['source_sha256'])
    assert init['state_bank_sha256']==digest(bank/'state_bank.npz') and not init['learning']
    panels={}
    for mode,count in [('speed_boundary_v2',144),('nearcut_v2',64)]:
        d=read(HERE/mode/'verification.json');matching(d['source_sha256'])
        assert d['baseline_version']==p['baseline_version'] and d['total']==d['physical']==d['design']==d['success']==count
        assert all(r['physical_steps']==r['physical_evidence_steps'] for r in d['episodes'])
        panels[mode]={k:d[k] for k in ('total','physical','design','success')}
    reports={}
    for method,mode in METHODS.items():
        folder=OUT/'engineering'/method;d=read(folder/'verification.json');c=read(folder/'configuration.json')
        assert d['passed'] and d['policy_steps']==12000 and d['updates']==240
        assert d['ppo_device']=='cuda' and d['policy_parameter_devices']==d['optimizer_moment_devices']==['cuda']
        assert d['physics_device'].startswith('cuda') and d['source_sha256']==p['source_sha256']
        assert d['weights_optimizer_normalization_restored_exactly'] and d['weights_changed']
        assert not d['checkpoint_promoted'] and not d['formal_training'] and not d['physical_environment_state_restored']
        assert c['protocol_sha256']==digest(OUT/'protocol.json') and c['ppo']==p['ppo']
        np.testing.assert_array_equal(np.array(c['initial_log_std'],np.float32),np.array(p['initial_log_std'][method],np.float32))
        verify_checkpoint(folder/'probe',d['before_restore_checkpoint']);verify_checkpoint(folder/'resumed_probe',d['final_checkpoint'])
        assert 3 in d['stages_after_resume'] and all(t['actual_episode_end'] for t in d['curriculum_transitions'])
        summary(d['evaluation'])
        reports[method]=dict(policy_steps=12000,updates=240,ppo_device='cuda',evaluation_success=sum(r['success'] for r in d['evaluation']),evaluation_design_failures=sum(not r['design_joint_passed'] for r in d['evaluation']))
    resume=read(HERE/'formal_resume_check.json');assert resume['passed'] and resume['ppo_device']=='cuda'
    assert resume['source_sha256']==digest(ROOT/'wheelleg_warp/train_height_comparison.py')
    assert resume['invalid_checkpoint_hash_rejected'] and resume['unsaved_budget_rejected'] and resume['fresh_initialization_calls']==1
    cuda=read(HERE/'initial_cuda_check.json');assert cuda['passed'] and cuda['states']==888 and len(cuda['rows'])==9
    assert cuda['protocol_sha256']==digest(OUT/'protocol.json')
    assert read(HERE/'contract_check.json')['passed']
    batch=read(HERE/'batch_check.json');assert batch['passed'] and batch['ppo_device']=='cuda'
    assert batch['environments']==100 and batch['policy_steps']==5000 and batch['optimizer_minibatch_updates']==200
    assert batch['protocol_sha256']==digest(OUT/'protocol.json')
    candidates=[read(OUT/'classical_selection'/(c['name']+'.json')) for c in p['b1_candidates']]
    for c,d in zip(p['b1_candidates'],candidates):
        assert d['candidate']==c and d['summary']==summary(d['runs'])
        assert [r['seed'] for r in d['runs']]==[r['seed'] for r in p['selection']]
        assert [r['scenario'] for r in d['runs']]==[r['scenario'] for r in p['selection']]
    selected=min((d for d in candidates if d['summary']['complete']),key=lambda d:selection_key(d['summary']))
    classical=read(OUT/'classical_selection.json')
    assert classical['selected']==selected and classical['absolute_headroom_possible']
    assert classical['protocol_sha256']==digest(OUT/'protocol.json') and not classical['gate_opened']
    finals=read(OUT/'sealed_final_cases.json')['cases'];assert len(finals)==15 and all(len(v)==200 for v in finals.values())
    assert all(not 720000<=r['seed']<730000 for rows in finals.values() for r in rows)
    assert not (OUT/'runs').exists() and not (OUT/'gate_opened.json').exists() and not (OUT/'gate_lock.json').exists()
    cleanup=read(HERE/'cleanup.json')
    for item in cleanup['exact_duplicate_final_parameters']:
        f=ROOT/item['path'];assert f.is_symlink() and digest(f)==item['sha256']
    result=dict(passed=True,round=30,engineering_ready_for_registered_three_seed_pilots=True,
        protocol_sha256=digest(OUT/'protocol.json'),source_sha256=p['source_sha256'],ppo_device='cuda',physics_device='cuda',
        panels=panels,original28_success=28,original28_nondegradation=True,public148_physical_design_passed=True,
        probe_reports=reports,formal_resume_checked=True,initial_cuda_checked=True,formal_size_batch_checked=True,
        selected_b1=selected['candidate'],classical_headroom_deg=classical['classical_headroom_deg'],
        classical_selection_success=selected['summary']['success_count'],gate_or_final_simulated=False,formal_training_started=False,
        policy_weights_from_probes_promoted=False,learning_convergence_proved=False,method_advantage_proved=False,
        invariant_safety_proved=False,robustness_scope='Finite development cases; remaining heading and learned-policy design failures retained',
        direction_review='VMC/fixed6state LQR/diff3 standard PPO retained conditionally; no additional controller mechanisms or gain scans',
        verifier_sha256=digest(__file__))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS round30 CUDA engineering readiness: three-mode updates/reload, exact budget, finite model panels, strong B1 headroom; gate/final closed')

if __name__=='__main__':run()
