"""Verify round26 interfaces, actual PPO updates/restoration and full B1 evidence."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def read(name):return json.loads((OUT/name).read_text())
def verify(hashes):
    for name,value in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name

def check():
    raw=read('torque6_fixture/verification.json');assert raw['passed']
    verify(raw['source_sha256'])
    assert raw['mapping_fixture_count']==288 and raw['raw_mapping_max_error_Nm']<1e-12
    assert raw['same_state_checks']>=60 and raw['same_state_zero_actor_command_max_difference_Nm']<1e-5
    rows=raw['normal_tasks'];assert len(rows)==18
    assert sorted((r['mode'],r['world']) for r in rows)==sorted((m,w) for m in ('diff3','virtual6','torque6') for w in range(6))
    for r in rows:
        assert r['success'] and r['physical_safety_passed'] and r['design_joint_passed']
        assert r['physical_steps']==r['physical_evidence_steps'] and r['observation_spec']['dimension']==38
    bp=read('b1_engineering_checked/protocol.json');verify(bp['source_sha256']);verify(bp['inputs_sha256'])
    b1=read('b1_engineering_checked/verification.json');assert b1['passed'] and len(b1['runs'])==144
    assert b1['selected_candidate'] is None and not b1['formal_selection']
    for i,summary in enumerate(b1['summaries']):
        part=b1['runs'][i*18:(i+1)*18]
        assert summary['candidate']==bp['candidates'][i]
        assert [r['seed'] for r in part]==list(range(18))
        assert summary['success_count']==sum(r['success'] for r in part)
        for w,r in enumerate(part):
            assert r['scenario']==bp['scenarios'][w] and r['physical_steps']==r['physical_evidence_steps']
            assert r['shared_reference_contract']=='public-region-v2-state-phase-vmc'
    old=read('../twentyfifth_round_native_admission_20261003/registered_v2/verification.json')
    assert [r['success'] for r in b1['runs'][:18]]==[r['success'] for r in old['episodes']]
    pp=read('ppo_torque6/protocol.json');ppo=read('ppo_torque6/verification.json')
    verify(pp['source_sha256']);verify(ppo['source_sha256']);verify(read('ppo_driver_sha256.json'))
    assert ppo['passed'] and ppo['learning_performed'] and not ppo['formal_training'] and not ppo['checkpoint_promoted']
    assert (ppo['policy_steps_before_restore'],ppo['updates_before_restore'],ppo['resumed_policy_steps'],ppo['resumed_updates'])==(2000,10,2200,11)
    for key in ('weights_changed','weights_optimizer_normalization_restored_exactly','deterministic_prediction_roundtrip','normalization_frozen_for_evaluation'):assert ppo[key]
    for key,ext in (('checkpoint_sha256','.zip'),('normalization_sha256','.pkl')):
        assert hashlib.sha256((OUT/('ppo_torque6/probe'+ext)).read_bytes()).hexdigest()==ppo['checkpoint_sha256'][key]
    assert len(ppo['evaluation'])==4 and ppo['collection_episodes']
    for r in ppo['evaluation']:
        assert r['residual_mode']=='torque6' and r['physical_steps']==r['physical_evidence_steps']
        assert r['observation_spec']['request_state_channels'][:4]==['alpha_left','beta_left','alpha_right','beta_right']
    for name,marker in (('virtual6.log','PASS: differential transfer'),('request_state.log','PASS virtual6: same38D packet')):assert marker in (OUT/name).read_text()
    result=dict(passed=True,round=26,normal_task_passes=18,b1_complete_episodes=144,
        b1_selected=False,raw_ppo_steps=2200,raw_ppo_updates=11,raw_probe_evaluation_success=sum(r['success'] for r in ppo['evaluation']),
        same_state_zero_actor_command_max_difference_Nm=raw['same_state_zero_actor_command_max_difference_Nm'],
        independent_closed_loop_qpos_max_difference=raw['independent_closed_loop_qpos_max_difference'],
        independent_closed_loop_command_max_difference_Nm=raw['independent_closed_loop_command_max_difference_Nm'],
        trajectory_bitwise_equivalence_claimed=False,formal_training_admitted=False,final_holdout_opened=False,
        remaining=['initial actual motor distribution fairness','complete new protocol, training/evaluation entrypoint and source freeze'],
        source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS ROUND26: torque6 mapping/timing/reset, shared normal18/18, eight B1 candidates144 episodes, real PPO2200steps11updates/restore and legacy regressions')

if __name__=='__main__':check()
