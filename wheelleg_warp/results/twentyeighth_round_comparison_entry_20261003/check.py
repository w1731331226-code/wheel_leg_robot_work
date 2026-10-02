"""Frozen entry/probes/B1/legacy/source evidence; no gate or final simulation."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from train_height_comparison import protocol,METHODS,summary
from training_contract import digest,verify_checkpoint
HERE=Path(__file__).resolve().parent;OUT=HERE/'protocol_v3'

def read(path):return json.loads(Path(path).read_text())
def check():
    p=protocol(OUT);assert p['environments']*p['ppo']['n_steps']==5000
    assert p['policy_steps_per_seed']%5000==p['evaluation_interval']%5000==0
    assert len(p['selection'])==32 and len(p['gate'])==64 and len(p['regression'])==28
    assert set(r['seed'] for r in p['selection']).isdisjoint(r['seed'] for r in p['gate'])
    for banks in p['training_banks'].values():
        for stage,rows in banks.items():
            assert len(rows)==100 and min(r['scenario']['stand_height_m'] for r in rows)==.115 and max(r['scenario']['stand_height_m'] for r in rows)==.38
            for r in rows:
                s=r['scenario'];assert s['solver_iterations']==100
                if int(stage)<3:assert s['mass']==7 and s['delay_ms']==0
                if int(stage)<2:assert s['mu_l']==s['mu_r']==.8 and s['drive_difference']==0
    finals=read(OUT/'sealed_final_cases.json')['cases'];assert len(finals)==15 and all(len(rows)==200 for rows in finals.values())
    assert all(not (720000<=r['seed']<730000) for rows in finals.values() for r in rows)
    for r in finals['pressure_height']:
        s=r['scenario'];assert s['terrain']=='legacy' and s['grade_deg']==s['roughness_m']==s['step_height_m']==0 and not s['relative_attitude']
    reports={}
    for method,mode in METHODS.items():
        folder=OUT/'engineering'/method;d=read(folder/'verification.json');c=read(folder/'configuration.json')
        assert d['passed'] and not d['formal_training'] and not d['checkpoint_promoted']
        assert c['protocol_sha256']==digest(OUT/'protocol.json') and c['ppo']==p['ppo']
        np.testing.assert_array_equal(np.array(c['initial_log_std'],np.float32),np.array(p['initial_log_std'][method],np.float32))
        assert (d['policy_steps'],d['updates'])==(12000,240)
        assert d['weights_changed'] and d['weights_optimizer_normalization_restored_exactly'] and not d['physical_environment_state_restored']
        assert d['source_sha256']==p['source_sha256'] and len(d['evaluation'])==4 and d['episodes']
        verify_checkpoint(folder/'probe',d['before_restore_checkpoint']);verify_checkpoint(folder/'resumed_probe',d['final_checkpoint'])
        assert 3 in d['stages_after_resume'] and all(x['actual_episode_end'] for x in d['curriculum_transitions'])
        for r in d['evaluation']:
            assert r['residual_mode']==mode and r['physical_steps']==r['physical_evidence_steps']
            assert r['observation_spec']['dimension']==38 and r['shared_reference_contract']=='public-region-v2-state-phase-vmc'
            assert 'final_mean_fk_leg_m' in r
        reports[method]=dict(policy_steps=d['policy_steps'],updates=d['updates'],evaluation_success=sum(r['success'] for r in d['evaluation']))
    classical=read(OUT/'classical_selection.json');assert classical['protocol_sha256']==digest(OUT/'protocol.json')
    candidates=[read(OUT/'classical_selection'/(c['name']+'.json')) for c in p['b1_candidates']]
    from pretrain_yaw import selection_key
    for c,d in zip(p['b1_candidates'],candidates):
        assert d['candidate']==c and [r['scenario'] for r in d['runs']]==[r['scenario'] for r in p['selection']]
        assert summary(d['runs'])==d['summary']
    best=min((r for r in candidates if r['summary']['complete']),key=lambda r:selection_key(r['summary']))
    assert best==classical['selected'] and classical['absolute_headroom_possible'] and not classical['gate_opened']
    for name in ('contract_check.json','curriculum_check.json'):assert read(HERE/name)['passed']
    legacy=read(HERE/'legacy_baseline.json');assert len(legacy['runs'])==28 and not legacy['gate_opened']
    assert [r['scenario'] for r in legacy['runs']]==[r['scenario'] for r in p['regression']]
    assert not (OUT/'gate_lock.json').exists() and not (OUT/'gate_opened.json').exists() and not (OUT/'readiness.json').exists() and not (OUT/'runs').exists()
    result=dict(passed=True,round=28,active_protocol=str(OUT.relative_to(ROOT)),protocol_sha256=digest(OUT/'protocol.json'),
        source_sha256=p['source_sha256'],probe_reports=reports,selected_b1=best['candidate'],classical_headroom_deg=classical['classical_headroom_deg'],
        classical_selection_success=best['summary']['success_count'],legacy_baseline_summary=legacy['summary'],
        legacy_original_speed_passes=sum(r['old_velocity_gate'] for r in legacy['comparisons']),legacy_original_attitude_passes=sum(r['old_roll_pitch_gate'] for r in legacy['comparisons']),
        gate_simulated=False,final_simulated=False,formal_training_started=False,formal_training_admitted=False,
        remaining=['explicit formal checkpoint resume with consumed-budget guard','original28 current B0 heading regression diagnosis','full final admission audit and round30 direction review'],
        verifier_sha256=digest(__file__))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS ROUND28: frozen current entry, three actual PPO probes, true model/reset switching, B1 selection/headroom and original28 evidence; gate/final closed')

if __name__=='__main__':check()
