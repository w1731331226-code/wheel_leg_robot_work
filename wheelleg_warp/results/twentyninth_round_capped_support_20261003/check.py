"""CPU replay of frozen conditional RMS and common-rule/source checks."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from audit_initial_actions import MODES,context,rms,policy_mean
from training_contract import digest
OUT=Path(__file__).resolve().parent

def read(name):return json.loads((OUT/name).read_text())
def verify(hashes):
    for name,value in hashes.items():assert digest(ROOT/name)==value,name

def check():
    torch.set_num_threads(1)
    reg=read('preregistration.json');config=read('initial_action_config.json');result=read('verification.json');collection=read('state_collection.json')
    verify(config['source_sha256']);verify(result['source_sha256']);verify(collection['source_sha256']);verify(collection['input_sha256'])
    run=read('run_sources.json');verify(run['source_sha256'])
    assert run['preregistration_sha256']==config['preregistration_sha256']==digest(OUT/'preregistration.json')
    assert config['state_bank_sha256']==digest(OUT/'state_bank.npz') and result['config_sha256']==digest(OUT/'initial_action_config.json')
    assert result['passed'] and not result['learning'] and not result['formal_training_admitted'] and not result['final_holdout_opened']
    assert collection['snapshots']==888 and collection['terminal_fallback_count']==0
    assert len(collection['episodes'])==148
    assert all(r['physical_safety_passed'] and r['design_joint_passed'] and r['physical_steps']==r['physical_evidence_steps'] for r in collection['episodes'])
    with np.load(OUT/'state_bank.npz',allow_pickle=False) as saved:bank={k:saved[k].copy() for k in saved.files}
    assert bank['base'].shape==(888,40,6) and bank['obs'].shape==(888,38)
    np.testing.assert_array_equal(bank['world'],np.repeat(np.arange(148),6));np.testing.assert_array_equal(bank['phase'],np.tile(np.arange(6),148))
    np.testing.assert_array_equal(bank['fit'],bank['world']<84)
    assert np.all(bank['memory'][:,16:19]==0) and np.all(bank['obs'][:,32:]==0)
    for w,s in enumerate(collection['scenarios']):
        np.testing.assert_array_equal(bank['param'][bank['world']==w,13],s['stand_height_m'])
        np.testing.assert_array_equal(bank['param'][bank['world']==w,0],s['speed'])
    target=config['target_rms_normalized'];fit=np.flatnonzero(bank['fit']);std={};ranks={}
    for m in MODES:
        n=3 if m=='diff3' else 6;entry=config['methods'][m];assert entry['action_dim']==n and entry['residual_scale']==1
        log_std=np.array(entry['initial_log_std']);assert log_std.shape==(n,)
        std[m]=np.exp(log_std);assert np.all((std[m]>=reg['std_bounds'][0])&(std[m]<=reg['std_bounds'][1]))
        mean=policy_mean(bank,m,reg['policy_seed']);np.testing.assert_allclose(mean,bank['mean_'+m],atol=1e-7,rtol=0)
        c=context(bank,m,mean,fit,bank['gaussian_z']);actual=rms(c,std[m]);np.testing.assert_allclose(actual,result['calibration'][m]['calibrated_rms'],atol=1e-10,rtol=0)
        assert result['calibration'][m]['nfev']<=reg['max_nfev']
        ranks[m]=sorted(set(np.linalg.matrix_rank(bank['map_'+m]).tolist()))
    assert ranks==dict(diff3=[3],virtual6=[6],torque6=[6])
    native=result['native_check'];assert native['native_host_command_max_difference_Nm']<1e-5 and native['cpu_direct_motor_force_max_error_Nm']<1e-12
    deviations={};height_deviations={}
    for seed in map(str,reg['verification_policy_seeds']):
        deviations[seed]={};height_deviations[seed]={}
        for split in ('fit','validation'):
            stats=[result['statistics'][seed][m][split]['calibrated']['40'] for m in MODES]
            values=np.array([s['actual_motor_rms_normalized'] for s in stats]);geometric=np.exp(np.log(values).mean(axis=0))
            deviations[seed][split]=float(abs(values/geometric-1).max())
            groups=[s['height_groups'] for s in stats];assert all([g['height_interval_m'] for g in item]==[g['height_interval_m'] for g in groups[0]] for item in groups)
            by_height=[]
            for i,g in enumerate(groups[0]):
                a=np.array([item[i]['motor_rms_normalized'] for item in groups]);average=np.exp(np.log(a).mean(axis=0))
                by_height.append(dict(height_interval_m=g['height_interval_m'],max_rms_relative_deviation=float(abs(a/average-1).max())))
            height_deviations[seed][split]=by_height
    for s in result['statistics'].values():
        for m in MODES:
            for split in ('fit','validation'):
                for kind in ('default','pilot','calibrated'):
                    for tick in ('1','10','40'):
                        row=s[m][split][kind][tick]
                        assert 0<=row['lambda_zero_fraction']<=row['lambda_below_one_fraction']<=1
                        assert len(row['lambda_histogram'])==20 and sum(row['lambda_histogram'])==(504 if split=='fit' else 384)*reg['gaussian_draws']
                        cov=np.array(row['covariance_normalized']);np.testing.assert_allclose(cov,cov.T,atol=1e-14,rtol=0)
    summary=dict(passed=True,snapshots=888,fit_states=504,validation_states=384,training_performed=False,
        policy_seeds=reg['verification_policy_seeds'],max_fit_deviation_from_common_target=float(max(abs(np.array(result['calibration'][m]['relative_error_to_target'])).max() for m in MODES)),
        max_rms_deviation_across_methods=deviations,height_group_deviations=height_deviations,
        mapping_subspace_ranks=ranks,native_check=native,source_sha256={str(Path(__file__).relative_to(ROOT)):digest(__file__)},
        initial_config_sha256=digest(OUT/'initial_action_config.json'),full_distribution_equivalence_claimed=False,
        interpretation='Matched aggregate end-of20ms RMS on fit states; remaining height/state/temporal/covariance and reachable-set differences are reported, not removed.',
        remaining='Executable new training/selection/evaluation protocol, applied initialization and integrated short PPO admission')
    write=json.dumps(summary,indent=2);(OUT/'summary.json').write_text(write+'\n')
    print('PASS frozen conditional RMS replay; fit/validation/3seeds/source/config/888 states/motor force; intrinsic rank3 versus6 remains')

if __name__=='__main__':check()
