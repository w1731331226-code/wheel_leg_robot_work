"""Native/host gate parity, frozen asset provenance and real PPO probe audit."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    package=ROOT/'wheelleg_warp/native/shared_reference.npz';z=np.load(package,allow_pickle=False)
    assert z['states'].shape==(2,400,6) and z['efforts'].shape==(2,400,3)
    assert np.linalg.eigvalsh(z['metric']).min()>0
    for name,h in zip(z['source_paths'],z['source_sha256']):assert sha(ROOT/str(name))==h
    fixture=json.loads((OUT/'native_fixture/verification.json').read_text())
    assert fixture['passed'] and fixture['native_cpu_request_peak_Nm']<=1e-5
    panels={}
    for mode in ('speed_boundary_v2','broad_v2','registered_v2','random_v2','nearcut_v2'):
        r=json.loads((OUT/mode/'verification.json').read_text())
        old=json.loads((ROOT/f'wheelleg_warp/results/twentyfourth_round_operating_regions_20261002/{mode}/verification.json').read_text())
        assert [e['success'] for e in r['episodes']]==[e['success'] for e in old['episodes']]
        assert r['baseline_version']=='height115-current-vmc-v6-public-region-reference-candidate'
        assert not r['host_request_injection']
        for e in r['episodes']:
            assert e['physical_steps']==e['physical_evidence_steps']
            assert e['design_joint_contract']=='active-1p4-v1'
            assert not e['success'] or(e['physical_safety_passed'] and e['design_joint_passed'])
        panels[mode]={k:r[k] for k in ('total','physical','design','success')}
        for name,h in r['source_sha256'].items():assert sha(ROOT/name)==h,name
    probes={}
    for mode in ('diff3','virtual6'):
        folder=OUT/f'ppo_{mode}';r=json.loads((folder/'verification.json').read_text());p=json.loads((folder/'protocol.json').read_text())
        assert r['passed'] and r['learning_performed'] and not r['formal_training'] and not r['checkpoint_promoted']
        assert r['policy_steps_before_restore']==2000 and r['updates_before_restore']==10
        assert r['resumed_policy_steps']==2200 and r['resumed_updates']==11
        assert r['weights_optimizer_normalization_restored_exactly'] and r['normalization_frozen_for_evaluation']
        assert len(r['evaluation'])==4 and r['collection_episodes']
        assert p['observation']['dimension']==38 and p['baseline_version']=='height115-current-vmc-v6-public-region-reference-candidate'
        for name,h in r['source_sha256'].items():assert sha(ROOT/name)==h,name
        assert sha(folder/'probe.zip')==r['checkpoint_sha256']['checkpoint_sha256']
        assert sha(folder/'probe.pkl')==r['checkpoint_sha256']['normalization_sha256']
        probes[mode]=dict(policy_steps=2200,updates=11,real_episodes=len(r['collection_episodes']),
            evaluation_completed=sum(e['reason']=='completed' for e in r['evaluation']),evaluation_success=sum(e['success'] for e in r['evaluation']))
    result=dict(native_panels=panels,real_ppo_probes=probes,native_host_gate_labels_equal=True,
        native_cpu_request_peak_Nm=fixture['native_cpu_request_peak_Nm'],
        stock_ppo_loss_used=True,probe_weights_not_promoted=True,
        full_admission=False,
        remaining='Full new comparison configuration and research protocol remain unfrozen: current NativeEnv supports diff3/virtual6; planned raw6/B1 evaluation and initial motor-distribution fairness still need a verified entry. Learning performance is not proved.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target=OUT/'verification.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS native frozen-region profile and real PPO2000→2200 steps/10→11 updates for two modes; formal comparison protocol pending')


if __name__=='__main__':run()
