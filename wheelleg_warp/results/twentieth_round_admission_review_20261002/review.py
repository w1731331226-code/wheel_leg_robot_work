"""Evidence ledger for the twentieth-round scope and readiness review."""
from pathlib import Path
import json,hashlib
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]


def run():
    inputs=[]
    def read(path):
        inputs.append(path);return json.loads(path.read_text())
    before=read(OUT.parent/'sixteenth_round_parameter_isolation_20261002/verification.json')
    static=read(OUT.parent/'seventeenth_round_projection_order_20261002/analysis.json')
    planning=read(OUT.parent/'eighteenth_round_robust_reference_20261002/full_verification.json')
    damping=read(OUT.parent/'nineteenth_round_motion_reference_20261002/verification.json')
    current=read(OUT/'native_panel/verification.json')
    default=read(OUT/'default_factory/verification.json')
    contract=read(OUT/'native_contract/verification.json')
    physical=read(OUT/'physical_legacy.json');packet=read(OUT/'packet_check/verification.json')
    assert contract['passed'] and physical['passed'] and packet['passed']
    assert contract['terminal_success']==[True,False] and contract['physical_pass']==[True,True]
    trace_path=OUT/'native_panel/trace.npz';inputs.append(trace_path)
    t=np.load(trace_path,allow_pickle=False)['trace']
    for w,row in enumerate(current['episodes']):
        valid=t[:,w,4]>0
        assert int(valid.sum())==row['physical_steps']==row['physical_evidence_steps']
        margin=float(np.min(1.4-abs(t[valid,w,:4])))
        assert margin==row['min_active_design_margin_rad']==current['active_design_margin_rad'][w]
        assert row['design_joint_passed']==(margin>=0)
        assert row['success']==current['original_full_gates'][w]
    assert sum(current['original_full_gates'])==15
    assert default['physical_pass_count']==6 and default['task_pass_count']==4
    assert all(r['baseline_version']=='height115-current-vmc-v5-full-design-candidate' for r in default['candidate'])
    result=dict(role='twentieth_round_objective_direction_and_admission_review',
        progress=dict(round16_single_factor_groups=before['groups'],
            round17_static_filter=static['static_projection_candidate'],
            round18_full_gate_groups=planning['full_panel_groups'],
            round19_full_gate_groups=damping['reports']['production_panel']['groups'],
            round20_native_full_gate_groups=current['groups']),
        default_factory=dict(physical=6,task=4,has_frozen_braking_reference=False),
        native_design_success_reward_history_contract=contract,
        previous_physical_contract_preserved=physical['passed'],observation_packet_preserved=packet['passed'],
        decision='Retain VMC/six-state LQR/diff3 research and the minimal damping fix. Next unify the actual shared reference execution/configuration, then verify its declared domain and freeze a same-information protocol before real short PPO update/restore/evaluation. No more online predictor/reset layers or static-filter tuning.',
        scope_rule='Keep original task/design thresholds and failed cases. Engineering readiness, learning performance and method advantage are distinct; historical paper readiness included28/32 classical success, not a blanket zero-Actor100% requirement. This does not grant current readiness: default normal tasks4/6 and missing unified/domain/learning evidence remain.',
        full_admission=False,learning=False,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    target=OUT/'verification.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS twenty-round ledger: native/history design parity, old physical/packet contracts, default4/6 versus external-reference15/18; no admission')


if __name__=='__main__':run()
