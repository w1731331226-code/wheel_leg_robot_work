"""Parameter-only registered data split/curriculum audit, never integrates holdouts."""
from pathlib import Path
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from training_contract import digest
from ppo_env import heldout_combination,Scenario
HERE=Path(__file__).resolve().parent;P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
p=json.loads((P/'protocol.json').read_text());final=json.loads((P/'sealed_final_cases.json').read_text())['cases']
basefields=Scenario.__dataclass_fields__
def reserved(s):return heldout_combination(Scenario(**{k:s[k] for k in basefields}))
trainids=set();banks=[]
for seed in p['training_seeds']:
    for stage,rows in p['training_banks'][str(seed)].items():
        assert len(rows)==100 and len({r['seed'] for r in rows})==100
        trainids.update(r['seed'] for r in rows);sc=[r['scenario'] for r in rows]
        assert all(not reserved(s) for s in sc)
        assert all(.115<=s['stand_height_m']<=.38 and 7<=s['mass']<=7.5 and .6<=s['mu_l']<=1 and .6<=s['mu_r']<=1 and abs(s['drive_difference'])<=.03 and 0<=s['delay_ms']<=10 for s in sc)
        if int(stage)<3:assert all(s['mass']==7 and s['delay_ms']==0 for s in sc)
        if int(stage)<2:assert all(s['mu_l']==s['mu_r']==.8 and s['drive_difference']==0 for s in sc)
        assert any(s['stand_height_m']==.115 for s in sc) and any(s['stand_height_m']==.38 for s in sc)
        banks.append(dict(seed=seed,stage=int(stage),worlds=100,terrains=dict(Counter(s['terrain'] for s in sc)),reserved_combinations=0))
selection={r['seed'] for r in p['selection']};gate={r['seed'] for r in p['gate']};finalids={r['seed'] for rows in final.values() for r in rows}
assert len(selection)==32 and len(gate)==64 and len(finalids)==3000
assert trainids.isdisjoint(selection|gate|finalids) and selection.isdisjoint(gate|finalids) and gate.isdisjoint(finalids)
assert all(reserved(r['scenario']) for r in final['reserved_combination'])
assert not any(720000<=n<730000 for n in trainids|selection|gate|finalids)
assert len(final)==15 and all(len(rows)==200 for rows in final.values())
assert not (P/'gate_opened.json').exists()
report=dict(passed=True,protocol_sha256=digest(P/'protocol.json'),final_parameter_sha256=digest(P/'sealed_final_cases.json'),
 train_unique_seed_ids=len(trainids),selection_ids=32,gate_ids=64,final_ids=3000,namespace_overlap=False,banks=banks,reserved_combination_excluded_training=True,
 final_role='Parameters read only, no simulation or scores',sampling_limit='100fixedworlds/stage/seed, same seed IDs reused across curriculum stages; not IID per episode',
 shared_bank_by_method='Training bank keyed only by seed/stage, all3methods use the identical entries',extra_task_evaluations=0,gate_or_final_simulated=False,verifier_sha256=digest(__file__))
(HERE/'registered_split_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS disjoint training300/selection32/gate64/final3000 namespaces; reserved combination excluded; shared3method banks and stage masks; holdout parameters only')
