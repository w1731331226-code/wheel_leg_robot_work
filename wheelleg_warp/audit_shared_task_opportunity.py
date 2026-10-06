"""Existing-case success witnesses and current Nom reference contract, no rollout."""
import json
from collections import Counter

import numpy as np

from analyze_reward_failures import flags
from score_reference_learning import load_rows
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

BASE = ROOT/'wheelleg_warp/results/paper_recovery_20261004'


def category(classical, learned, intervened):
    if any(classical): return 'classical_success_observed'
    if any(learned): return 'main_learned_success_observed'
    if any(intervened): return 'fixed_context_success_observed'
    return 'no_success_in_inspected_results'


def unit():
    assert category([True],[False],[False]) == 'classical_success_observed'
    assert category([False],[True],[False]) == 'main_learned_success_observed'
    assert category([False],[False],[True]) == 'fixed_context_success_observed'
    assert category([False],[False],[False]) == 'no_success_in_inspected_results'


def nominal_contract(cases):
    from train_height_comparison import raw_env
    selected = []
    for h in (.115,.16,.24,.30,.38):
        selected.append(next(c for c in cases if c['scenario']['stand_height_m'] == h))
    raw = raw_env(selected,'diff3')
    try:
        raw.reset(); ref = raw.k['reference'].numpy()
        assert not raw.state.numpy()[:,0].any() and not raw.data.time.numpy().any()
        rows = []
        for r,c in zip(ref,selected):
            rows.append(dict(case=c['seed'],commanded_height_m=c['scenario']['stand_height_m'],
                internal_reference_height_m=float(r[2]),lower_reference_m=float(r[3]),
                upper_reference_m=float(r[6]),roll_room_clamp_enabled=bool(r[4] > 0),
                symmetric_roll_offset_room_m=max(0.,min(float(r[2]-r[3]),float(r[6]-r[2])))))
        return dict(baseline=raw.baseline_version,worlds=5,constructor_reset_only=True,
                    recorded_environment_physics_steps=0,references=rows,
                    interpretation='Current control_step clips roll offset to this symmetric room. This is a setpoint contract; not a trajectory failure cause or physical impossibility proof.')
    finally: raw.close()


def run():
    unit(); main = BASE/'reference_budget_learning_v1'; comp = BASE/'wheel_component_projection_v1'
    p = json.loads((main/'proposal.json').read_text()); cp = json.loads((comp/'proposal.json').read_text())
    assert sha(ROOT/cp['parent_proposal']) == cp['parent_proposal_sha256']
    contracts = {}
    for out in (main,comp):
        contract = json.loads((out/'trainer_contract.json' if out == main else out/'source_contract.json').read_text())
        assert contract['proposal_sha256'] == sha(out/'proposal.json')
        assert all(sha(ROOT/n) == v for n,v in contract['source_sha256'].items())
        for path in (out/'proposal.json',out/('trainer_contract.json' if out == main else 'source_contract.json')):
            contracts[str(path.relative_to(ROOT))] = sha(path)
    paths = [(f'main/{a}/{s}/{panel}','learned',main/'runs'/a/str(s)/(panel+'_final.json'),panel)
             for s in p['seeds'] for a in p['arms'] for panel in ('regular','controlled')]
    paths += [(f'classical/{a}/{panel}','classical',main/'classical'/f'{a}_{panel}.json',panel)
              for a in p['classical_reference_labels'] for panel in ('regular','controlled')]
    paths += [(f'main_zero/{s}/{panel}','intervened',main/'zero_controls'/f'{s}_{panel}.json',panel)
              for s in p['seeds'] for panel in ('regular','controlled')]
    ledger = json.loads((comp/'completion.json').read_text())
    for j in ledger['records']:
        path = comp/j['path']; assert sha(path) == j['sha256']
        paths.append((f'component/{j["seed"]}/{j["condition"]}/{j["allocator"]}/{j["panel"]}','intervened',path,j['panel']))
    observed = {}; sources = {}; total = 0
    for label,kind,path,panel in paths:
        d = load_rows(path,p[panel]); sources[str(path.relative_to(ROOT))] = sha(path); total += len(d['runs'])
        for r in d['runs']:
            f = flags(r); assert r['success'] == (not any(f.values()))
            key = (panel,r['seed'])
            entry = observed.setdefault(key,dict(case=r['seed'],panel=panel,scenario=r['scenario'],height_m=r['target_leg_m'],
                successes=dict(classical=[],learned=[],intervened=[]),failures={}))
            assert entry['scenario'] == r['scenario']
            if r['success']: entry['successes'][kind].append(label)
            else: entry['failures'][label] = dict(flags=[k for k,v in f.items() if v],peak_deg=r['peak_deg'])
    assert len(paths) == 60 and total == 4080 and len(observed) == 136
    cases = []
    for e in observed.values():
        e['category'] = category(*[e['successes'][k] for k in ('classical','learned','intervened')]); cases.append(e)
    unresolved = [e for e in cases if e['category'] == 'no_success_in_inspected_results']
    learned_only = [e for e in cases if e['category'] == 'main_learned_success_observed']
    nd = nominal_contract(p['controlled'])
    result = dict(verified=True,inspected_result_panels=60,inspected_evaluations=4080,unique_development_cases=136,
        category_counts=dict(Counter(e['category'] for e in cases)),cases=sorted(cases,key=lambda e:e['case']),
        no_inspected_success_case_ids=[e['case'] for e in unresolved],learned_success_without_classical_case_ids=[e['case'] for e in learned_only],
        nominal_reference_contract=nd,input_sha256=sources,input_contract_sha256=contracts,source_sha256=sha(__file__),
        dependency_sha256={f'wheelleg_warp/{n}':sha(ROOT/'wheelleg_warp'/n) for n in ('analyze_reward_failures.py','score_reference_learning.py','native/controller.py','native/environment.py')},
        new_evaluation_rollouts=0,training_updates=0,
        interpretation='Case-wise witnesses under the original predicate, not a deployable oracle-switching policy, loaded-obstacle traversal certificate or aggregate learner superiority. V6 victories do not identify responsible channels or isolate authority/init/context. No inspected success does not prove unreachable. Symmetric Nom offset room is a verified setpoint fact, not established failure causality.',
        next='160 review whether a bounded shared-Nom/reference-feasibility intervention or controlled authority identification is needed before richer tasks/new learning; current closed wheel branches remain closed. Do not expand PPO or call all current tasks too easy/unreachable.')
    atomic_json(BASE/'round159_shared_task_opportunity.json',result)
    print('PASS4080/136 existing-case audit',result['category_counts'],'unresolved',result['no_inspected_success_case_ids'],'learned witnesses',result['learned_success_without_classical_case_ids'],flush=True)
    print('Nom reference rooms',[(r['commanded_height_m'],r['lower_reference_m'],r['upper_reference_m'],r['symmetric_roll_offset_room_m']) for r in nd['references']],flush=True)


if __name__ == '__main__': run()
