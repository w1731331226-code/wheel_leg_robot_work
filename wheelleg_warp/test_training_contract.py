"""CPU-only checks: no physics construction, learning, or historical result rewrites."""
import ast
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from training_contract import (TASK_CONTRACT_VERSION,yaw_total,promotion_allowed,checkpoint_hashes,
    verify_checkpoint,source_hashes,verify_protocol,digest)


def rejects(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected fail-closed rejection')


def check():
    # Load the actual small scoring functions without importing the GPU runtime.
    scores=[]
    for filename in ('train_terrain.py','train_terrain_v3.py'):
        path=Path(__file__).with_name(filename)
        function=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='score')
        scope=dict(yaw_total=yaw_total);exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
        scores.append(scope['score'])
    terrain=dict(total=48,success_count=40,complete=True,mean_yaw_score_deg=1.)
    legacy=dict(total=2,success_count=1,complete=True,mean_yaw_score_deg=1.)
    runs=[dict(seed=i,scenario={'terrain':'legacy','speed':.5},task_contract_version=TASK_CONTRACT_VERSION,success=i==0) for i in range(2)]
    reference=dict(terrain=terrain,legacy=legacy,terrain_runs=[],legacy_runs=runs)
    failed=deepcopy(reference);failed['terrain'].update(complete=False,mean_yaw_score_deg=None)
    assert math.isinf(scores[0](failed['terrain'],failed['legacy'])[-1])
    assert math.isinf(scores[1](failed)[-1])
    assert not promotion_allowed(failed,reference)
    candidate=deepcopy(reference);candidate['terrain']['success_count']=42
    assert promotion_allowed(candidate,reference) and scores[1](candidate)<scores[1](reference)
    # A compensating win cannot hide loss of an already-successful original case.
    candidate['legacy_runs'][0]['success']=False;candidate['legacy_runs'][1]['success']=True
    assert not promotion_allowed(candidate,reference)
    candidate=deepcopy(reference);candidate['legacy_runs'][0]['scenario']['speed']=.7
    rejects(lambda:promotion_allowed(candidate,reference))
    candidate=deepcopy(reference);candidate['legacy_runs'][0].pop('task_contract_version')
    rejects(lambda:promotion_allowed(candidate,reference))
    # The current best is a safe fallback when every new candidate is rejected.
    selected=min([reference]+[r for r in (failed,) if promotion_allowed(r,reference)],key=scores[1])
    assert selected is reference
    with TemporaryDirectory(prefix='wheelleg-training-contract-') as temporary:
        folder=Path(temporary);checkpoint=folder/'policy'
        for suffix in ('.zip','.pkl'):Path(str(checkpoint)+suffix).write_bytes(suffix.encode())
        source=folder/'extra_source.py';source.write_text('frozen = True\n')
        protocol=dict(task_contract_version=TASK_CONTRACT_VERSION,checkpoint=str(checkpoint),
            source_sha256={**source_hashes(__file__),str(source):digest(source)},**checkpoint_hashes(checkpoint))
        manifest=folder/'protocol.json';manifest.write_text(json.dumps(protocol))
        verify_protocol(folder,__file__,protocol);verify_checkpoint(checkpoint,protocol)
        Path(str(checkpoint)+'.pkl').write_bytes(b'changed normalization')
        rejects(lambda:verify_protocol(folder,__file__))
        Path(str(checkpoint)+'.pkl').write_bytes(b'.pkl')
        Path(str(checkpoint)+'.zip').write_bytes(b'changed checkpoint')
        rejects(lambda:verify_protocol(folder,__file__))
        Path(str(checkpoint)+'.zip').write_bytes(b'.zip')
        source.write_text('frozen = False\n');rejects(lambda:verify_protocol(folder,__file__))
        source.write_text('frozen = True\n')
        changed={**protocol,'training_budget_steps':123};manifest.write_text(json.dumps(changed))
        rejects(lambda:verify_protocol(folder,__file__,protocol))
        changed={**protocol,'task_contract_version':1};manifest.write_text(json.dumps(changed))
        rejects(lambda:verify_protocol(folder,__file__))
        changed=deepcopy(protocol);changed['source_sha256'].pop('wheelleg_warp/terrain_eval.py');manifest.write_text(json.dumps(changed))
        rejects(lambda:verify_protocol(folder,__file__))
    print('PASS: None scoring, incomplete rejection, per-case legacy veto, fallback, source/zip/pkl/protocol freezing')


def check_entrypoints():
    import train_terrain_v3 as formal
    import optimize_residual as targeted
    with TemporaryDirectory(prefix='wheelleg-training-entry-') as temporary:
        folder=Path(temporary);source=folder/'source'
        for suffix in ('.zip','.pkl'):Path(str(source)+suffix).write_bytes(suffix.encode())
        candidate=folder/'candidate';candidate.mkdir()
        (candidate/'protocol.json').write_text(json.dumps(dict(
            development_cases=[dict(terrain='step',terrain_seed=1)],legacy_regression_cases=[dict(terrain='legacy',terrain_seed=2)])))
        # The old initializer reads this file; the fixed initializer must ignore it.
        (candidate/'initial_evaluation.json').write_text('invalid historical score; do not trust')
        def replay(model,normalization,scenarios):
            assert str(normalization)==str(folder/'new/bootstrap/policy.pkl')
            return [dict(seed=s.terrain_seed,scenario=asdict(s),success=s.terrain=='legacy',reason='completed' if s.terrain=='legacy' else 'fall',
                task_contract_version=TASK_CONTRACT_VERSION,terrain_passed=s.terrain=='legacy',terrain_exit_passed=s.terrain=='legacy',
                terrain_evidence_passed=s.terrain=='legacy',peak_deg=[0.,0.,0.],task_goal_progress_m=2.75,
                physical_steps=10000,duration_s=5.,arrival_s=3.,rms_deg=[0.,0.,.1]) for s in scenarios]
        with patch.object(formal.TimedPPO,'load',return_value=SimpleNamespace(num_timesteps=10)),patch.object(formal,'evaluate_terrain',side_effect=replay) as evaluation:
            formal.initialize(folder/'new',source,candidate)
            assert evaluation.call_count==2
        initial=json.loads((folder/'new/initial.json').read_text())
        assert initial['terrain']['success_count']==0 and not initial['terrain']['complete']
        assert initial['legacy']['success_count']==1 and initial['legacy']['complete']
        assert json.loads((folder/'new/status.json').read_text())['initial_score'] is None
        formal.verify(folder/'new')
        # An invalid targeted protocol must fail before constructing 1024 worlds.
        stale=folder/'stale';stale.mkdir();(stale/'protocol.json').write_text('{}')
        with patch.object(targeted,'LiveNativeEnv') as environment:
            rejects(lambda:targeted.train(stale));environment.assert_not_called()
    print('PASS: v3 initializer re-evaluates source, incomplete initial score serializes, targeted training rejects stale input before physics')


def check_task_summary():
    from terrain_eval import summarize_terrain
    from pretrain_yaw import yaw_score
    scenario=dict(terrain='mixed',speed=.7,center=2.,offset=0.)
    row=dict(seed=1,scenario=scenario,task_contract_version=2,success=True,reason='completed',
        terrain_passed=True,terrain_exit_passed=True,terrain_evidence_passed=True,task_goal_progress_m=3.171,
        physical_steps=20000,duration_s=10.,arrival_s=8.,rms_deg=[0.,0.,2.])
    value=summarize_terrain([row],[1])['mean_yaw_score_deg']
    assert math.isclose(value,2*math.sqrt(10/(3.5+1.5*3.171/.7)),rel_tol=1e-12)
    legacy={**row,'scenario':{**scenario,'terrain':'legacy'},'task_goal_progress_m':2.75}
    assert summarize_terrain([legacy],[1])['mean_yaw_score_deg']==yaw_score(legacy)
    for invalid in (0.,-1.,float('nan'),float('inf')):
        rejects(lambda:summarize_terrain([{**row,'task_goal_progress_m':invalid}],[1]))
    rejects(lambda:summarize_terrain([{**row,'terrain_evidence_passed':False}],[1]))
    rejects(lambda:summarize_terrain([{**row,'task_contract_version':1}],[1]))
    print('PASS: actual-goal yaw normalization, legacy equality and invalid evidence/goal rejection')


def check_round_history():
    import train_terrain
    import train_terrain_v3
    from dashboard import server
    for entry in (train_terrain,train_terrain_v3):
        with TemporaryDirectory(prefix='wheelleg-round-history-') as temporary:
            folder=Path(temporary)
            summary=dict(total=1,success_count=1,complete=True,mean_yaw_score_deg=1.)
            best=dict(round=0,path=str(folder/'bootstrap/policy'),policy_steps=10,
                terrain=summary,legacy=summary,summary=summary,terrain_runs=[],legacy_runs=[])
            protocol=dict(max_rounds=1,patience=3,steps_per_round=1000,
                final_ood_cases=[],final_holdout_cases=[],legacy_regression_cases=[])
            (folder/'protocol.json').write_text(json.dumps(protocol))
            (folder/'selection.json').write_text(json.dumps(dict(best=best,anchor=best,rounds=[],stagnant_rounds=0)))
            def completed_round(*args,**kwargs):
                directory=folder/'round_001';directory.mkdir()
                (directory/'completed.json').write_text(json.dumps(dict(selected=best,stage=1,train_seconds=2.,total_seconds=3.,
                    last=dict(round_steps=1000,policy_steps=1010,updates=20))))
                return SimpleNamespace(returncode=0)
            with ExitStack() as mocks:
                mocks.enter_context(patch.object(entry,'protocol' if entry is train_terrain else 'verify',return_value=protocol))
                mocks.enter_context(patch.object(entry,'verify_protocol',return_value=protocol))
                mocks.enter_context(patch.object(entry,'verify_checkpoint'))
                mocks.enter_context(patch.object(entry.subprocess,'run',side_effect=completed_round))
                mocks.enter_context(patch.object(entry.TimedPPO,'load',return_value=SimpleNamespace()))
                mocks.enter_context(patch.object(entry,'evaluate_terrain',return_value=[]))
                mocks.enter_context(patch.object(entry,'summarize',return_value=summary))
                entry.orchestrate(folder)
            selection=json.loads((folder/'selection.json').read_text());history=selection['rounds'][0]
            assert selection['best']==best and selection['anchor']==best
            assert history['round']==1 and history['selected_source_round']==0 and history['policy_steps']==10
            assert history['round_steps']==1000 and history['last_trained_policy_steps']==1010
            assert (history['train_seconds'],history['total_seconds'],history['updates'])==(2.,3.,20)
            with patch.object(server,'RUN',folder),patch.object(server,'DATA',folder/'dashboard'):
                displayed=server.status()
            assert displayed['rate']==500. and displayed['selection']['rounds'][0]['policy_steps']==10
    print('PASS: both supervisors retain best but record current-round workload; dashboard status survives all-rejected rounds')


if __name__=='__main__':check();check_entrypoints();check_task_summary();check_round_history()
