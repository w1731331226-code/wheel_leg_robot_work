"""Registered fixed-policy GPU qualification; no learning/parameter selection."""
from pathlib import Path
import json,gc,weakref
import numpy as np
import warp as wp
from stable_baselines3 import PPO
import route_pilot_env as experiment
from native.controller import D,control_physical_nominal
from gpu_reference_budget import apply_budget,WIDTH
from reference_residual_budget import reference
from smoke_reward_training import weight_digest
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004'
OUT=BASE/'reference_budget_v1'
MODES={'original':0,'reference_budget':1,'zero_leg_after_filter':2}


def capture(cases,agent,norm,classical,mode,owner_only=False):
    factory=experiment.raw_env;launch=wp.launch;pending=set(range(len(cases)));recorded={}
    def wrapped(selected,native):
        assert selected==cases and native=='diff3';n=len(cases);initial=np.zeros((n,WIDTH));initial[:,2]=1.
        stats=wp.array(initial,dtype=D);tracking=wp.array(np.ones(n,np.int32))
        def patched(kernel,dim,inputs=None,**kwargs):
            result=launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
            if kernel is control_physical_nominal:launch(apply_budget,n,[mode,inputs[0],inputs[7],inputs[12],inputs[5],inputs[15],inputs[14],tracking,stats])
            return result
        wp.launch=patched
        try:raw=factory(cases,native)
        finally:wp.launch=launch
        raw._reference_budget_buffers=(stats,tracking)
        gc.collect();assert all(weakref.ref(x)() is not None for x in raw._reference_budget_buffers)
        expected=np.array([reference(c['scenario']['stand_height_m'])[:2] for c in cases]);np.testing.assert_array_equal(raw.k['reference'].numpy()[:,:2],expected)
        wait=raw.step_wait
        def step_wait():
            result=wait();done=[w for w in np.flatnonzero(result[2]) if w in pending]
            if done:
                values=stats.numpy();marks=tracking.numpy()
                for w in done:
                    s=values[w].copy();assert np.isfinite(s).all();recorded[w]=s.tolist();result[3][w]['reference_budget_stats']=s.tolist();marks[w]=0;pending.remove(w)
                tracking.assign(marks)
            return result
        raw.step_wait=step_wait
        if owner_only:
            raw.reset();assert raw._reference_budget_buffers[0] is stats and raw._reference_budget_buffers[1] is tracking
        return raw
    experiment.raw_env=wrapped;before=(agent.num_timesteps,agent._n_updates,weight_digest(agent)) if agent else None
    try:
        if owner_only:
            raw=wrapped(cases,'diff3');raw.close();return dict(verified=True,worlds=len(cases),explicit_buffer_owners=True,reset_checked=True,physical_steps=0)
        result=experiment.evaluate(cases,'L2-route' if agent else 'M3-route',agent,norm,classical)
    finally:experiment.raw_env=factory;wp.launch=launch
    if agent:assert before==(agent.num_timesteps,agent._n_updates,weight_digest(agent))
    assert not pending and len(recorded)==len(cases)
    for w,row in enumerate(result['runs']):assert row['reference_budget_stats']==recorded[w]
    return result


def freeze():
    proposal=json.loads((OUT/'proposal.json').read_text());unit=json.loads((OUT/'gpu_unit.json').read_text());assert unit['verified'] and not (OUT/'runtime_contract.json').exists()
    proof=capture(proposal['cases'],None,None,proposal['classical_B1'],1,True)
    atomic_json(OUT/'gpu_owner_check.json',proof)
    prior=json.loads((BASE/'route_pilot_v1/proposal.json').read_text())
    names=['gpu_reference_budget.py','test_gpu_reference_budget.py','pilot_reference_budget.py','review_reference_budget.py','route_pilot_env.py','reference_residual_budget.py']
    sources={**prior['source_sha256'],**proposal['source_sha256'],**{f'wheelleg_warp/{n}':sha(ROOT/'wheelleg_warp'/n) for n in names}}
    assert all(sha(ROOT/n)==v for n,v in sources.items())
    atomic_json(OUT/'runtime_contract.json',dict(proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,gpu_unit_sha256=sha(OUT/'gpu_unit.json'),owner_check_sha256=sha(OUT/'gpu_owner_check.json'),
        inserted='After originalcontrol_physical_nominal filtering/projection, before physics. Extra rho applies to executed Nom/current leg endpoints; wheels/calculated Nom/filtermemory/originallambda unchanged.',
        diagnostic_semantics='Changed leg accepted-increment diagnostic reflects rho*(executed_current-executed_Nom) at existing residual scale; zero control reports0. Physical packet values follow changed actuation; packet shape/timing/rawfiltered-request fields unchanged.',
        stats_fields=['samples','sum_applied_alpha','min_applied_alpha','changed_steps','old_leg_energy','new_leg_energy','convex_excess','reserved_wheel_excess','zero_leg_identity_errors','invalid_projection_or_reference','max_applied_alpha','zero_leg_samples','reserved','valid_samples','sum_formula_rho','sum_original_lambda','reserved_lambda_change','reserved_nom_change'],
        budget_episodes=1840,training_updates=0,b1_identity='B1 evaluated through candidate observer; all leg endpoints must coincide, no commands changed. No bitwise cross-CUDA replay claim.'))
    print('FROZEN1840-episode GPU pilot after603 static checks and184-world owner/reset check',flush=True)


def verify():
    p=json.loads((OUT/'runtime_contract.json').read_text());assert p['proposal_sha256']==sha(OUT/'proposal.json') and all(sha(ROOT/n)==v for n,v in p['source_sha256'].items());return p


def run():
    verify();p=json.loads((OUT/'proposal.json').read_text());destination=OUT/'runs';destination.mkdir(exist_ok=False);jobs=[];completed=0
    try:
        plan=[('B1',None,'reference_budget')]+[(str(m['seed']),m,name) for m in p['models'] for name in MODES]
        for label,m,name in plan:
            job=label if m is None else label+'_'+name;atomic_json(OUT/'progress.json',dict(status='running',completed_episodes=completed,pending_job=job))
            if m:
                prefix=Path(m['prefix']);assert sha(prefix.with_suffix('.zip'))==m['checkpoint']['checkpoint_sha256'] and sha(prefix.with_suffix('.pkl'))==m['checkpoint']['normalization_sha256']
                agent=PPO.load(str(prefix)+'.zip',device='cuda');norm=str(prefix)+'.pkl';classic=None
            else:agent=None;norm=None;classic=p['classical_B1']
            result=capture(p['cases'],agent,norm,classic,MODES[name]);path=destination/(job+'.json');atomic_json(path,result)
            jobs.append(dict(label=job,path='runs/'+path.name,sha256=sha(path),episodes=184,summary=result['summary'],physical=result['physical'],design=result['design']));completed+=184
            atomic_json(OUT/'completed_jobs.json',dict(completed_episodes=completed,records=jobs));print('COMPLETED',completed,job,result['summary']['success_count'],result['physical'],result['design'],flush=True)
            verify()
        assert completed==1840;atomic_json(OUT/'completion.json',dict(completed_episodes=1840,records=jobs,training_updates=0,runtime_contract_sha256=sha(OUT/'runtime_contract.json')))
        atomic_json(OUT/'progress.json',dict(status='complete',completed_episodes=1840))
    except BaseException as e:
        atomic_json(OUT/'interruption.json',dict(completed_episodes=completed,error=str(e),pending_job_consumption_unknown=True,silently_resumable=False));raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);globals()[p.parse_args().command]()
