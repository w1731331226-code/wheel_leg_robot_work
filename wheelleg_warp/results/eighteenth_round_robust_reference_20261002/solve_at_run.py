"""Original bounded SLSQP parameterization, actual plants, shared robust constraints."""
from pathlib import Path
import json
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from evaluate import Episodes,OUT,ROOT,SOURCE,SCALE,schedules,sha,VARIATIONS


def run_direction(direction):
    output=OUT/f'solve_world{direction}';output.mkdir()
    source=np.load(SOURCE/f'world{direction}_best.npz',allow_pickle=False)
    common=source['common'];x0=source['parameters'].copy()
    limit=1/np.max(abs(common),axis=0);low=np.tile(-limit,6);high=np.tile(limit,6)
    evaluator=Episodes(direction,arms=19)
    cache={};count=0;normalizer=None;best=None;started=perf_counter()
    log=output/'evaluations.jsonl'

    def evaluate(x):
        nonlocal count,normalizer,best
        if 'x' in cache and np.array_equal(cache['x'],x):return cache
        if count>=120:raise RuntimeError('Registered120 full-batch evaluation budget reached')
        points=np.tile(x,(19,1));increments=np.full(18,.01);increments[x+.01>high]*=-1
        for axis in range(18):points[axis+1,axis]+=increments[axis]
        plan=schedules(points,common,400)
        r=evaluator.evaluate(plan)
        costs=np.array(r['costs']).reshape(5,19)
        margins=np.array(r['margins']).reshape(5,19,48)
        # Actual task termination replaces the predictor's unconditional2s
        # propagation. Early-ended episodes stay infeasible, never padded.
        complete=np.array([1. if e['reason']=='completed' else -1. for e in r['episodes']]).reshape(5,19,1)
        constraints=np.concatenate([margins/SCALE,complete],axis=2)
        if normalizer is None:normalizer=float(costs[0,0])
        objective=costs[0]/normalizer
        values=constraints[:,0].reshape(-1)
        jac=((constraints[:,1:]-constraints[:,0:1])/increments[None,:,None]).transpose(0,2,1).reshape(-1,18)
        valid=bool(np.array(r['valid']).reshape(5,19)[:,0].all())
        violation=float(np.maximum(-values,0).max())
        row=dict(evaluation=count,parameters=x.tolist(),nominal_cost=float(costs[0,0]),
            scaled_violation=violation,all_registered_feasible=valid,wall_s=r['wall_s'],
            group_valid=np.array(r['valid']).reshape(5,19)[:,0].tolist(),
            finite_difference_increments=increments.tolist(),all_costs=costs.tolist(),all_scaled_constraints=constraints.tolist())
        with log.open('a') as stream:stream.write(json.dumps(row)+'\n')
        count+=1
        key=(valid,-violation,-float(costs[0,0]))
        if best is None or key>best['key']:
            best=dict(key=key,x=x.copy(),valid=valid,violation=violation,cost=float(costs[0,0]))
            np.savez_compressed(output/'best.npz',parameters=x,common=common,schedule=plan[:,0],margins=margins[:,0],costs=costs[:,0])
        print('SOLVE',direction,count,'valid',valid,'violation',violation,'cost',float(costs[0,0]),'seconds',r['wall_s'],flush=True)
        cache.clear();cache.update(x=x.copy(),cost=float(objective[0]),jac=(objective[1:]-objective[0])/increments,
            constraints=values,conjac=jac)
        return cache

    status=None
    try:
        result=minimize(lambda x:evaluate(x)['cost'],x0,jac=lambda x:evaluate(x)['jac'],method='SLSQP',
            bounds=list(zip(low,high)),constraints={'type':'ineq','fun':lambda x:evaluate(x)['constraints'],'jac':lambda x:evaluate(x)['conjac']},
            options={'maxiter':40,'ftol':1e-7})
        evaluate(result.x)
        status=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))
    except RuntimeError as error:
        status=dict(success=False,message=str(error))
    finally:evaluator.close()
    assert best is not None
    summary=dict(direction=direction,solver=status,evaluations=count,wall_s=perf_counter()-started,
        best_feasible=best['valid'],best_scaled_violation=best['violation'],best_nominal_cost=best['cost'],
        learning=False,default_promoted=False,full_admission=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),OUT/'evaluate.py',ROOT/'wheelleg_warp/optimize_braking_trajectory.py')})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # A new one-arm batch reruns standing-to-stop physics independently. Keep
    # the least-infeasible replay too; it is evidence, never a feasible policy.
    independent=Episodes(direction)
    try:
        independent.evaluate(schedules(best['x'][None,:],common,400),output/'independent')
    finally:independent.close()
    print('DIRECTION FINISHED',summary,flush=True)


if __name__=='__main__':
    assert json.loads((OUT/'checks.json').read_text())['preflight_passed']
    for direction in range(2):run_direction(direction)
