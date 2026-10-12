"""Versioned rollout parity plus pre-call graph limits and failed-write preservation."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import run_cartesian_interface_v2 as worker
import run_cartesian_light_interface as old


def run():
    a=ast.parse(Path(old.__file__).read_text());b=ast.parse(Path(worker.__file__).read_text())
    oldfn=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name=='rollout')
    newfn=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name=='rollout')
    oldtry=next(x for x in oldfn.body if isinstance(x,ast.Try));newtry=next(x for x in newfn.body if isinstance(x,ast.Try))
    original=copy.deepcopy(oldtry.body);changed=copy.deepcopy(newtry.body)
    added=changed[0].body
    addition=next(x for x in added if isinstance(x,ast.If) and ast.dump(x.test)==ast.dump(ast.parse('changed.any()',mode='eval').body))
    assert len(addition.body)==1 and isinstance(addition.body[0],ast.Expr) and addition.body[0].value.func.id=='check_bank'
    added.remove(addition)
    for tree in (original,changed):
        for statement in tree:
            for node in ast.walk(statement):
                if isinstance(node,ast.Constant) and node.value=='LIGHT374':node.value='LIGHT368'
    assert [ast.dump(x,include_attributes=False) for x in original]==[ast.dump(x,include_attributes=False) for x in changed]
    p=dict(worlds=10,actual_graph_world_step_budget=960800,constructor_FD_budget=1);budget=worker.Budget(p);calls=[]
    raw=lambda:SimpleNamespace(num_envs=10,graph=object(),data=SimpleNamespace(qpos=SimpleNamespace(shape=(10,17))))
    r=raw();budget.bind(r);budget.active['calls']=1200;budget.graph_steps=480000
    budget.capture(lambda *x:calls.append('last'),r.graph)
    try:budget.capture(lambda *x:calls.append('bad'),r.graph)
    except RuntimeError:pass
    else:raise AssertionError('1202nd packet allowed')
    r2=raw();budget.bind(r2);oldgraph=r2.graph;r2.graph=object()
    try:budget.capture(lambda *x:calls.append('bad'),r2.graph)
    except RuntimeError:pass
    else:raise AssertionError('Replaced graph accepted')
    r2.graph=oldgraph;budget.graph_steps=960800
    try:budget.capture(lambda *x:calls.append('bad'),r2.graph)
    except RuntimeError:pass
    else:raise AssertionError('Total budget exceeded')
    budget.fd(lambda *x:calls.append('FD'),None,None,1e-6)
    try:budget.fd(lambda *x:calls.append('bad'),None,None,1e-6)
    except RuntimeError:pass
    else:raise AssertionError('FD overflow accepted')
    assert calls==['last','FD']
    model=SimpleNamespace(field=SimpleNamespace(numpy=lambda:np.array([[1],[2]])),stat=SimpleNamespace(meaninertia=SimpleNamespace(numpy=lambda:np.array([3,4]))))
    env_bank=SimpleNamespace(stages=np.array([1,2]),cache={1:dict(model={'field':np.array([[1],[9]])},meaninertia=np.array([3,9])),2:dict(model={'field':np.array([[8],[2]])},meaninertia=np.array([8,4]))})
    with patch.object(worker.np,'savez_compressed',lambda *a,**k:None):
        worker.check_bank(SimpleNamespace(model=model),env_bank,Path('mock'),'matched')
        env_bank.cache[2]['model']['field'][1]=7
        try:worker.check_bank(SimpleNamespace(model=model),env_bank,Path('mock'),'bad')
        except AssertionError:pass
        else:raise AssertionError('Wrong installed bank accepted')
    errors=[];writes=[];env=SimpleNamespace(transition_log=[])
    def npz(path,**values):writes.append(path.name);raise OSError('npz write')
    def blocks(*args):writes.append('blocks');raise OSError('block write')
    def metadata(path,value):writes.append(path.name)
    with patch.object(worker.np,'savez_compressed',npz),patch.object(worker,'flush',blocks),patch.object(worker,'atomic_json',metadata):
        try:
            try:raise ValueError('primary gate')
            except BaseException as primary:
                worker.finish(errors,primary,Path('mock'),[],[],[],env,{},dict(packet=np.zeros(1)));raise
        except ValueError as primary:assert str(primary)=='primary gate'
        assert writes==['last_submitted_actor.npz','blocks','episodes.json'] and len(errors)==2
        try:worker.finish([],None,Path('mock'),[],[],[],env,{},dict(packet=np.zeros(1)))
        except OSError:pass
        else:raise AssertionError('Successful path hid archive failure')
    print('PASS374 frozen rollout-body AST parity;originalgraph1201/total/FD pre-call caps;primary + independent archival errors;0physics/learning',flush=True)


if __name__=='__main__':run()
