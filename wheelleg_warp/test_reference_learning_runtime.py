"""Shared-kernel light runtime interface checks; no robot/model/physics call."""
import ast
from pathlib import Path
import numpy as np
import warp as wp
import joint_reference_adapter as adapter
import reference_learning_runtime as runtime
from test_joint_reference_adapter import run as adapter_unit


def run():
    adapter_unit()
    # Both paths use one dispatcher/control/filter/reset implementation, not a second controller.
    source=Path(adapter.__file__).read_text();tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='instrument')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='instrument'
               for n in ast.walk(function))
    assert source.count('result = launch(candidate.control_physical_nominal, dim, args, **kwargs)')==1
    hooks=(wp.launch,adapter.environment.control_physical_nominal,adapter.complete.old.control_physical_nominal,adapter.complete.old.geometry)
    def fail(*args):raise RuntimeError('deliberate_light_factory_stop')
    try:adapter.instrument(fail,[dict(seed=321001,scenario=dict(solver_iterations=100))],'virtual6',dense=False)
    except RuntimeError as error:assert str(error)=='deliberate_light_factory_stop'
    else:raise AssertionError('Expected stopped fakefactory')
    assert hooks==(wp.launch,adapter.environment.control_physical_nominal,adapter.complete.old.control_physical_nominal,adapter.complete.old.geometry)
    # Parent curriculum result drives the exact per-world reference reset mask.
    parent=runtime.base.CurriculumEnv.step_wait;launch=wp.launch;events=[]
    done=np.array([False,True,False]);result=(np.zeros((3,38)),np.zeros(3),done,[{}, {}, {}])
    class Mask:
        def assign(self,value):events.append(value.copy())
    class Bare(runtime.ReferenceCurriculum):
        def __init__(self):pass
        num_envs=3
        venv=type('Raw',(),{'_joint_buffers':tuple([object()]*12+[Mask()])})()
    runtime.base.CurriculumEnv.step_wait=lambda self:result
    wp.launch=lambda kernel,dim,inputs:events.append((kernel,dim,inputs))
    try:
        got=runtime.ReferenceCurriculum.step_wait(Bare());assert got is result
    finally:runtime.base.CurriculumEnv.step_wait=parent;wp.launch=launch
    np.testing.assert_array_equal(events[0],[0,1,0]);assert events[1][0] is adapter.clear and events[1][1]==3
    from types import SimpleNamespace
    from run_joint_reference_zero_pair import evaluate
    bad=SimpleNamespace(observation_space=SimpleNamespace(shape=(481,)),action_space=SimpleNamespace(shape=(3,)))
    try:evaluate([], 'U_ref6',None,0,model=bad,normalization=None)
    except ValueError:pass
    else:raise AssertionError('Wrong policy interface admitted')
    try:evaluate([], 'joint_zero',None,0,model=bad,normalization='unused')
    except ValueError:pass
    else:raise AssertionError('Unregistered learned arm admitted')
    print('PASS321 shareddense/light control implementation,fakefactory hookrestoration,episode-endcurriculum resetmask/policy-boundary;0realenv/physics/learning',flush=True)


if __name__=='__main__':run()
