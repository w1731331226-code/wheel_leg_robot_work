"""Light/dense share the frozen kernel; actor modes preserve their contract."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import fixed_reference_force as force
import joint_state_guard as guard


def run():
    force.self_check()
    snapshot=Path(__file__).resolve().parent/'results/paper_recovery_20261004/joint_reference_candidate_v1/guarded_force_main_runtime_v1/guard_before_light.py'
    def funcs(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    before=funcs(snapshot.read_text());after=funcs(Path(guard.__file__).read_text())
    assert all(before[n]==after[n] for n in ('gain','inside','project','clear','guard'))
    events=[]
    class Model:
        policy=None
        num_timesteps=0
        _n_updates=0
        def predict(self,obs,deterministic=True):events.append(deterministic);return np.zeros((len(obs),3),np.float32),None
    model=Model();obs=np.zeros((2,481),np.float32)
    assert force.PolicyActor(model,'D3').predict(obs,True)[0].shape==(2,6)
    force.PolicyActor(model,'D3').predict(obs,False);force.PriorActor(model,'D3').predict(obs,True)
    assert events==[True,False,False]
    import execution_history_engineering as engine
    factory=engine.base.raw_env
    def fail(*args):raise RuntimeError('deliberate factory stop')
    engine.base.raw_env=fail
    p=dict(worlds=1,training_banks={'33331':{'1':[dict(seed=33300000,scenario=dict(solver_iterations=100))]}},curriculum_milestones=[20000,100000])
    try:
        try:force.make_env(p,'D3',33331,guarded=True)
        except RuntimeError as error:assert str(error)=='deliberate factory stop'
        else:raise AssertionError('Expected fakefactory failure')
        assert engine.base.raw_env is fail
    finally:engine.base.raw_env=factory
    print('PASS333 sharedkernel AST unchanged/latent3→6frozenmode/legacyprior/factory restoration;0robot/physics/learning')


if __name__=='__main__':run()
