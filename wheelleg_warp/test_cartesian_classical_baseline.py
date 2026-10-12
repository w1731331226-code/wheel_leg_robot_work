"""CPU/mock pre-constructor resource, actual parameter and graph/FD bounds."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
import run_cartesian_classical_baseline as worker


def run():
    try:worker.resource_gate(dict(verified=False),'p','b')
    except worker.ResourceNotAdmitted:pass
    else:raise AssertionError('Unadmitted dense resources accepted')
    with patch.object(worker,'runtime',side_effect=AssertionError('Runtime queried before resource rejection')):
        try:worker.verify(False)
        except worker.ResourceNotAdmitted:pass
        else:raise AssertionError('Current unadmitted entry accepted')
    descriptor=dict(num_envs=2,deadline_packets=2,graph_worldstep_cap_per_constructor=160,solver_iterations=100,max_delay_ms=.5,
        cases=[dict(speed=.7,task_goal_progress_m_archived=1,param3_s_arithmetic=1.5+1.5/.7)]*2)
    descriptor['deadline_packets']=int(np.ceil((descriptor['cases'][0]['param3_s_arithmetic']+2)/.02))+2
    descriptor['graph_worldstep_cap_per_constructor']=80*descriptor['deadline_packets']
    class Array:
        def __init__(self,x):self.x=np.asarray(x)
        def numpy(self):return self.x.copy()
    raw=SimpleNamespace(num_envs=2,graph=object(),param=Array([[.7,1,1,descriptor['cases'][0]['param3_s_arithmetic'],1]]*2),
        cpu=SimpleNamespace(opt=SimpleNamespace(iterations=100)),model=SimpleNamespace(opt=SimpleNamespace(iterations=100)),
        data=SimpleNamespace(naconmax=128),scenarios=[SimpleNamespace(delay_ms=.5)]*2,_cartesian_topology={'counts':{'control':80}},
        _joint_guard_topology={'captured_controller_calls':80},_complete_topology={'physics_calls':40},
        _execution_topology={'control_steps':40},_parking_contract={'condition':'original'})
    calls=[];budget=worker.Budget();budget.open(descriptor)
    for _ in range(10):budget.fd(lambda *a:None,None,None,1e-6)
    try:budget.fd(lambda *a:calls.append('bad'),None,None,1e-6)
    except RuntimeError:pass
    else:raise AssertionError('Constructor11th FD accepted')
    with patch.object(worker,'require_accepted_nominal'):
        budget.bind(raw);raw.param.x[0,3]+=.01
        try:worker.check_actual(raw,descriptor)
        except AssertionError:pass
        else:raise AssertionError('Actual deadline drift accepted')
        raw.param.x[0,3]-=.01
    for graph in (object(),):
        try:budget.capture(lambda *a:calls.append('bad'),graph)
        except RuntimeError:pass
        else:raise AssertionError('Foreign graph accepted')
    budget.job['calls']=descriptor['deadline_packets']-1
    budget.capture(lambda *a:calls.append('last'),raw.graph)
    try:budget.capture(lambda *a:calls.append('bad'),raw.graph)
    except RuntimeError:pass
    else:raise AssertionError('Per-job deadline extended')
    raw.graph=object()
    try:budget.capture(lambda *a:calls.append('bad'),raw.graph)
    except RuntimeError:pass
    else:raise AssertionError('Graph replacement accepted')
    assert calls==['last'];first=budget.close();assert len(first['FD_calls'])==10
    budget.open(descriptor);assert budget.job['fd_limit']==5
    cases=[dict(seed=1),dict(seed=2)];actor=worker.Actor(cases,dict(gamma=0),dict(kp=.4,kd=.3,roll_gain=0,arm=1))
    obs=np.zeros((2,481),np.float32);action,_=actor.predict(obs);np.testing.assert_array_equal(action,0)
    terminal=np.arange(481,dtype=np.float32);actor.received((obs,np.zeros(2),np.array([True,False]),[dict(terminal_observation=terminal),{}]))
    actor.predict(obs);assert len(actor.parts[0])==1 and len(actor.parts[1])==2
    np.testing.assert_array_equal(actor.terminals[0],terminal);assert actor.policy.state_dict()=={}
    chain_check(raw,Array,descriptor)
    closure_check();record_check()
    print('PASS379 CPU/mock: resource-false before runtime/constructor,initial capacity vs ongoing reserve,metadata/source HEAD closure,receiver cases binding,first-episode vs graph steps,actual param/deadline/delay/contact capacity,first10/later5 FD,original graph/job cap,scoped v2 Nom/original B1 parking,dense physical stop,no-NN first-episode actor/actual terminal/reset preservation;0physics/GPU/FD execution')


def closure_check():
    with TemporaryDirectory() as folder:
        root=Path(folder);out=root/'protocol';out.mkdir();durable=root/'durable';durable.mkdir();(root/'wheelleg_warp').mkdir()
        sources={}
        for name in ('run_cartesian_classical_baseline.py','test_cartesian_classical_baseline.py','review_cartesian_classical_baseline.py'):
            file=root/'wheelleg_warp'/name;file.write_text('# CPU fixture\n');sources[str(file.relative_to(root))]=worker.sha(file)
        proposal=dict(source_sha256=sources,input_sha256={},prospective_budget=dict(constructor_FD_plan=dict(total_max_calls=345)))
        (out/'proposal.json').write_text(json.dumps(proposal))
        budget=dict(proposal_sha256=worker.sha(out/'proposal.json'),totals=dict(constructors=68,episodes=584,graph_worldstep_cap=12839760,world_packet_cap=320994),input_sha256={},current_formula_source_sha256={})
        (out/'job_budget_review.json').write_text(json.dumps(budget))
        resource=dict(verified=True,resource_admitted=True,archive_strategy_frozen=True,proposal_sha256=worker.sha(out/'proposal.json'),job_budget_review_sha256=worker.sha(out/'job_budget_review.json'),
            durable_root=str(durable),workspace_peak_B=1024,free_reserve_B=2*1024**3,required_durable_free_B=100*1024**3)
        (out/'resource_admission.json').write_text(json.dumps(resource))
        admission=dict(verified=True,proposal_sha256=resource['proposal_sha256'],job_budget_review_sha256=resource['job_budget_review_sha256'],
            resource_admission_sha256=worker.sha(out/'resource_admission.json'),runtime={},source_sha256=sources)
        (out/'source_admission.json').write_text(json.dumps(admission))
        def head(stdout,names):
            assert {'protocol/'+n for n in ('proposal.json','job_budget_review.json','resource_admission.json','source_admission.json')}<=set(names)
            return {n:worker.sha(root/n) for n in names}
        with patch.object(worker,'ROOT',root),patch.object(worker,'OUT',out),patch.object(worker,'runtime',return_value={}),\
             patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=3*1024**3)),\
             patch.object(worker.subprocess,'run',return_value=SimpleNamespace(stdout=b'')),patch.object(worker,'blob_hashes',side_effect=head):
            try:worker.verify(True)
            except worker.ResourceNotAdmitted:pass
            else:raise AssertionError('Initial full-queue space admission skipped')
            actual=worker.verify(True,initial_space=False);assert actual==(proposal,budget,admission,resource,durable/'protocol'/'runtime')
            with patch.object(worker,'blob_hashes',side_effect=lambda data,names:{**head(data,names),'protocol/resource_admission.json':'wrong'}):
                try:worker.verify(True,initial_space=False)
                except AssertionError:pass
                else:raise AssertionError('Uncommitted resource metadata accepted')
            with patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**3)):
                try:worker.verify(True,initial_space=False)
                except worker.ResourceNotAdmitted:pass
                else:raise AssertionError('Ongoing reserve exhaustion accepted')


def record_check():
    import review_cartesian_classical_baseline as receiver
    condition=dict(label='CartReset3_gamma0',chart='CartReset3',gamma=0)
    item=dict(panel='selection',batch=0,indices=[0],cases=[dict(seed=1,scenario={})])
    result=dict(runs=[dict(seed=1,scenario={},physical_steps=13,physical_safety_passed=True,design_joint_passed=True)])
    with TemporaryDirectory() as folder:
        queue=Path(folder);directory=queue/'job';directory.mkdir();(directory/'result.json').write_text(json.dumps(result))
        entry=worker.job_record(queue,directory,condition,item,'selection',result,{},dict(graph_calls=1,actual_graph_world_steps=40,FD_calls=[1e-6]*10))
        assert entry['first_episode_world_steps']==13 and entry['actual_graph_world_steps']==40 and entry['cases']==item['cases']
        proposal=dict(conditions=[condition],selection_jobs=[item]);completion=dict(records=[entry])
        with patch.object(receiver,'validated_result',return_value=result),patch('train_height_comparison.summary',return_value={}):
            received,_=receiver.load_phase(queue,proposal,completion,'selection',runtime=queue)
            assert received[condition['label']]['runs']==result['runs']
            entry['cases']=[]
            try:receiver.load_phase(queue,proposal,completion,'selection',runtime=queue)
            except AssertionError:pass
            else:raise AssertionError('Receiver accepted an unbound case list')


def chain_check(raw,Array,descriptor):
    n=2;cases=[dict(seed=w,scenario=dict(delay_ms=.5)) for w in range(n)]
    obs=np.zeros((n,481),np.float32);terminal=np.arange(n*481,dtype=np.float32).reshape(n,481)
    raw.data.qpos=Array(np.zeros((n,17)));raw.data.qvel=Array(np.zeros((n,16)));raw.data.ctrl=Array(np.zeros((n,6)))
    raw.obs=Array(np.zeros((n,39)));raw.k={'state':Array(np.zeros((n,30)))};raw.ids=Array([7,10,12,15,4])
    raw._joint_guard_buffers=(Array(np.zeros((n,16))),)
    raw._execution_buffers=(Array(np.zeros((n,6))),Array(np.zeros((n,82,6))),Array(np.zeros((n,82),int)))
    raw._execution_previous_sensor_steps=np.zeros(n,dtype=np.int64)
    raw._cartesian_buffers={name:Array(np.zeros((n,size))) for name,size in [('latent',3),('shadow',22),('reference',4)]}
    frames=np.zeros((40,n,26));frames[:,:,0]=1;frames[:,:,24]=1
    raw._complete_buffers=(Array(frames),None,None,None,Array(np.zeros((40,n,39))),Array(np.zeros((40,n,39))))
    raw.step_wait=lambda:(obs.copy(),np.zeros(n),np.ones(n,bool),[dict(terminal_observation=terminal[w].copy()) for w in range(n)])
    history=SimpleNamespace(frames=np.zeros((n,10,39),np.float32),inputs=np.zeros((n,9,8),np.float32),
        elapsed=np.zeros((n,9),np.float32),valid=np.zeros((n,10),np.float32),reset=lambda:obs.copy())
    history.step_wait=lambda:raw.step_wait()
    budget=worker.Budget();budget.open(descriptor)
    for _ in range(10):budget.fd(lambda *a:None,None,None,1e-6)
    saved_deliver=worker.cart.deliver;events=[]
    def base(rows,arm,norm,directory):
        assert rows==cases and arm=='B1-route' and norm is None
        assert worker.cart.deliver is worker.deliver_accepted_nominal;events.append('original_B1-route')
        return raw,None,history,None,history
    def guarded(factory,rows,directory,dense):
        assert rows==cases and dense is True;events.append('dense_guard');return factory()
    def repaired(factory,rows,directory,mode):
        assert mode==1 and worker.cart.deliver is worker.deliver_accepted_nominal;events.append('accepted_Nom');return factory()
    def evaluate(rows,arm,actor,norm,directory):
        env=worker.evaluation.make_env(rows,arm,norm,directory)[4]
        action=actor.predict(env.reset())[0];np.testing.assert_array_equal(action,0);env.step_wait()
        return dict(runs=[dict(physical_steps=40)]*n,physical=n,design=n),dict(evaluations=n)
    with TemporaryDirectory() as folder:
        directory=Path(folder);errors=[]
        with patch.object(worker.evaluation,'make_env',base),patch.object(worker.evaluation,'evaluate',evaluate),\
             patch.object(worker.guard,'instrument',guarded),patch.object(worker,'repaired_instrument',repaired),\
             patch.object(worker,'require_accepted_nominal'),patch.object(worker,'model_values',return_value=({'param':raw.param.numpy()},{})),\
             patch.object(worker,'chain_minima',return_value=([1,1],0)),patch.object(worker,'physical_metrics',return_value=(1,0,0)) as metrics,\
             patch.object(worker.evaluation.recorder,'preserve'),patch.object(worker.evaluation.parking,'preserve'),\
             patch.object(worker.evaluation.collector,'preserve'),patch.object(worker.cart,'preserve'):
            result,_=worker.measure(cases,dict(chart='CartReset3',gamma=0),descriptor,
                dict(parameters=dict(yaw_config=dict(kp=.4,kd=.3,roll_gain=0,arm=1))),budget,directory,errors)
        assert result['physical']==n and not errors and metrics.call_count==1
        assert events==['accepted_Nom','dense_guard','original_B1-route'] and worker.cart.deliver is saved_deliver
        for w in range(n):
            with np.load(directory/f'actual_terminal_history_{w}.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['actual_raw481'],terminal[w])
            with np.load(directory/f'actual_after_reset_{w}.npz',allow_pickle=False) as z:
                assert z['prefix'].shape==(82,6) and z['frames'].shape==(10,39)
                np.testing.assert_array_equal(z['cart_reference'],0);np.testing.assert_array_equal(z['cart_latent'],0)
        with patch.object(worker,'chain_minima',return_value=([1,1],0)),patch.object(worker,'physical_metrics',return_value=(1,0,.01)):
            try:raw.step_wait()
            except AssertionError:pass
            else:raise AssertionError('Original physical violation did not stop the next packet')
        closed=budget.close();assert budget.job is None and closed['graph_calls']==0


if __name__=='__main__':run()
