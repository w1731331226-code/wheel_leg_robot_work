"""CPU/mock pre-constructor resource, actual parameter and graph/FD bounds."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import gc
import weakref
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
    for eps in (0,-1,float('nan'),float('inf'),True):
        try:budget.fd(lambda *a:calls.append('bad-FD'),None,None,eps)
        except ValueError:pass
        else:raise AssertionError('Invalid FD epsilon reached the callback')
    assert not budget.FD and not calls
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
    closure_check();record_check();failure_check(Array);delivery_check(Array);wrapper_dispatch_check()
    print('PASS380 CPU/mock: inherited/direct source and metadata HEAD closure; initial/ongoing resource rejection; receiver cases and graph/first-episode binding; separate-zero Nom/v2/original-parking launch routing; independent failure saves/primary chain/close/release; actual terminal/reset and bounded no-NN actor;0physics/CUDA/FD/NN execution')


def closure_check():
    with TemporaryDirectory() as folder:
        root=Path(folder);out=root/'protocol';out.mkdir();durable=root/'durable';durable.mkdir();(root/'wheelleg_warp').mkdir()
        sources={}
        for name in worker.DIRECT_SOURCES+('old_only_engine_fixture.py',):
            file=root/'wheelleg_warp'/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('# CPU fixture\n');sources[str(file.relative_to(root))]=worker.sha(file)
        old=root/worker.OLD_SOURCE;old.parent.mkdir(parents=True);old.write_text(json.dumps(dict(verified=True,source_sha256=sources)))
        proposal=dict(source_sha256=sources,input_sha256={worker.OLD_SOURCE:worker.sha(old)},prospective_budget=dict(constructor_FD_plan=dict(total_max_calls=345)))
        (out/'proposal.json').write_text(json.dumps(proposal))
        budget=dict(proposal_sha256=worker.sha(out/'proposal.json'),totals=dict(constructors=68,episodes=584,graph_worldstep_cap=12839760,world_packet_cap=320994),input_sha256={},current_formula_source_sha256={})
        (out/'job_budget_review.json').write_text(json.dumps(budget))
        capacity=root/'installed_capacity_fixture.py';capacity.write_text('# Resolver fixture\n')
        resource=dict(verified=True,resource_admitted=True,archive_strategy_frozen=True,proposal_sha256=worker.sha(out/'proposal.json'),job_budget_review_sha256=worker.sha(out/'job_budget_review.json'),
            durable_root=str(durable),workspace_peak_B=1024,free_reserve_B=2*1024**3,required_durable_free_B=100*1024**3,input_sha256={},installed_capacity_resolver_sha256={str(capacity):worker.sha(capacity)})
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
            for bad in (dict(resource,workspace_peak_B=-1),dict(resource,required_durable_free_B=True)):
                try:worker.resource_gate(bad,resource['proposal_sha256'],resource['job_budget_review_sha256'],False)
                except AssertionError:pass
                else:raise AssertionError('Invalid resource byte count accepted')
            with patch.object(worker,'blob_hashes',side_effect=lambda data,names:{**head(data,names),'protocol/resource_admission.json':'wrong'}):
                try:worker.verify(True,initial_space=False)
                except AssertionError:pass
                else:raise AssertionError('Uncommitted resource metadata accepted')
            for name in ('execution_input_collector.py','old_only_engine_fixture.py'):
                missing=dict(admission,source_sha256={n:h for n,h in sources.items() if not n.endswith(name)})
                (out/'source_admission.json').write_text(json.dumps(missing))
                try:worker.verify(False,initial_space=False)
                except AssertionError:pass
                else:raise AssertionError('Actual/inherited engine absent from source admission')
            (out/'source_admission.json').write_text(json.dumps(admission))
            capacity.write_text('# Changed resolver\n')
            try:worker.verify(False,initial_space=False)
            except AssertionError:pass
            else:raise AssertionError('Changed installed contact-capacity resolver accepted')
            capacity.write_text('# Resolver fixture\n')
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


def failure_check(Array):
    try:
        try:raise AssertionError('physical-primary')
        except AssertionError:raise OSError('old-save-secondary')
    except OSError as error:assert worker.exception_chain(error)==["AssertionError('physical-primary')","OSError('old-save-secondary')"]
    class Broken(Array):
        def numpy(self):raise OSError('injected-full-save')
    raw=SimpleNamespace(_complete_buffers=[Broken([1])],_complete_chunks=[],_gyro_buffers=[None,Array([2]),Array([0])],_gyro_chunks=[],
        _role_buffers=[Array([3]),Array([4])],_role_chunks=[],_phase_buffer=Array([5]),_phase_chunks=[],
        _parking_buffers=[Array([0]),Array([0]),Array([6])],_parking_chunks=[],_joint_guard_buffers=[Array([7]),Array([8])],_classical_guard_chunks=[])
    with TemporaryDirectory() as folder:
        directory=Path(folder);errors=[];worker.preserve_dense(raw,directory,[],errors)
        assert len(errors)==1 and (directory/'interrupted_gyro_last_buffers.npz').exists() and (directory/'interrupted_guard_last_buffers.npz').exists()
        actor=worker.Actor([{},{}],dict(gamma=0),dict(kp=.4,kd=.3,roll_gain=0,arm=1));actor.terminals={0:np.zeros(481),1:np.ones(481)}
        original=worker.np.savez_compressed
        def save(path,**values):
            if Path(path).name=='actual_terminal_history_0.npz':raise OSError('terminal0-save')
            return original(path,**values)
        with patch.object(worker.np,'savez_compressed',side_effect=save):actor.save(directory,errors)
        assert len(errors)==2 and (directory/'actual_terminal_history_1.npz').exists()
        terminal_actor=worker.Actor([{}],dict(gamma=0),{})
        with patch.object(worker,'state_snapshot',side_effect=OSError('private-snapshot-read')):
            try:terminal_actor.received((None,None,np.array([True]),[dict(terminal_observation=np.arange(481))]),raw,object())
            except OSError:pass
            else:raise AssertionError('Private read failure swallowed')
        np.testing.assert_array_equal(terminal_actor.terminals[0],np.arange(481));assert not terminal_actor.pending[0]
        cart=SimpleNamespace(_cartesian_buffers={'latent':Array(np.zeros((2,3)))},_cartesian_chunks=[[np.zeros((1,38))],[np.ones((1,38))]],_cartesian_times=[[np.zeros((1,3))],[np.ones((1,3))]])
        worker.preserve_cart(cart,directory,True,errors);assert not (directory/'cartesian_prefix_0.npz').exists()
        worker.preserve_cart(cart,directory,False,errors);assert (directory/'cartesian_prefix_0.npz').exists() and (directory/'cartesian_prefix_1.npz').exists()
        closed=[];env=SimpleNamespace(close=lambda:closed.append(True))
        def broken(factory,*args):factory();raise ValueError('injected-constructor-failure')
        with patch.object(worker.evaluation,'make_env',return_value=(raw,None,env,None,env)),patch.object(worker.guard,'instrument',side_effect=lambda factory,*a,**k:factory()),\
             patch.object(worker,'build_delivery',side_effect=broken),patch.object(worker,'preserve'):
            try:worker.measure([],dict(chart='CartLive3',gamma=0),{},dict(parameters=dict(yaw_config={})),None,directory,errors)
            except ValueError as error:assert str(error)=='injected-constructor-failure'
            else:raise AssertionError('Constructor failure swallowed')
        assert closed==[True]
    class Node:pass
    node=Node();node.cycle=node;ref=weakref.ref(node);budget=worker.Budget();budget.open(dict(num_envs=1));budget.job['raw']=node
    budget.close();del node;gc.collect();assert ref() is None,'A closed job retained its raw cycle'


def delivery_check(Array):
    import native.environment as native
    parking=worker.evaluation.parking;calls=[];original_deliver=worker.cart.deliver
    def spy(kernel,dim,inputs=None,**kwargs):calls.append((kernel,inputs))
    def base(cases,mode,directory):
        q=Array(np.zeros((1,17)));v=Array(np.zeros((1,16)));targets=Array(np.zeros((1,6)));command=Array(np.zeros(1));active=Array(np.ones(1))
        state=Array(np.zeros((1,30)));other=Array(np.zeros((1,4)));obs=Array(np.zeros((1,38)));diag=Array(np.zeros((1,38)))
        raw=SimpleNamespace(num_envs=1,observation_spec={},obs=obs,k={'state':state},reset=lambda:obs.numpy(),step_wait=lambda:None,
            nominal_correction_enabled=True,shared_reference_enabled=True,control_kernel=worker.cart.NOMINAL,actuator_gain_upper=np.array(native.PUBLIC_ACTUATOR_GAIN_UPPER))
        args=[q,v,Array(np.zeros((1,28))),targets,command,active,state,other]+[other]*6+[Array(np.zeros((1,6))),diag,0,0,other,other]
        worker.wp.launch(native.reset_rows,1,[active]+[other]*11+[obs]+[other]*5)
        for _ in range(80):
            worker.wp.launch(native.command_step,1,[other,other,command,active]);worker.wp.launch(worker.cart.NOMINAL,1,args)
            after=[other]*22;after[10]=state;after[11]=diag;after[16]=obs;worker.wp.launch(native.after,1,after)
        return raw
    with TemporaryDirectory() as folder,patch.object(worker.wp,'init'),patch.object(worker.wp,'set_device'),patch.object(worker.wp,'load_module'),\
         patch.object(worker.wp,'zeros',side_effect=lambda shape,**k:Array(np.zeros(shape))),patch.object(worker.wp,'ones',side_effect=lambda shape,**k:Array(np.ones(shape))),\
         patch.object(worker.wp,'launch',side_effect=spy),patch.object(parking,'BASE_FACTORY',side_effect=base):
        directory=Path(folder);cases=[dict(seed=1)]
        def factory(condition='original'):
            return worker.guard.instrument(lambda:(parking.instrument(cases,'virtual6',condition,directory),),cases,directory,dense=True)
        raw=worker.build_delivery(factory,cases,directory,0)[0]
        assert raw._classical_delivery_contract['compiled_parking_prepare_enabled']==0 and raw._cartesian_topology['accepted_nominal_mode']==1
        assert sum(k is worker.deliver_accepted_nominal for k,_ in calls)==80 and worker.cart.deliver is original_deliver
        try:worker.build_delivery(lambda:factory('parking_request_withdrawal'),cases,directory,0)
        except AssertionError as error:assert 'Compiled parking' in str(error)
        else:raise AssertionError('Withdrawal parking mislabelled as original accepted')


def wrapper_dispatch_check():
    from gymnasium.spaces import Box
    from stable_baselines3.common.vec_env import VecEnv
    from route_state import RouteState
    from execution_history_env import ExecutionHistory
    events=[]
    class Raw(VecEnv):
        def __init__(self):
            self._execution_intervals=None;super().__init__(1,Box(-np.inf,np.inf,(38,),dtype=np.float32),Box(-1,1,(6,),dtype=np.float32))
        def reset(self):return np.zeros((1,38),np.float32)
        def step_async(self,actions):events.append('native_async');self.actual=actions.copy()
        def step_wait(self):
            events.append('native_wait');terminal=np.zeros(38,np.float32);terminal[0]=.1;terminal[20]=.05
            self._execution_intervals=dict(terminal_worlds=np.array([True]),sensor_elapsed_s=np.array([.0065]),command_integral=np.full((1,6),.001))
            return self.reset(),np.zeros(1),np.array([True]),[dict(terminal_observation=terminal,duration_s=.0065)]
        def close(self):events.append('close')
        def get_attr(self,name,indices=None):return [None]
        def set_attr(self,*a,**k):pass
        def env_method(self,*a,**k):return [None]
        def env_is_wrapped(self,*a,**k):return [False]
    raw=Raw();history=ExecutionHistory(RouteState(raw),raw,'H1');actor=worker.Actor([{}],dict(gamma=0),dict(kp=.4,kd=.3,roll_gain=0,arm=1))
    original=history.step_wait
    def waited():
        events.append('history_wait');value=original();actor.received(value);return value
    history.step_wait=waited;obs=history.reset();action=actor.predict(obs)[0];value=history.step(action)
    assert events==['native_async','history_wait','native_wait'];np.testing.assert_array_equal(raw.actual,action)
    assert actor.terminals[0].shape==(481,) and actor.terminals[0][351]==np.float32(.1)
    assert actor.terminals[0][371]==np.float32(.05) and actor.terminals[0][470]==np.float32(.0065/.02)
    np.testing.assert_array_equal(value[0][:,:470],0);assert history.valid[0,-1]==1 and history.valid[0].sum()==1
    history.close();assert events[-1]=='close'


def chain_check(raw,Array,descriptor):
    n=2;cases=[dict(seed=w,scenario=dict(delay_ms=.5)) for w in range(n)]
    obs=np.zeros((n,481),np.float32);terminal=np.arange(n*481,dtype=np.float32).reshape(n,481)
    raw.data.qpos=Array(np.zeros((n,17)));raw.data.qvel=Array(np.zeros((n,16)));raw.data.ctrl=Array(np.zeros((n,6)))
    raw.targets=Array(np.zeros((n,6)));raw.state=Array(np.zeros((n,39)));raw.active=Array(np.ones(n,int));raw.done=Array(np.zeros(n,int))
    raw.command=Array(np.zeros(n));raw.stopped_q=Array(np.zeros((n,17)));raw.stopped_v=Array(np.zeros((n,16)))
    raw.obs=Array(np.zeros((n,39)));raw.k={'state':Array(np.zeros((n,30)))};raw.ids=Array([7,10,12,15,4])
    raw._joint_guard_buffers=(Array(np.zeros((n,16))),Array(np.zeros((40,n,62))))
    raw._execution_buffers=(Array(np.zeros((n,6))),Array(np.zeros((n,82,6))),Array(np.zeros((n,82),int)))
    raw._execution_previous_sensor_steps=np.zeros(n,dtype=np.int64)
    raw._cartesian_buffers={name:Array(np.zeros((n,size))) for name,size in [('latent',3),('shadow',22),('reference',4)]}
    frames=np.zeros((40,n,26));frames[:,:,0]=1;frames[:,:,24]=1
    raw._complete_buffers=(Array(frames),None,None,None,Array(np.zeros((40,n,39))),Array(np.zeros((40,n,39))))
    raw.step_wait=lambda:(obs.copy(),np.zeros(n),np.ones(n,bool),[dict(terminal_observation=terminal[w].copy()) for w in range(n)])
    history=SimpleNamespace(frames=np.zeros((n,10,39),np.float32),inputs=np.zeros((n,9,8),np.float32),
        elapsed=np.zeros((n,9),np.float32),valid=np.zeros((n,10),np.float32),reset=lambda:obs.copy(),close=lambda:events.append('close'))
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
        runs=[]
        for w in range(n):
            trace=directory/f'cartesian_fake_{w}.npz';terminal=directory/f'cartesian_terminal_fake_{w}.npz'
            np.savez_compressed(trace,trace=np.zeros((40,38)));np.savez_compressed(terminal,latent=np.zeros(3))
            runs.append(dict(physical_steps=40,cartesian_trace=dict(path=trace.name,sha256=worker.sha(trace),rows=40),cartesian_terminal=dict(path=terminal.name,sha256=worker.sha(terminal))))
        return dict(runs=runs,physical=n,design=n),dict(evaluations=n)
    with TemporaryDirectory() as folder:
        directory=Path(folder);errors=[]
        with patch.object(worker.evaluation,'make_env',base),patch.object(worker.evaluation,'evaluate',evaluate),\
             patch.object(worker.guard,'instrument',guarded),patch.object(worker,'build_delivery',repaired),\
             patch.object(worker,'require_accepted_nominal'),patch.object(worker,'model_values',return_value=({'param':raw.param.numpy()},{})),\
             patch.object(worker,'chain_minima',return_value=([1,1],0)),patch.object(worker,'physical_metrics',return_value=(1,0,0)) as metrics,\
             patch.object(worker,'preserve_dense'),\
             patch.object(worker.evaluation.collector,'preserve'),patch.object(worker.cart,'preserve'):
            result,_=worker.measure(cases,dict(chart='CartReset3',gamma=0),descriptor,
                dict(parameters=dict(yaw_config=dict(kp=.4,kd=.3,roll_gain=0,arm=1))),budget,directory,errors)
        assert result['physical']==n and not errors and metrics.call_count==1
        assert events==['accepted_Nom','dense_guard','original_B1-route','close'] and worker.cart.deliver is saved_deliver
        for w in range(n):
            with np.load(directory/f'actual_terminal_history_{w}.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['actual_raw481'],terminal[w])
            with np.load(directory/f'actual_after_reset_{w}.npz',allow_pickle=False) as z:
                assert z['prefix'].shape==(82,6) and z['frames'].shape==(10,39)
                np.testing.assert_array_equal(z['cart_reference'],0);np.testing.assert_array_equal(z['cart_latent'],0)
        raw.param.x[0,3]+=.01
        with patch.object(worker,'require_accepted_nominal'):
            try:history.reset()
            except AssertionError:pass
            else:raise AssertionError('Full reset changed actual param before first launch')
        raw.param.x[0,3]-=.01
        with patch.object(worker,'chain_minima',return_value=([1,1],0)),patch.object(worker,'physical_metrics',return_value=(1,0,.01)):
            try:raw.step_wait()
            except AssertionError:pass
            else:raise AssertionError('Original physical violation did not stop the next packet')
        closed=budget.close();assert budget.job is None and closed['graph_calls']==0


if __name__=='__main__':run()
