"""CPU/native-kernel parity and buffer ownership; never constructs or steps a robot."""
import json
import numpy as np
import warp as wp
from cartesian_pair_action import OUT,BASE,request
from cartesian_pair_kernel import deliver,clear_native,reset_aux,observe_latent,TRACE_WIDTH
from native.controller import D
from state_estimation import leg_kinematics
from review_yaw_sector import ROOT,sha
from run_wheel_interference import runtime
from dashboard.live_env import atomic_json


def bounds(speeds):
    rpm=abs(speeds.astype(float))*60/(2*np.pi);rated=np.array([175.]*4+[490.]*2);free=np.array([280.]*4+[710.]*2)
    return np.array([40.]*4+[4.5]*2)*np.where(rpm>rated,np.maximum(0,(free-rpm)/(free-rated)),1)/np.array([1.]*4+[1.05]*2)


def launch(q,v,target,previous,reference,base,mode,active=None,errors=None):
    n=len(q);active=np.ones(n,np.int32) if active is None else active
    state=np.tile(np.arange(22),(n,1)).astype(float);shadow=state.copy()
    for j in range(3):shadow[:,16+2*j]=previous[:,j];shadow[:,17+2*j]=-previous[:,j]
    diag=np.zeros((n,38));diag[:,:6]=base;diag[:,12]=1
    if errors is not None:diag[:,14]=errors
    args=dict(q=wp.array(q,dtype=float),v=wp.array(v,dtype=float),ids=wp.array([0,1,2,3,0,1,2,3,4,5],dtype=int),
        target=wp.array(target,dtype=float),active=wp.array(active,dtype=int),latent=wp.array(previous,dtype=D),reference=wp.array(reference,dtype=D),
        state=wp.array(state,dtype=D),shadow=wp.array(shadow,dtype=D),ctrl=wp.array(base,dtype=float),diag=wp.array(diag,dtype=D),trace=wp.zeros((1,n,TRACE_WIDTH),dtype=D))
    wp.launch(clear_native,n,[args['active'],args['state']])
    wp.launch(deliver,n,[0,mode,*[args[k] for k in ('q','v','ids','target','active','latent','reference','state','shadow','ctrl','diag','trace')]])
    result={k:args[k].numpy().copy() for k in ('latent','state','shadow','ctrl','diag','trace')}
    np.testing.assert_array_equal(result['state'][:,:16],state[:,:16])
    return result


def unit(q,v,reference):
    previous=np.ones((4,3))*.25;previous[-1]=0;target=np.zeros((4,6),np.float32);target[2,1]=.123
    active=np.array([0,1,1,1],np.int32);errors=np.array([0,2,0,0])
    base=np.ones((4,6),np.float32)*.1
    result=launch(q[:4],v[:4],target,previous,reference[:4],base,0,active,errors)
    np.testing.assert_array_equal(result['latent'],previous)
    np.testing.assert_array_equal(result['trace'][0,:,33],[0,2,2,0])
    np.testing.assert_array_equal(result['ctrl'][0],base[0]);np.testing.assert_array_equal(result['ctrl'][3],base[3])
    np.testing.assert_array_equal(result['ctrl'][2],0)
    public=np.zeros((3,38),np.float32);public[:,12:16]=q[:3]
    mask=wp.array([1,0,1],dtype=int);lat=wp.ones((3,3),dtype=D);shadow=wp.ones((3,22),dtype=D);ref=wp.ones((3,4),dtype=D)
    wp.launch(reset_aux,3,[mask,wp.array(public),lat,shadow,ref])
    np.testing.assert_array_equal(lat.numpy(),[[0,0,0],[1,1,1],[0,0,0]])
    np.testing.assert_array_equal(shadow.numpy()[1],1);np.testing.assert_array_equal(ref.numpy()[1],1)
    for i in (0,2):
        expected=np.concatenate([leg_kinematics(q[i,2*s:2*s+2].astype(float),np.zeros(2))[0] for s in range(2)])
        np.testing.assert_allclose(ref.numpy()[i],expected,atol=1e-7,rtol=0)
    observation=np.tile(np.arange(38),(3,1)).astype(np.float32);obs=wp.array(observation)
    wp.launch(observe_latent,3,[lat,obs]);np.testing.assert_array_equal(obs.numpy()[:,:32],observation[:,:32])
    np.testing.assert_array_equal(obs.numpy()[1,32:],[1,-1,1,-1,1,-1]);np.testing.assert_array_equal(obs.numpy()[0,32:],0)
    print('PASS363 inactive/upstream-error/bad-pair/cold-zero/masked-reset/observation ownership',flush=True)


def run():
    p=json.loads((OUT/'runtime_proposal.json').read_text());old=json.loads((OUT/'delivery_review.json').read_text())
    assert not (OUT/'kernel_review.json').exists()
    assert all(sha(ROOT/f)==h for f,h in {**p['source_sha256'],**p['evidence_sha256']}.items())
    assert old['fixture_sha256']==sha(OUT/'delivery_fixtures.npz')
    with np.load(OUT/'delivery_fixtures.npz') as z:indices=z['indices'];expected_latent=z['filtered_latent'];canonical=z['canonical'];motor=z['motor_request_Nm']
    parent=json.loads((OUT/'review.json').read_text());cache={};q=[];v=[];reference=[];hashes={}
    data=BASE/'reflection_retention_falsification_v1'
    for record in parent['records']:
        name=record['condition']
        if name not in cache:
            row=json.loads((data/name/'result.json').read_text())['runs'][0];f=data/name/row['actor_trace']['path']
            assert sha(f)==parent['input_sha256'][str(f.relative_to(ROOT))];hashes[str(f.relative_to(ROOT))]=sha(f)
            with np.load(f) as z:cache[name]=z['trace']
        actor=cache[name];frame=actor[record['actor_index'],351:390];reset=actor[0,351:390]
        q.append(frame[12:16]);v.append(frame[16:22]);reference.append(np.concatenate([leg_kinematics(reset[12+2*s:14+2*s].astype(float),np.zeros(2))[0] for s in range(2)]))
    q=np.array(q,np.float32)[indices[:,0]];v=np.array(v,np.float32)[indices[:,0]];reference=np.array(reference)[indices[:,0]]
    import itertools
    targets=np.array(list(itertools.product([-1.,0.,1.],repeat=3)),np.float32)[indices[:,2]]
    target=np.column_stack([targets[:,0],-targets[:,0],targets[:,1],-targets[:,1],targets[:,2],-targets[:,2]])
    previous=np.array([[0,0,0],[1,-1,1],[-1,1,-1],[.25,.25,-.25]],float)[indices[:,1]]
    n=len(q);assert n==55296;wp.init();wp.set_device('cuda:0');unit(q,v,reference)
    limits=bounds(v);reports=[];outputs={};tol=p['leaf_kernel_check']['cpu_tolerances'];count=0
    for mode in (0,1):
        for near_limit in (False,True):
            base=(.95*limits*np.array([1,-1,1,-1,1,-1]) if near_limit else np.zeros_like(limits)).astype(np.float32)
            got=launch(q,v,target,previous,reference,base,mode);t=got['trace'][0]
            np.testing.assert_array_equal(got['latent'],expected_latent)
            np.testing.assert_array_equal(got['shadow'][:,16::2],expected_latent)
            np.testing.assert_array_equal(got['shadow'][:,17::2],-expected_latent)
            assert not t[:,33].any() and np.all(t[:,0]==1)
            command_error=float(abs(t[:,7:13]-canonical[:,mode]).max());motor_error=float(abs(t[:,15:21]-motor[:,mode]).max())
            assert command_error<=tol['canonical_abs'] and motor_error<=tol['motor_request_Nm']
            np.testing.assert_allclose(t[:,27:33],limits,atol=1e-12,rtol=0)
            res=motor[:,mode];ratios=np.full_like(res,np.inf)
            np.divide(limits-base,res,out=ratios,where=res>0)
            np.divide(-limits-base,res,out=ratios,where=res<0)
            lam=np.clip(np.minimum(1,ratios.min(axis=1)),0,1)
            expected=np.clip(base.astype(float)+lam[:,None]*res,-limits,limits).astype(np.float32)
            issued_error=float(abs(got['ctrl']-expected).max());bound_error=float(np.maximum(abs(got['ctrl'])-limits,0).max())
            assert issued_error<=tol['issued_Nm'] and bound_error<=tol['bound_excess_Nm']
            cold=np.all(expected_latent==0,axis=1);np.testing.assert_array_equal(got['ctrl'][cold],base[cold])
            assert np.all((t[:,14]>=0)&(t[:,14]<=1))
            file=OUT/f'kernel_mode{mode}_near{int(near_limit)}.npz'
            np.savez_compressed(file,trace=t,expected_ctrl=expected,base=base)
            outputs[str(file.relative_to(ROOT))]=sha(file);count+=n
            reports.append(dict(mode=mode,near_limit=near_limit,rows=n,canonical_error=command_error,motor_request_error_Nm=motor_error,
                issued_error_Nm=issued_error,bound_excess_Nm=bound_error,projected_rows=int((t[:,14]<1-1e-12).sum()),cold_zero_rows=int(cold.sum())))
            print('KERNEL363',reports[-1],flush=True)
    assert count==p['leaf_kernel_check']['total_evaluations'] and any(r['projected_rows'] for r in reports)
    atomic_json(OUT/'kernel_review.json',dict(round=363,verified=True,evaluations=count,reports=reports,unit_passed=True,
        proposal_sha256=sha(OUT/'runtime_proposal.json'),fixture_sha256=sha(OUT/'delivery_fixtures.npz'),kernel_sha256=sha(ROOT/'wheelleg_warp/cartesian_pair_kernel.py'),
        checker_sha256=sha(__file__),input_sha256=hashes,outputs_sha256=outputs,runtime=runtime(),
        new_physics_steps=0,new_FD=0,new_model_forward_rows=0,new_learning_samples=0,complete_runtime_admitted=False,
        limits='Real CUDA request/VMC/common motor projection kernels on saved-state fixtures only. No original Nom, joint guard, parking or actual Native graph was composed/executed here; their full zero/nonzero admission remains required.'))


if __name__=='__main__':run()
