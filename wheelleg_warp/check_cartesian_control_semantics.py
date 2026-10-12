"""Clone complete saved inputs through actual cold-zero control kernels; no solver."""
import json
from unittest.mock import patch
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
import wheelleg_sim as sim
import phase_support_probe as phase
import parking_withdrawal_probe as parking
import joint_state_guard as guard
from cartesian_pair_runtime import NOMINAL
from cartesian_pair_kernel import clear_native,reset_aux
from cartesian_pair_kernel_v2 import deliver_accepted_nominal
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'
OUT=BASE/'cartesian_training_interface_v2'


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def require_cold(state,latent,shadow,request):
    if any(np.any(x!=0) for x in (state[:,16:22],latent,shadow[:,16:22],request)):
        raise ValueError('Cold neutral requires zero residual memories, not only zero request')


def branch(fixture,table,ids,mode):
    n=20;require_cold(fixture['controller_memory'],np.zeros((n,3)),np.zeros((n,22)),np.zeros((n,6)))
    array=lambda x:wp.array(x,dtype=wp.float32)
    double=lambda x:wp.array(x,dtype=wp.float64)
    q,v,sensor=[array(fixture[k]) for k in ('q','v','sensors')]
    state=double(fixture['controller_memory']);envstate=double(fixture['state']);active=wp.ones(n,dtype=int);index=wp.array(ids,dtype=int)
    target=wp.zeros((n,6));command=wp.zeros(n,dtype=wp.float64);ctrl=array(fixture['ctrl']);diag=wp.zeros((n,38),dtype=wp.float64)
    base_ref=double(table['reference']);reference=wp.zeros((n,16),dtype=wp.float64)
    role=wp.zeros((1,n,11),dtype=wp.float64);monitor=wp.zeros((1,n,6),dtype=wp.float64)
    seen=wp.zeros(n,dtype=int);effective=wp.zeros((n,6));log=wp.zeros((1,n,30),dtype=wp.float64)
    latent=wp.zeros((n,3),dtype=wp.float64);shadow=wp.zeros((n,22),dtype=wp.float64);reset_ref=wp.zeros((n,4),dtype=wp.float64)
    cart_log=wp.zeros((1,n,38),dtype=wp.float64);guard_memory=double(fixture['guard_memory']);guard_log=wp.zeros((1,n,62),dtype=wp.float64)
    mask=wp.ones(n,dtype=int);public=array(table['reset_public'])
    wp.launch(reset_aux,n,[mask,public,latent,shadow,reset_ref])
    wp.launch(phase.roles.prepare,n,[0,0,base_ref,q,sensor,index,state,envstate,active,reference,role])
    wp.launch(phase.select_anchor,n,[0,1,command,active,reference,role,monitor])
    parking_state=state if mode is None else shadow
    wp.launch(parking.prepare,n,[0,1,command,active,envstate,parking_state,target,seen,effective,log])
    if mode is not None:wp.launch(clear_native,n,[active,state])
    heights=double([.115,sim.L_SQUAT_MIN,sim.L_PREP,sim.L_STAND,sim.L_MAX])
    gains=double(table['gains']);feed=double(table['feed']);angles=double(table['angles']);yaw=double([.4,2.,.24,.3])
    upper=double(np.tile([1,1,1,1,1.05,1.05],(n,1)));correction=wp.zeros((n,6),dtype=wp.float64)
    wp.launch(NOMINAL,n,[q,v,sensor,effective,command,active,state,index,heights,gains,feed,angles,reference,yaw,ctrl,diag,0,0,upper,correction],block_dim=32)
    nominal={k:b.numpy().copy() for k,b in dict(state=state,ctrl=ctrl,diag=diag,reference=reference).items()}
    if mode is not None:
        wp.launch(deliver_accepted_nominal,n,[0,mode,q,v,index,effective,active,latent,reset_ref,state,shadow,ctrl,diag,cart_log])
    wp.launch(guard.guard,n,[0,q,v,active,index,ctrl,diag,guard.V6(1.,1.,1.,1.,1.05,1.05),guard_memory,guard_log])
    wp.launch(parking.finish,n,[0,parking_state,diag,log]);wp.launch(phase.roles.finish,n,[0,diag,role])
    result={k:b.numpy().copy() for k,b in dict(state=state,ctrl=ctrl,diag=diag,reference=reference,role=role,monitor=monitor,
        parking=log,guard_memory=guard_memory,guard=guard_log,q=q,v=v,sensors=sensor).items()}
    for k in ('q','v','sensors'):np.testing.assert_array_equal(result[k],fixture[k])
    assert not result['diag'][:,14].any()
    if mode is not None:
        values=cart_log.numpy();np.testing.assert_array_equal(values[:,:,7:13],0);np.testing.assert_array_equal(values[:,:,14],1)
        np.testing.assert_array_equal(latent.numpy(),0);np.testing.assert_array_equal(shadow.numpy()[:,16:22],0)
        result['candidate']=values
    return nominal,result


def run():
    p=json.loads((OUT/'proposal.json').read_text());semantic=p['semantic_gate'];assert not (OUT/'semantic_review.json').exists()
    assert semantic['fixture_rows']==20 and semantic['control_rows']==60 and semantic['solver_steps']==semantic['FD_calls']==0
    for name,digest in semantic['inputs_sha256'].items():assert sha(ROOT/name)==digest
    source=BASE/'cartesian_zero_input_diagnostic_v1/runtime/B0_repeat1'
    initial=load(source/'initial.npz');terminal=load(source/'terminal_before_wait.npz');end=load(source/'zero_prefix.npz');model=load(source/'constructor_stage1.npz')
    fixture={k:np.concatenate((initial[k],terminal[k])) for k in ('q','v','sensors','controller_memory','state')}
    fixture['ctrl']=np.concatenate((initial['ctrl'],end['ctrl']));fixture['guard_memory']=np.concatenate((initial['guard_memory'],end['guard_memory']))
    assert np.all(fixture['state'][:,1]<0) and not fixture['controller_memory'][:,26:30].any()
    cpu=mujoco.MjModel.from_xml_path(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'))
    np.testing.assert_array_equal(cpu.jnt_qposadr,model['model.jnt_qposadr']);np.testing.assert_array_equal(cpu.jnt_dofadr,model['model.jnt_dofadr'])
    ids=[int(cpu.jnt_qposadr[cpu.joint(n).id]) for n in ('alphaL','betaL','alphaR','betaR')]
    ids += [int(cpu.jnt_dofadr[cpu.joint(n).id]) for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]+[int(cpu.sensor('body_gyro').adr[0])]
    assert ids==[7,10,12,15,6,9,11,14,8,13,4]
    table={k:model[k] for k in ('gains','feed','angles')};table['reference']=np.tile(model['reference'],(2,1))
    table['reset_public']=np.tile(initial['public'],(2,1))
    wp.init();wp.set_device('cuda:0');records={};baseline=None
    def forbidden(*args,**kwargs):raise AssertionError('Physics/FD forbidden in semantic gate')
    with patch.object(mjw,'step',forbidden),patch.object(mujoco,'mj_step',forbidden),patch.object(mujoco,'mjd_transitionFD',forbidden):
        for arm,mode in [('B0',None),('CartLive3',0),('CartReset3',1)]:
            nominal,result=branch(fixture,table,ids,mode)
            np.savez_compressed(OUT/f'semantic_{arm}.npz',**result,**{f'Nom_{k}':v for k,v in nominal.items()})
            if baseline is None:baseline=(nominal,result)
            else:
                for key in nominal:np.testing.assert_array_equal(nominal[key],baseline[0][key],err_msg=arm+' Nom '+key)
                for key in baseline[1]:np.testing.assert_array_equal(result[key],baseline[1][key],err_msg=arm+' final '+key)
            records[arm]=dict(rows=20,path=f'semantic_{arm}.npz',sha256=sha(OUT/f'semantic_{arm}.npz'))
    bad=fixture['controller_memory'].copy();bad[0,16]=.1
    try:require_cold(bad,np.zeros((20,3)),np.zeros((20,22)),np.zeros((20,6)))
    except ValueError:pass
    else:raise AssertionError('Zero request with old latent misclassified as neutral')
    atomic_json(OUT/'semantic_review.json',dict(round=373,verified=True,control_rows=60,records=records,
        exact_common_Nom_guard_final_outputs=True,fixture_inputs_sha256=semantic['inputs_sha256'],proposal_sha256=sha(OUT/'proposal.json'),
        checker_sha256=sha(__file__),new_physics_steps=0,new_FD=0,new_model_forward_rows=0,new_learning_samples=0,
        scope='20 saved-input cold-neutral rows through actualphase/parking/Nom/candidate/guard;CPU model only parsedforids,noData/solver. Not nonzero/oldlatent/curriculum/physics-path equivalence or PPO.',
        old_failures_unchanged=True,trajectory_runtime_admitted=False,formal5_admitted=False))
    print('SEMANTIC373 60actual CUDA control rows;all sharedNom/guard/float32ctrl exact;0solver/FD/learning',flush=True)


if __name__=='__main__':run()
