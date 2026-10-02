"""Raw motor mapping, shared zero-action physics, own-request timing and B1 reuse."""
from pathlib import Path
from dataclasses import replace
import argparse,json
import numpy as np
import warp as wp
from native.environment import NativeEnv,reset_rows
from probe_height_115_margin import cases
from pretrain_yaw import b1_action
from training_contract import observation_spec,source_hashes
import wheelleg_sim as sim


def check(output):
    output.mkdir(parents=True,exist_ok=False)
    scenes=[replace(s,delay_ms=20.) for s in cases()[:6]]
    envs=[NativeEnv.height115_candidate(n=6,scenario=scenes,residual_mode=m,shared_reference=True)
          for m in ('diff3','virtual6','torque6')]
    raw=envs[-1];mapping_error=0.;zero_error=0.;command_error=0.;same_state_error=0.;same_state_checks=0;rows=[]
    try:
        assert raw.observation_spec['request_state_channels'][:4]==['alpha_left','beta_left','alpha_right','beta_right']
        assert observation_spec('request_state_v1',3)==observation_spec('request_state_v1',6)
        for rpm in (0.,600.,800.,-600.):
            for sign in (-1.,1.):
                for channel in range(6):
                    raw.reset();a=np.zeros((6,6),np.float32);a[:,channel]=sign*.2
                    v=raw.data.qvel.numpy();v[:,raw.ids.numpy()[8:10]]=rpm*np.pi/30;raw.data.qvel.assign(v)
                    state=raw.k['state'].numpy();state[:,16:22]=a;raw.k['state'].assign(state);raw.targets.assign(a)
                    d=raw.data
                    wp.launch(raw.control_kernel,6,[d.qpos,d.qvel,d.sensordata,raw.targets,raw.command,raw.active,raw.k['state'],raw.ids,
                        raw.k['heights'],raw.k['gains'],raw.k['feed'],raw.k['angles'],raw.k['reference'],raw.k['yaw'],d.ctrl,raw.diag,0,0]+raw.control_extra,block_dim=32)
                    diag=raw.diag.numpy();request=a.astype(float)*[1.,1.,1.,1.,.3,.3]
                    bounds=np.array([sim.hw.torque_limit(float('inf'),v[w,j],i<4,0.,.0005)[0]/raw.actuator_gain_upper[i]
                        for w in range(6) for i,j in enumerate(raw.ids.numpy()[4:10])]).reshape(6,6)
                    lam=np.ones(6)
                    for j in range(6):
                        if request[0,j]>0:lam=np.minimum(lam,(bounds[:,j]-diag[:,j])/request[:,j])
                        elif request[0,j]<0:lam=np.minimum(lam,(-bounds[:,j]-diag[:,j])/request[:,j])
                    lam=np.clip(lam,0.,1.)
                    np.testing.assert_allclose(diag[:,12],lam,atol=1e-12,rtol=0)
                    error=float(abs(diag[:,6:12]-lam[:,None]*request).max());mapping_error=max(mapping_error,error)
                    assert error<1e-12 and np.all(diag[:,14]==0)
                    np.testing.assert_allclose(d.ctrl.numpy(),diag[:,:6]+diag[:,6:12],atol=3e-7,rtol=0)
        initial=raw.reset();a=np.tile([.8,-.6,.2,-.3,.1,-.1],(6,1)).astype(np.float32)
        obs,_,done,_=raw.step(a);assert not done.any()
        np.testing.assert_array_equal(obs[:,:32],initial[:,:32])
        np.testing.assert_allclose(obs[:,32:],np.tile([.4,-.4,.2,-.3,.1,-.1],(6,1)),atol=1e-7,rtol=0)
        mask=np.zeros(6,np.int32);mask[0]=1;raw.mask.assign(mask);wp.launch(reset_rows,6,raw.reset_args)
        assert np.all(raw.k['state'].numpy()[0,16:]==0) and np.any(raw.k['state'].numpy()[1,16:22]!=0)
        # The frozen B1 formula reads the same delayed roll/yaw/gyro fields in either packet.
        fixture=np.zeros(38,np.float32);fixture[[0,2,5]]=[.04,.1,.2]
        candidate=dict(kp=.8,kd=.6,roll_gain=1)
        np.testing.assert_array_equal(b1_action(fixture,candidate),b1_action(fixture[:32],candidate))
        assert b1_action(fixture,candidate)[2]<0
        for e in envs:e.reset()
        finished=[set() for _ in envs]
        for step in range(1000):
            if step%50==0:
                base=envs[0];s=base.k['state'].numpy();commands=[]
                for e in envs:
                    memory=wp.array(np.c_[s[:,:16],np.zeros((6,e.action_dim)),s[:,19:]],dtype=wp.float64)
                    ctrl=wp.zeros((6,6));diag=wp.zeros((6,38),dtype=wp.float64)
                    wp.launch(e.control_kernel,6,[base.data.qpos,base.data.qvel,base.data.sensordata,wp.zeros((6,e.action_dim)),
                        base.command,base.active,memory,e.ids,e.k['heights'],e.k['gains'],e.k['feed'],e.k['angles'],
                        base.k['reference'],e.k['yaw'],ctrl,diag,0,0,e.control_extra[0],base.nominal_correction],block_dim=32)
                    commands.append(ctrl.numpy())
                    np.testing.assert_array_equal(diag.numpy()[:,6:12],0)
                same_state_error=max(same_state_error,max(float(abs(c-commands[0]).max()) for c in commands[1:]))
                assert same_state_error<1e-5;same_state_checks+=6
            packets=[e.step(np.zeros((6,e.action_dim),np.float32)) for e in envs]
            comparable=np.array([w not in set.union(*finished) and not any(p[2][w] for p in packets) for w in range(6)])
            for e in envs[1:]:
                if comparable.any():
                    error=float(abs(e.data.qpos.numpy()[comparable]-envs[0].data.qpos.numpy()[comparable]).max());zero_error=max(zero_error,error)
                    command_error=max(command_error,float(abs(e.data.ctrl.numpy()[comparable]-envs[0].data.ctrl.numpy()[comparable]).max()))
                np.testing.assert_allclose(e.diag.numpy()[:,6:12],0,atol=0,rtol=0)
            for k,(_,_,done,infos) in enumerate(packets):
                for w in np.flatnonzero(done):
                    if w not in finished[k]:rows.append(dict(mode=envs[k].residual_mode,world=int(w),**{x:y for x,y in infos[w].items() if x!='terminal_observation'}));finished[k].add(w)
            if all(len(s)==6 for s in finished):break
        assert len(rows)==18 and all(r['success'] and r['design_joint_passed'] and r['physical_safety_passed'] for r in rows)
        for e in envs:assert np.all(e.reset()[:,32:]==0)
        for bad in (np.zeros((6,3),np.float32),np.full((6,6),np.nan),np.full((6,6),1.1)):
            try:raw.step(bad)
            except ValueError:pass
            else:raise AssertionError('invalid action accepted')
        result=dict(passed=True,mapping_fixture_count=48*6,raw_mapping_max_error_Nm=mapping_error,
            independent_closed_loop_qpos_max_difference=zero_error,independent_closed_loop_command_max_difference_Nm=command_error,
            same_state_zero_actor_command_max_difference_Nm=same_state_error,same_state_checks=same_state_checks,
            normal_tasks=rows,physical_delay_and_current_own_requests=True,
            masked_and_full_reset=True,b1_shared_observation_formula=True,source_sha256=source_hashes(__file__))
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PASS torque6: independent motor mapping/limits, delayed38D packet/reset, B1 formula; shared zero-Actor18/18')
    finally:
        for e in envs:e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);check(p.parse_args().output)
