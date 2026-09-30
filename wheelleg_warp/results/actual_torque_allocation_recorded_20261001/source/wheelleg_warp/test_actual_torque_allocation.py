"""Check actual-gain allocation with CPU forces and both captured GPU graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
import warp as wp
import wheelleg_sim as sim
from native.controller import control
from native.environment import NativeEnv
from native.live import LiveNativeEnv
from native.terrain import HeightTerrainScenario,bank_height_115,model
from trace_failure_chain import RecordedEnv


def launch(env,kernel,extra):
    wp.launch(kernel,env.num_envs,[env.data.qpos,env.data.qvel,env.data.sensordata,env.targets,env.command,
        env.active,env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],
        env.k['reference'],env.k['yaw'],env.data.ctrl,env.diag,0,0]+extra,block_dim=32)


def run(output):
    assert not output.exists(),output
    cases=[HeightTerrainScenario(stand_height_m=.3,drive_difference=d) for d in (0.,.02,.04,.05)]
    env=NativeEnv(n=4,scenario=cases,bank_factory=bank_height_115,height_conditioned=True,height_design='range115')
    cpu=[model(s) for s in cases];rows=[];old_excess=0.;new_excess=0.;unit_error=0.
    try:
        for rpm in (0.,600.,800.,-600.):
            env.reset();v=env.data.qvel.numpy()
            v[:,env.ids.numpy()[8:10]]=rpm*2*np.pi/60
            env.data.qvel.assign(v);env.command.assign(np.full(4,50.))
            env.targets.assign(np.tile([.5,-.5,.5],(4,1)))
            state=env.k['state'].numpy();state[:,0]=2.;env.k['state'].assign(state)
            launch(env,control,[]);old=env.data.ctrl.numpy().copy();old_diag=env.diag.numpy().copy()
            # Unit gain must preserve the full old command and diagnostic semantics.
            env.k['state'].assign(state);launch(env,env.control_kernel,[wp.array(np.ones((4,6)),dtype=wp.float64)])
            unit_error=max(unit_error,float(abs(env.data.ctrl.numpy()-old).max()),float(abs(env.diag.numpy()-old_diag).max()))
            env.k['state'].assign(state);launch(env,env.control_kernel,env.control_extra)
            new=env.data.ctrl.numpy();bound=sim.hw.torque_limit(float('inf'),rpm*2*np.pi/60,False,0.,.0005)[0]
            for w,m in enumerate(cpu):
                d=mujoco.MjData(m);d.qpos[:]=env.q0.numpy()[w];d.qvel[:]=v[w]
                d.ctrl[:]=old[w];mujoco.mj_forward(m,d);old_force=d.actuator_force.copy()
                d.ctrl[:]=new[w];mujoco.mj_forward(m,d);actual=d.actuator_force.copy()
                old_excess=max(old_excess,float((abs(old_force[4:])-bound).max()))
                new_excess=max(new_excess,float((abs(actual[4:])-bound).max()))
                assert np.max(abs(new[w,4:])-bound)<=1e-6
                assert np.max(abs(actual[4:])-bound)<=1e-6
                np.testing.assert_allclose(new[w],env.diag.numpy()[w,:6]+env.diag.numpy()[w,6:12],atol=3e-7,rtol=0)
                rows.append(dict(rpm=rpm,drive_difference=cases[w].drive_difference,bound_Nm=bound,
                    old_command=old[w,4:].tolist(),old_actual=old_force[4:].tolist(),
                    new_command=new[w,4:].tolist(),new_actual=actual[4:].tolist()))
        assert old_excess>.22 and unit_error==0.,(old_excess,unit_error)
        env.reset();env.step(np.array([[.4,-.2,.3]]*4,np.float32));s=env.state.numpy()
        np.testing.assert_array_equal(s[:,37],s[:,0]);assert np.max(s[:,35:37])<=1e-6
        # The live recorder rebuilds the CUDA graph; it must carry the same allocation and monitor.
        live=LiveNativeEnv(directory=output.parent/'live',n=4,scenario=cases,bank_factory=bank_height_115,
            height_conditioned=True,height_design='range115')
        try:
            live.reset();live.step(np.zeros((4,3),np.float32));ls=live.state.numpy()
            np.testing.assert_array_equal(ls[:,37],ls[:,0]);assert np.all(ls[:,37]==40)
            assert np.max(ls[:,35:37])<=1e-6
            assert live.frame_metadata(0)['control_limit_scope']=='actual_torque_and_nominal_command'
        finally:live.close()
        recorded=RecordedEnv(4,scenario=cases,height_conditioned=True,height_design='range115')
        try:
            recorded.reset();recorded.step(np.zeros((4,3),np.float32));rs=recorded.state.numpy()
            np.testing.assert_array_equal(rs[:,37],rs[:,0]);assert np.all(rs[:,37]==40)
            assert np.max(rs[:,35:37])<=1e-6
            assert np.all(recorded.trace.numpy()[:,:,0]==1)
        finally:recorded.close()
        result=dict(passed=True,rows=rows,old_max_actual_excess_Nm=old_excess,new_max_actual_excess_Nm=new_excess,
            unit_gain_command_and_diagnostic_max_error=unit_error,native_live_recorded_evidence_steps=40,
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/live.py',ROOT/'wheelleg_warp/trace_failure_chain.py')})
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('PASS',old_excess,new_excess,unit_error,flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
