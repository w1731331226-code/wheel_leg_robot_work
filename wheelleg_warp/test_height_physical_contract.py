"""Actual GPU safety predicates against archived false-safe frames and CPU sites."""
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
from native.environment import NativeEnv,collect_physical,physical_passed,after,D
from native.terrain import HeightTerrainScenario,bank_height_115,HEIGHT_115_GEOMETRIC_MIN as LIMIT
from probe_height_115_passive import geometry


@wp.kernel
def verdict(state:wp.array2d[D],out:wp.array[int]):
    w=wp.tid();out[w]=int(physical_passed(state,D(LIMIT),w))


def run(output):
    assert not output.exists(),output
    folder=ROOT/'wheelleg_warp/results/height_115_passive_20260928'
    meta=json.loads((folder/'verification.json').read_text())
    trace=folder/'passive_trace.npz'
    assert hashlib.sha256(trace.read_bytes()).hexdigest()==meta['trace_sha256']
    cases=[HeightTerrainScenario(**row['scenario']) for row in meta['rows']]
    env=NativeEnv(n=len(cases),scenario=cases,bank_factory=bank_height_115,
                  height_conditioned=True,height_design='range115',residual_scale=0)
    try:
        assert env.height_safety=='physical_v1'
        with np.load(trace) as z:
            false=(z['fk_length_m']>=LIMIT)&(z['actual_length_m']<LIMIT)&z['live_sample'][...,None]
            indices=np.argwhere(false);assert len(indices)==352
            q=z['qpos'][indices[:,0],indices[:,1]]
            expected_A=z['actual_length_m'][indices[:,0],indices[:,1]].min(axis=1)
        worlds=indices[:,1];n=len(worlds)
        state=np.zeros((n,38));state[:,0]=1;state[:,30:33]=1;state[:,33]=1e30
        s=wp.array(state,dtype=D);out=wp.zeros(n,dtype=wp.int32)
        args=[wp.array(q,dtype=wp.float32),wp.zeros((n,env.cpu.nv)),wp.zeros((n,6)),wp.zeros((n,6)),
              wp.ones(n,dtype=wp.int32),env.ids,env.physical_args[6],
              wp.array(env.physical_args[7].numpy()[worlds],dtype=D),
              wp.array(env.physical_args[8].numpy()[worlds],dtype=wp.vec3d),s]
        wp.launch(collect_physical,n,args);wp.launch(verdict,n,[s,out])
        assert not out.numpy().any(),'真实A越线帧被接受'
        got=s.numpy();aerror=float(abs(got[:,31]-expected_A).max())
        assert aerror<1e-7,aerror
        # CPU site reconstruction of both chains, with actual batched float32 offsets.
        cpu=mujoco.MjData(env.cpu);site_error=0.
        for i in (0,n//2,n-1):
            cpu.qpos[:]=q[i];mujoco.mj_kinematics(env.cpu,cpu)
            aa=[];bb=[];closure=[]
            for side in ('L','R'):
                middle=(cpu.xpos[env.cpu.body('leg'+side).id]+cpu.xpos[env.cpu.body('leg'+side+'_D').id])/2
                a=cpu.site_xpos[env.cpu.site('wheel_axle_'+side).id]
                b=cpu.site_xpos[env.cpu.site('couplerB_'+side+'_end').id]
                aa.append(np.linalg.norm(a-middle));bb.append(np.linalg.norm(b-middle));closure.append(np.linalg.norm(a-b))
            site_error=max(site_error,abs(got[i,31]-min(aa)),abs(got[i,32]-min(bb)),abs(got[i,34]-max(closure)))
        assert site_error<1e-7,site_error
        # Check force evidence from a real actuator, not an assumed ctrl==force.
        m=env.cpu;data=mujoco.MjData(m);wid=m.actuator('motor_wheelL').id
        saved=m.actuator_gainprm[wid,0];m.actuator_gainprm[wid,0]=1.05
        data.qpos[:]=env.q0.numpy()[0];data.ctrl[wid]=4.5;mujoco.mj_forward(m,data)
        actual=float(data.actuator_force[wid]);assert abs(actual-4.725)<1e-12
        m.actuator_gainprm[wid,0]=saved
        # Production after() must reject each failure, and cannot accept missing monitor samples.
        env.reset();q=env.data.qpos.numpy();q[:]=env.q0.numpy()[0]
        q[1]=q[0];q[1,env.physical_args[6].numpy()[2]]-=.01
        q[2,env.physical_args[6].numpy()[0]]=1.5001
        metrics=geometry(m,q);assert metrics[1][1].min()<LIMIT
        state=env.state.numpy();state[:,0]=3999;state[:,1]=0;state[:,5]=1
        state[:,2:4]=q[:,:2];state[:,37]=3999
        f=np.zeros((len(cases),6),np.float32);u=f.copy();f[3,wid]=actual;u[3,wid]=4.5
        env.data.qpos.assign(q);env.stopped_q.assign(q);env.state.assign(state)
        env.data.actuator_force.assign(f);env.data.ctrl.assign(u)
        wp.launch(collect_physical,len(cases),env.physical_args)
        state=env.state.numpy();state[4,37]=3999;env.state.assign(state)
        after_args=[env.data.qpos,env.data.qvel,env.data.sensordata,env.data.qacc_warmstart,env.data.time,
            env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,env.residual,
            env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets]
        # Use legacy flat completion evidence in this terminal fixture; physical conditions are unchanged.
        p=env.param.numpy();p[:,5:8]=0;env.param.assign(p)
        wp.launch(after,len(cases),after_args)
        terminal=env.state.numpy();assert terminal[0,19]==1 and np.all(terminal[1:5,19]==0)
        _,_,done,infos=env.step_wait();assert done.all()
        assert infos[0]['physical_safety_passed'] and all(not x['physical_safety_passed'] for x in infos[1:5])
        assert infos[3]['max_actual_torque_excess_Nm']>.2249
        assert infos[4]['physical_evidence_steps']<infos[4]['physical_steps']
        assert np.all(env.state.numpy()[:,37]==0),'重置未清除证据'
        # Real captured graph includes the monitor once per active physical substep.
        env.reset();env.step(np.zeros((len(cases),3),np.float32));state=env.state.numpy()
        np.testing.assert_array_equal(state[:,37],state[:,0]);assert np.all(state[:,0]==40)
        result=dict(passed=True,height_safety_contract=env.height_safety,false_safe_leg_frames_rejected=n,
            actual_A_archive_max_error_m=aerror,cpu_both_chain_site_max_error_m=float(site_error),
            terminal_cases=['valid','actual_A_crossing','joint_crossing','actual_force_excess','missing_evidence'],
            actual_force_counterexample_Nm=actual,terminal_success=terminal[:5,19].tolist(),
            graph_evidence_steps=state[:,37].tolist(),reset_clears_evidence=True,closure_is_recorded_not_certified=True,
            source_trace_sha256=meta['trace_sha256'],
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/controller.py')})
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('PASS',json.dumps(result,ensure_ascii=False),flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
