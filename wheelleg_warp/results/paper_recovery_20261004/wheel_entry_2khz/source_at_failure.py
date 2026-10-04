"""Read-only 2kHz wheel geometry added to the frozen public lateral experiment."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco
import warp as wp
from native.controller import D
from native.environment import after,wheel_center
from training_contract import digest
from dashboard.live_env import atomic_json
import test_lateral_intervention as experiment

COL=['valid','time_s','body_x_m','body_y_m','yaw_rad','left_x_m','left_y_m','left_z_m',
     'right_x_m','right_y_m','right_z_m','wheel_support_x_m','wheel_support_y_m','contact_mask','cumulative_contact_mask','body_z_m']

@wp.kernel
def record(slot:int,q:wp.array2d[float],flags:wp.array2d[int],ids:wp.array[int],state:wp.array2d[D],
           active:wp.array[int],offsets:wp.array3d[wp.vec3d],out:wp.array3d[D]):
    w=wp.tid();out[slot,w,0]=D(active[w])
    if active[w]==0:return
    out[slot,w,1]=(state[w,0]+D(1))*D(.0005)
    out[slot,w,2]=D(q[w,0]);out[slot,w,3]=D(q[w,1]);out[slot,w,15]=D(q[w,2])
    qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
    out[slot,w,4]=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    rotation=wp.quatd(qx,qy,qz,qw)
    axis=wp.quat_rotate(rotation,wp.vec3d(D(0),D(1),D(0)))
    out[slot,w,11]=wp.sqrt(D(.05*.05)+D(.0275*.0275-.05*.05)*axis[0]*axis[0])
    out[slot,w,12]=wp.sqrt(D(.05*.05)+D(.0275*.0275-.05*.05)*axis[1]*axis[1])
    for side in range(2):
        center=wheel_center(q,ids,offsets,w,side)
        for k in range(3):out[slot,w,5+3*side+k]=center[k]
    out[slot,w,13]=D(flags[w,1]);out[slot,w,14]=D(int(state[w,14])|flags[w,1])

def run(out):
    original_factory=experiment.raw_env;original_launch=wp.launch
    chunks=[];terminal_errors=[];terminal_q=[]
    def factory(cases,mode):
        n=len(cases);buffer=wp.zeros((40,n,len(COL)),dtype=D);slot=0
        atomic_json(out/'telemetry_registration.json',dict(sample_hz=2000,cases='Same 36 public conditions as round92; fresh replay, not independent validation',
            injection='Read-only kernel immediately before native.after, after physics/contact reduction/physical evidence; no changes to physics or controller order',
            columns=COL,geometry='Actual passive chain wheel_center; axisymmetric ellipsoid world support, original identity geom quaternion and sizes checked',
            source_sha256=digest(__file__),experiment_sha256=digest(Path(experiment.__file__)),training_updates=0,old_gate_or_final_replayed=False))
        def launch(kernel,dim,inputs=None,**kwargs):
            nonlocal slot
            if kernel is after:
                assert len(inputs)==22 and slot<40
                original_launch(record,n,[slot,inputs[0],inputs[5],inputs[6],inputs[9],inputs[13],inputs[21],buffer])
                slot+=1
            return original_launch(kernel,dim,inputs,**kwargs)
        wp.launch=launch
        try:env=original_factory(cases,mode)
        finally:wp.launch=original_launch
        assert slot==40
        for side in ['L','R']:
            g=env.cpu.geom('wheel_collide_'+side)
            np.testing.assert_array_equal(g.quat,[1,0,0,0]);np.testing.assert_array_equal(g.size,[.05,.0275,.05])
        chunks.extend([[] for _ in cases]);pending=set(range(n));wait=env.step_wait
        cpu_data=mujoco.MjData(env.cpu)
        def step_wait():
            answer=wait();data=buffer.numpy();stopped=env.stopped_q.numpy()
            for w in list(pending):
                valid=data[:,w,0]>0;trace=data[valid,w].copy();assert len(trace)>0
                chunks[w].append(trace)
                if answer[2][w]:
                    cpu_data.qpos[:]=stopped[w];mujoco.mj_forward(env.cpu,cpu_data)
                    expected=np.concatenate([cpu_data.geom_xpos[env.cpu.geom('wheel_collide_'+s).id] for s in ['L','R']])
                    error=float(np.max(abs(trace[-1,5:11]-expected)));terminal_errors.append(error)
                    assert error<1e-7,'Native wheel chain disagrees with independent CPU forward'
                    terminal_q.append(dict(world=w,qpos=stopped[w].tolist(),maximum_wheel_error_m=error))
                    pending.remove(w)
            return answer
        env.step_wait=step_wait
        return env
    experiment.raw_env=factory
    try:experiment.run(out)
    finally:experiment.raw_env=original_factory
    arrays=[np.concatenate(c) for c in chunks];rows=json.loads((out/'result.json').read_text())['runs'];details=[]
    for row,t in zip(rows,arrays):
        assert len(t)==row['physical_steps']==row['physical_evidence_steps']
        np.testing.assert_allclose(t[:,1],np.arange(1,len(t)+1)*.0005,atol=1e-9,rtol=0)
        assert np.isfinite(t).all() and int(t[-1,14])&12==row['touched_terrain_contact_mask']
        side_details=[];direction=row['condition']['direction']
        for side in range(2):
            x=t[:,5+3*side];y=t[:,6+3*side];long_overlap=(direction*x+t[:,11]>=1.55)&(direction*x-t[:,11]<=2.05)
            clearance=.16+t[:,12]-abs(y)
            touches=np.flatnonzero(t[:,13].astype(int)&(4<<side))
            assert long_overlap.any()
            side_details.append(dict(side=['L','R'][side],first_terrain_contact_s=float(t[touches[0],1]) if len(touches) else None,
                max_lateral_overlap_during_longitudinal_overlap_m=float(clearance[long_overlap].max()),
                first_longitudinal_overlap_s=float(t[np.flatnonzero(long_overlap)[0],1]),
                last_longitudinal_overlap_s=float(t[np.flatnonzero(long_overlap)[-1],1])))
        details.append(dict(seed=row['seed'],condition=row['condition'],success=row['success'],wheels=side_details))
    offsets=np.cumsum([0]+[len(t) for t in arrays])
    np.savez_compressed(out/'wheel_trace.npz',columns=np.array(COL),offsets=offsets,trace=np.concatenate(arrays))
    atomic_json(out/'wheel_verification.json',dict(verified=True,episodes=len(rows),physics_samples=int(offsets[-1]),
        terminal_CPU_wheel_error_m=max(terminal_errors),terminal_states=terminal_q,details=details,
        scope='Negative lateral overlap excludes geometric contact during longitudinal AABB overlap; positive overlap is necessary, not sufficient, for loaded contact. Contact bits retain native broad contact semantics.',
        source_sha256=digest(__file__),experiment_sha256=digest(Path(experiment.__file__))))
    print('PASS actual 2kHz wheels',len(rows),'episodes',int(offsets[-1]),'samples; CPU terminal error',max(terminal_errors),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
