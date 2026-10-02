"""Same-current-state causal initialization comparison at the frozen failing request."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from native.environment import NativeEnv,begin,command_step,reduce_contacts,collect_physical,after
from native.controller import D
from forecaster import Forecaster,record_forecast
from probe_height_115_margin import cases
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent


def run():
    assert not (OUT/'verification.json').exists()
    source=OUT.parent/'twelfth_round_causal_allocation_past_init_20261002/verification.json'
    request=np.array(json.loads(source.read_text())['decisions'][0]['request'])
    assert abs(request).max()<=1 and np.sum(abs(request),axis=1).max()<=.1+1e-12
    env=NativeEnv.height115_candidate(n=2,scenario=cases()[4:6],residual_scale=0,nominal_correction=True)
    try:
        env.reset();d=env.data;raw=wp.zeros((10,2,env.cpu.nq+env.cpu.nv+12),dtype=D)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,2,[env.reward])
            for slot in range(10):
                wp.launch(command_step,2,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
                wp.launch(env.control_kernel,2,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
                mjw.step(env.model,d);wp.launch(record_forecast,2,[slot,d.qpos,d.qvel,d.ctrl,d.actuator_force,raw])
                wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags]);wp.launch(collect_physical,2,env.physical_args)
                wp.launch(after,2,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        for _ in range(1600):
            wp.capture_launch(capture.graph)
            if np.all(env.state.numpy()[:,1]>=0):break
            assert not env.done.numpy().any()
        else:raise RuntimeError('Original arrival budget')
        np.savez_compressed(OUT/'known_pre.npz',q=d.qpos.numpy(),v=d.qvel.numpy(),past_q=env.stopped_q.numpy(),past_v=env.stopped_v.numpy(),past_command=d.ctrl.numpy(),controller=env.k['state'].numpy(),sensor=d.sensordata.numpy(),task=env.state.numpy(),request=request)
        # Separate predictor instances; both read the same past/current inputs, before any future.
        p=Forecaster(env,candidate_count=13,nominal_boundary=True)
        a=p.forecast_nominal(env,np.repeat(request,13,axis=0))[:,::13].copy()
        b=p.forecast_with_past_state(env,np.repeat(request,13,axis=0))[:,::13].copy()
        env.set_nominal_correction(request);wp.capture_launch(capture.graph);actual=raw.numpy()
        rows=[]
        for label,pred in [('current_state_past_command_init',a),('past_state_own_step_init',b)]:
            qe=float(abs(actual[:,:,:p.nq]-pred[:,:,:p.nq]).max());ve=float(abs(actual[:,:,p.nq:p.nq+p.nv]-pred[:,:,p.nq:p.nq+p.nv]).max());ce=float(abs(actual[:,:,-12:-6]-pred[:,:,-12:-6]).max())
            rows.append(dict(mode=label,qpos=qe,qvel=ve,command=ce,initial_command_equal=bool(np.array_equal(actual[0,:,-12:-6],pred[0,:,-12:-6])),original_gate_pass=bool(qe<=2e-6 and ve<=1e-3 and ce<=1e-5)))
        np.savez_compressed(OUT/'pairing.npz',actual=actual,current_init=a,past_step_init=b)
        result=dict(role='same_frozen_request_same_current_state_numerical_history_comparison',rows=rows,original_gates=dict(qpos=2e-6,qvel=1e-3,command=1e-5),source_request_sha256=sha(source),
            no_actual_numerical_warmstart_copied=True,actual_future_read_only_after_predictions=True,default_promoted=False,learning_performed=False,
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),OUT/'forecaster.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
        (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',rows,flush=True)
    finally:env.close()


if __name__=='__main__':run()
