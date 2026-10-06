"""Static original-noop and declared-role checks; no physics integration."""
import json
import numpy as np
import warp as wp
import reference_role_probe as p
from test_nom_yaw_filter_probe import control_args


def check():
    proposal=json.loads((p.OUT/'proposal.json').read_text());cases=proposal['cases']
    owners=[];examples=[]
    for std in (0.,.02):
        for arm in ('original','floor_only','consistent_pair'):
            raw=p.instrument(cases,'diff3',arm,std)
            try:
                raw.reset();ref,log=raw._role_buffers;mode=raw._role_topology['mode']
                for degrees in (-5.,0.,5.):
                    raw.reset();q=raw.data.qpos.numpy();q[:,3]=np.cos(np.deg2rad(degrees)/2);q[:,4]=np.sin(np.deg2rad(degrees)/2);raw.data.qpos.assign(q)
                    args=control_args(raw)
                    # Stabilize finite-difference caches for this synthetic static query.
                    wp.launch(p.original.control_physical_nominal,raw.num_envs,args,block_dim=32)
                    memory=raw.k['state'].numpy();memory[:,0]=2.;memory[:,3:9]=0.;raw.k['state'].assign(memory)
                    arrays=[a for a in args if isinstance(a,wp.array)];saved=[a.numpy().copy() for a in arrays]
                    wp.launch(p.prepare,raw.num_envs,[0,mode,args[12],args[0],args[2],args[7],args[6],raw.state,args[5],ref,log])
                    for a,b in zip(arrays,saved):np.testing.assert_array_equal(a.numpy(),b)
                    refs=ref.numpy();assert not refs[:,10:15].any()  # No arrival/parking shape side effects.
                    altered=list(args);altered[12]=ref
                    wp.launch(p.experimental.control_physical_nominal,raw.num_envs,altered,block_dim=32)
                    wp.launch(p.finish,raw.num_envs,[0,raw.diag,log])
                    actual=[raw.k['state'].numpy().copy(),raw.diag.numpy().copy(),raw.data.ctrl.numpy().copy()]
                    for w in range(raw.num_envs):p.check_log(log.numpy()[:1,w],mode)
                    assert np.isfinite(actual[2]).all() and np.all(abs(actual[2][:,:4])<=40+1e-6) and np.all(abs(actual[2][:,4:])<=4.5+1e-6)
                    if arm=='original':
                        for a,b in zip(arrays,saved):a.assign(b)
                        wp.launch(p.original.control_physical_nominal,raw.num_envs,args,block_dim=32)
                        for a,b in zip(actual,[raw.k['state'].numpy(),raw.diag.numpy(),raw.data.ctrl.numpy()]):np.testing.assert_array_equal(a,b)
                    if std==0 and degrees==5:
                        for w in (0,4,16):
                            examples.append(dict(arm=arm,case=cases[w]['seed'],role_log=log.numpy()[0,w].tolist(),max_hip_command_Nm=float(abs(actual[2][w,:4]).max())))
                raw.reset();active=raw.active.numpy();active[0]=0;raw.active.assign(active)
                wp.launch(p.prepare,raw.num_envs,[0,mode,raw.k['reference'],raw.data.qpos,raw.data.sensordata,raw.ids,raw.k['state'],raw.state,raw.active,ref,log])
                assert log.numpy()[0,0,0]==0
                raw.reset();assert not ref.numpy().any() and not log.numpy().any() and not raw._role_frozen and not raw._role_chunks[0]
                assert not raw.state.numpy()[:,0].any() and not raw.data.time.numpy().any()
                owners.append(dict(**raw._role_topology,core=raw._complete_topology,gyro=raw._gyro_topology))
            finally:raw.close()
    p.rec.atomic_json(p.OUT/'unit.json',dict(verified=True,owners=owners,static_examples=examples,
        original_branch_exact_inputs_state_diag_control=True,only_declared_control_function_override=True,
        reference_shape_does_not_enable_arrival_or_parking=True,governor_pair_bounds_mean_shift_and_guard_role_checked=True,
        protected_original_arrays_not_changed_by_prepare=True,inactive_reset_owned_buffers_checked=True,
        new_physics_evaluations=0,training_updates=0,
        limits='6x20constructor/capture and static original-control queries at0/+/-5deg rootroll, not live dynamics/160collected episodes. Reference feasibility/motor bounds do not guarantee actualheight/attitude/design safety.'))
    print('PASS6x20 capture/static/noop/reference-role/reset;0physics evaluation',flush=True)


if __name__=='__main__':check()
