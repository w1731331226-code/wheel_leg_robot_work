"""Thirty static current-Nom queries;no physics stepping or gain changes."""
import json
from pathlib import Path
import numpy as np
import warp as wp
import light_phase_reference as light
from native.controller import D,fk,PI
from native.terrain import HeightTerrainScenario
from train_height_comparison import raw_env
from test_nom_yaw_filter_probe import control_args
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_command_compatibility_v1'


@wp.kernel
def initialize(q:wp.array2d[float],v:wp.array2d[float],ids:wp.array[int],
               memory:wp.array2d[D],integral:D):
    w=wp.tid()
    for j in range(memory.shape[1]):memory[w,j]=D(0)
    a=fk(D(q[w,ids[0]]),D(q[w,ids[1]]));b=fk(D(q[w,ids[2]]),D(q[w,ids[3]]))
    qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
    pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    memory[w,0]=D(2)
    memory[w,1]=(a[3]+b[3])/D(2)
    memory[w,2]=(a[2]+b[2])/D(2)+D(PI)/D(2)-pitch
    memory[w,7]=wp.cos(yaw)*D(v[w,0])+wp.sin(yaw)*D(v[w,1])
    memory[w,11]=integral;memory[w,12]=D(-1)


def run():
    assert not any((OUT/n).exists() for n in ('source_contract.json','completion.json','failure.json'))
    p=json.loads((OUT/'proposal.json').read_text());assert p['static_controller_queries']==30
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    q=[];v=[];required=[];cases=[]
    for i,r in enumerate(p['references']):
        path=ROOT/r['path'];assert sha(path)==r['sha256']
        with np.load(path,allow_pickle=False) as z:q.append(z['q'].copy());v.append(z['v'].copy());required.append(z['ctrl'].copy())
        cases.append(dict(seed=2660000+i,scenario=HeightTerrainScenario(speed=r['speed'],mass=7.,
             height_l=0.,height_r=0.,center=1.8,offset=0.,mu_l=.8,mu_r=.8,drive_difference=0.,
             delay_ms=0.,solver_iterations=100,stand_height_m=r['height']).__dict__))
    q=np.array(q);v=np.array(v);required=np.array(required)
    raw=light.instrument(raw_env,cases,'virtual6')
    assert raw._light_phase_topology['verified'] and raw._light_phase_topology['control_calls']==40
    assert raw.cpu.opt.timestep==.0005 and raw.cpu.opt.iterations==100 and int(raw.cpu.opt.integrator)==3
    sources={**p['source_sha256'],'wheelleg_warp/query_nominal_compatibility.py':sha(__file__),
             'wheelleg_warp/test_nom_yaw_filter_probe.py':sha(ROOT/'wheelleg_warp/test_nom_yaw_filter_probe.py')}
    atomic_json(OUT/'source_contract.json',dict(verified=True,proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,
        topology=raw._light_phase_topology,static_individual_query_budget=30,
        scope='Currentoriginalphasecontrol zeroActor;allreset/wiredinputsheldexceptstate11. Capturedgraphneverlaunched.'))
    records=[];queries=0;commands=[]
    try:
        for j,integral in enumerate(p['speed_integral_states']):
            raw.reset();raw.data.qpos.assign(q.astype(np.float32));raw.data.qvel.assign(v.astype(np.float32))
            raw.data.sensordata.zero_();raw.data.ctrl.zero_();raw.targets.zero_();raw.nominal_correction.zero_()
            raw.command.assign(np.array([r['speed'] for r in p['references']],float))
            state=raw.state.numpy();state[:,0]=4000;state[:,1]=-1;raw.state.assign(state)
            wp.launch(initialize,raw.num_envs,[raw.data.qpos,raw.data.qvel,raw.ids,raw.k['state'],D(integral)])
            light.prepare_step(raw,0);args=control_args(raw);args[12]=raw._light_phase_buffers[0]
            before={name:a.numpy().copy() for name,a in [('q',raw.data.qpos),('v',raw.data.qvel),('sensor',raw.data.sensordata),('environment_state',raw.state),('command',raw.command),('clock',raw.data.time)]}
            memory_before=raw.k['state'].numpy().copy();reference_before=args[12].numpy().copy()
            queries+=raw.num_envs
            wp.launch(light.phase.roles.experimental.control_physical_nominal,raw.num_envs,args,block_dim=32)
            wp.launch(light.phase.roles.finish,raw.num_envs,[0,raw.diag,raw._light_phase_buffers[1]])
            memory_after=raw.k['state'].numpy().copy();control=raw.data.ctrl.numpy().copy();diag=raw.diag.numpy().copy()
            assert np.isfinite(control).all() and np.isfinite(diag).all() and np.isfinite(memory_after).all()
            for name,a in [('q',raw.data.qpos),('v',raw.data.qvel),('sensor',raw.data.sensordata),('environment_state',raw.state),('command',raw.command),('clock',raw.data.time)]:
                np.testing.assert_array_equal(a.numpy(),before[name])
            np.testing.assert_array_equal(raw.targets.numpy(),0);np.testing.assert_array_equal(raw.nominal_correction.numpy(),0)
            path=OUT/f'integral_{j}.npz';np.savez_compressed(path,q=before['q'],v=before['v'],sensor=before['sensor'],command=before['command'],
                 memory_before=memory_before,memory_after=memory_after,reference=reference_before,environment_state=before['environment_state'],
                 ctrl=control,diag=diag,required_ctrl=required,ctrl_minus_required=control-required,
                 original_q=q,original_v=v,role=raw._light_phase_buffers[1].numpy(),phase=raw._light_phase_buffers[2].numpy(),integral=np.array(integral))
            commands.append(control)
            for i,r in enumerate(p['references']):
                records.append(dict(height=r['height'],speed=r['speed'],state11=integral,path=path.name,world=i,sha256=sha(path),
                    ctrl=control[i].tolist(),required_ctrl=required[i].tolist(),ctrl_minus_required=(control[i]-required[i]).tolist(),
                    maximum_command_gap_Nm=float(abs(control[i]-required[i]).max()),
                    maximum_q_upload_error=float(abs(before['q'][i].astype(float)-q[i]).max()),
                    maximum_v_upload_error=float(abs(before['v'][i].astype(float)-v[i]).max()),
                    memory11_before=float(memory_before[i,11]),memory11_after=float(memory_after[i,11]),projection_error=float(diag[i,14])))
        assert queries==30 and len(records)==30 and all(sha(ROOT/n)==value for n,value in sources.items())
        change=max(float(abs(commands[j]-commands[1]).max()) for j in [0,2])
        atomic_json(OUT/'completion.json',dict(verified=True,individual_static_queries=30,records=records,
            integral_endpoint_vs_zero_finalcommand_max_difference_Nm=change,
            source_contract_sha256=sha(OUT/'source_contract.json'),physics_clock_advanced=False,
            integration_steps=0,transitionFD_calls=0,optimization_calls=0,new_training_samples=0,controller_admitted=False,
            limits='Tenmirrorbalancedpoint/staticmemory fixtures only;state11endpoint effectnot continuum/asymmetric/dynamic proof. Nomgap isnot uniqueoldPPO failure ornewmethod contribution. Independentreview required.'))
        print('PASS26630staticqueries;integral command effect',change,'Nm;0physics/FD/PPO',flush=True)
    except BaseException as e:
        atomic_json(OUT/'failure.json',dict(error=repr(e),attempted_individual_queries=queries,records=records,implicit_retry=False));raise
    finally:raw.close()


if __name__=='__main__':run()
