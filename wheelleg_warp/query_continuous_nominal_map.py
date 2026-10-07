"""Exactly50 static paired controller queries; no physics graph launches."""
import json
import time
import mujoco
import numpy as np
import warp as wp
import light_phase_reference as light
import continuous_nominal_map as mapping
import continuous_nominal_control as candidate
from native.controller import D
from native.terrain import HeightTerrainScenario
from train_height_comparison import raw_env
from query_nominal_compatibility import initialize
from test_nom_yaw_filter_probe import control_args
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT=mapping.OUT


def run():
    assert not any((OUT/n).exists() for n in ('query_contract.json','query_completion.json','query_failure.json'))
    started=time.monotonic()
    p=json.loads((OUT/'proposal.json').read_text())
    a=json.loads((OUT/'GPU_source_admission.json').read_text())
    manifest=json.loads((OUT/'table_manifest.json').read_text())
    assert a['verified'] and a['registered50static_queries_admitted'] and p['static_query_budget']==50
    assert a['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/n)==h for n,h in {**p['source_sha256'],**a['source_sha256']}.items())
    assert manifest['builder_sha256']==sha(ROOT/'wheelleg_warp/build_continuous_nominal_map.py')
    assert mapping.COPY.read_text()==mapping.expected_source()
    review_path=ROOT/p['validation_review']['path']
    assert sha(review_path)==p['validation_review']['sha256']
    references=[]
    for r in json.loads(review_path.read_text())['records']:
        assert r['passed']
        references.append(dict(height=r['height'],speed=r['speed'],sha256=r['sha256'],
            path=str((review_path.parent/r['path']).relative_to(ROOT)),kind='nonfitvalidation'))
    for r in p['static_zero_references']:
        references.append(dict(height=r['height_m'],speed=0.,sha256=r['sha256'],path=r['path'],kind='zero'))
    assert len(references)==25
    q,v,required,cases=[],[],[],[]
    for i,r in enumerate(references):
        path=ROOT/r['path'];assert sha(path)==r['sha256']
        with np.load(path,allow_pickle=False) as z:
            prefix='reference_' if r['kind']=='zero' else ''
            q.append(z[prefix+'q']);v.append(z[prefix+'v']);required.append(z[prefix+'ctrl'])
        # Constructor validates task endpoint speed; static command is overwritten below.
        constructor_speed=np.sign(r['speed'])*max(.5,abs(r['speed'])) if r['speed'] else .7
        cases.append(dict(seed=2820000+i,scenario=HeightTerrainScenario(speed=constructor_speed,
            mass=7.,height_l=0.,height_r=0.,center=1.8,offset=0.,mu_l=.8,mu_r=.8,drive_difference=0.,
            delay_ms=0.,solver_iterations=100,stand_height_m=r['height']).__dict__))
    q,v,required=map(np.array,(q,v,required))
    sources={**p['source_sha256'],**a['source_sha256']}
    for n in ('query_continuous_nominal_map.py','query_nominal_compatibility.py','test_nom_yaw_filter_probe.py',
              'light_phase_reference.py','build_continuous_nominal_map.py'):
        sources['wheelleg_warp/'+n]=sha(ROOT/'wheelleg_warp'/n)
    fd_calls=[];fd=mujoco.mjd_transitionFD
    def counted_fd(*args,**kwargs):
        fd_calls.append(float(args[2]));return fd(*args,**kwargs)
    mujoco.mjd_transitionFD=counted_fd
    try:raw=light.instrument(raw_env,cases,'virtual6')
    finally:mujoco.mjd_transitionFD=fd
    queries,records,results=0,[],{}
    graph_launch=wp.capture_launch
    def reject_graph(*args,**kwargs):raise AssertionError('No physicsgraph launch admitted')
    wp.capture_launch=reject_graph
    try:
        assert len(fd_calls)==10 and raw._light_phase_topology['verified']
        device=raw.data.qpos.device;assert device.is_cuda
        frozen=ROOT/p['frozen_bank']['path'];assert sha(frozen)==p['frozen_bank']['sha256']
        with np.load(frozen) as z:
            for n in ('heights','gains','feed','angles'):np.testing.assert_array_equal(raw.k[n].numpy(),z[n])
        table=mapping.arrays(device);table_before=[x.numpy().copy() for x in table]
        private=wp.zeros((25,21),dtype=D,device=device)
        atomic_json(OUT/'query_contract.json',dict(
            verified=True,proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'GPU_source_admission.json'),
            table_manifest_sha256=sha(OUT/'table_manifest.json'),source_sha256=sources,
            references=references,constructor_cases=cases,topology=raw._light_phase_topology,bank_exact_frozen=True,
            baseline_constructor_transitionFD_calls=fd_calls,static_query_budget=50,
            scope='25fixtures x2arms,zeroActor/nomcorrection/gyro,steady FK-vx memory, boot2/state4000/I0. Constructor speed may differ to meet taskendpoint0.5min; actual public commands explicitly replaced by frozen reference speeds. Sameoriginalbank/geometry andinputs; no physicsgraph.'))
        for arm,kernel in (('old',light.phase.roles.experimental.control_physical_nominal),
                           ('map',candidate.control_physical_nominal)):
            raw.reset();raw.diag.zero_()
            raw.data.qpos.assign(q.astype(np.float32));raw.data.qvel.assign(v.astype(np.float32))
            raw.data.sensordata.zero_();raw.data.ctrl.zero_();raw.targets.zero_();raw.nominal_correction.zero_()
            raw.command.assign(np.array([r['speed'] for r in references]))
            state=raw.state.numpy();state[:,0]=4000;state[:,1]=-1;raw.state.assign(state)
            wp.launch(initialize,25,[raw.data.qpos,raw.data.qvel,raw.ids,raw.k['state'],D(0.)])
            light.prepare_step(raw,0);base=raw._light_phase_buffers[0]
            args=control_args(raw);args[12]=base
            if arm=='map':
                wp.launch(mapping.prepare,25,[base,raw.command,raw.active,*table,private])
                args[12]=private
            readonly=[('q',raw.data.qpos),('v',raw.data.qvel),('sensor',raw.data.sensordata),
                ('command',raw.command),('environment_state',raw.state),('clock',raw.data.time),
                ('active',raw.active),('targets',raw.targets),('nominal_correction',raw.nominal_correction),
                ('heights',raw.k['heights']),('gains',raw.k['gains']),('feed',raw.k['feed']),
                ('angles',raw.k['angles']),('reference',args[12]),('base_reference',base)]
            before={n:x.numpy().copy() for n,x in readonly}
            before['memory_before']=raw.k['state'].numpy().copy()
            if arm=='map':
                assert np.all(before['reference'][:20,20]==1.)
                np.testing.assert_array_equal(before['reference'][20:,16:],0.)
            queries+=25
            wp.launch(kernel,25,args,block_dim=32)
            wp.launch(light.phase.roles.finish,25,[0,raw.diag,raw._light_phase_buffers[1]])
            output=dict(**before,memory_after=raw.k['state'].numpy().copy(),ctrl=raw.data.ctrl.numpy().copy(),
                diag=raw.diag.numpy().copy(),role=raw._light_phase_buffers[1].numpy().copy(),
                phase=raw._light_phase_buffers[2].numpy().copy(),required_ctrl=required,original_q=q,original_v=v)
            for n,x in readonly:np.testing.assert_array_equal(x.numpy(),before[n])
            for x,y in zip(table,table_before):np.testing.assert_array_equal(x.numpy(),y)
            for n in ('ctrl','diag','memory_after'):assert np.isfinite(output[n]).all()
            assert not output['clock'].any()
            output['ctrl_minus_required']=output['ctrl'].astype(float)-required
            path=OUT/f'queries_{arm}.npz';np.savez_compressed(path,**output);results[arm]=output
            for i,r in enumerate(references):
                records.append(dict(arm=arm,world=i,**r,query_path=path.name,query_sha256=sha(path),
                    maximum_command_gap_Nm=float(abs(output['ctrl_minus_required'][i]).max()),
                    wheel_gap_Nm=float(abs(output['ctrl_minus_required'][i,-2:]).max()),
                    raw_nominal_gap_Nm=float(abs(output['diag'][i,15:21]-required[i]).max()),
                    projection_error=float(output['diag'][i,14]),
                    q_upload_error=float(abs(output['q'][i].astype(float)-q[i]).max()),
                    v_upload_error=float(abs(output['v'][i].astype(float)-v[i]).max())))
            print('completed',queries,'/50',arm,flush=True)
        old,new=results['old'],results['map']
        for n in ('q','v','sensor','command','environment_state','clock','active','targets',
                  'nominal_correction','heights','gains','feed','angles','memory_before','base_reference'):
            np.testing.assert_array_equal(old[n],new[n])
        np.testing.assert_array_equal(old['reference'],new['reference'][:,:16])
        zero_exact=all(np.array_equal(old[n][20:],new[n][20:]) for n in ('ctrl','diag','memory_after'))
        zero_exact=zero_exact and all(np.array_equal(old[n][:,20:],new[n][:,20:]) for n in ('role','phase'))
        assert queries==50 and len(records)==50 and all(sha(ROOT/n)==h for n,h in sources.items())
        atomic_json(OUT/'query_completion.json',dict(
            verified=True,individual_static_queries=50,records=records,zero_speed_runtime_exact=zero_exact,
            source_contract_sha256=sha(OUT/'query_contract.json'),baseline_constructor_transitionFD_calls=10,
            new_experimental_transitionFD_calls=0,integration_steps=0,optimization_calls=0,new_training_samples=0,
            end_to_end_query_seconds=time.monotonic()-started,controller_admitted=False,
            limits='Static nonfitnominalreferences andfivezero fixtures only; reportgapswithoutfittedthreshold. Independentreview283 required before finite next-step admission. No task/dynamics/PPO or production.'))
        print('DONE28250static;zero exact',zero_exact,flush=True)
    except BaseException as error:
        atomic_json(OUT/'query_failure.json',dict(error=repr(error),attempted_individual_queries=queries,records=records,implicit_retry=False))
        raise
    finally:
        wp.capture_launch=graph_launch;raw.close()


if __name__=='__main__':
    run()
