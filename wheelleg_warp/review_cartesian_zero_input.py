"""Independent saved-stream receipt and closure of the cross-run exact-zero route."""
import json
import xml.etree.ElementTree as ET
import numpy as np
from review_yaw_sector import ROOT,sha
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'
OUT=BASE/'cartesian_zero_input_diagnostic_v1'


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def same(a,b):
    assert set(a)==set(b)
    for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=k)


def stream(initial,after,trace,terminal,gyro):
    p,b,c=trace['pre'],trace['boundaries'],trace['control'];g=after['guard'];n=10
    assert p.shape==(40,n,66) and b.shape==(40,n,6,10) and c.shape==(40,n,2,44)
    assert all(np.isfinite(v).all() for v in trace.values())
    np.testing.assert_array_equal(p[:,:,:2],1);np.testing.assert_array_equal(b[:,:,:,:2],1)
    np.testing.assert_array_equal(p[:,:,2],np.tile(np.arange(1,41)[:,None],(1,n)))
    np.testing.assert_array_equal(p[:,:,4],p[:,:,38+gyro])
    for stage in (0,1,2):
        np.testing.assert_array_equal(b[:,:,stage,2],p[:,:,3]);np.testing.assert_array_equal(b[:,:,stage,3],p[:,:,4])
    for stage in (4,5):np.testing.assert_array_equal(b[:,:,stage],b[:,:,3])
    np.testing.assert_array_equal(b[:,:,:,4:],0)
    np.testing.assert_array_equal(p[1:,:,3],b[:-1,:,3,2])
    np.testing.assert_array_equal(p[0,:,3],initial['controller_memory'][:,5])
    np.testing.assert_array_equal(b[-1,:,3,2],terminal['controller_memory'][:,5])
    np.testing.assert_array_equal(p[0,:,5:22],initial['q']);np.testing.assert_array_equal(p[0,:,22:38],initial['v'])
    np.testing.assert_array_equal(p[-1,:,5:22],terminal['stopped_q']);np.testing.assert_array_equal(p[-1,:,22:38],terminal['stopped_v'])
    for name in ('state','public','controller_memory','q','v','sensors'):np.testing.assert_array_equal(terminal[name],after[name])
    np.testing.assert_array_equal(terminal['done'],0);np.testing.assert_array_equal(after['done'],False)
    np.testing.assert_array_equal(g[:,:,0],1);np.testing.assert_array_equal(g[:,:,1],p[:,:,2])
    np.testing.assert_array_equal(g[:,:,2:8],c[:,:,1,6:12]);np.testing.assert_array_equal(g[:,:,8:14],c[:,:,1,12:18])
    np.testing.assert_array_equal(g[:,:,26],c[:,:,1,18]);np.testing.assert_array_equal(c[:,:,1,20],0)
    np.testing.assert_array_equal(c[:,:,1,:6],g[:,:,20:26]);np.testing.assert_array_equal(c[1:,:,0,:6],g[:-1,:,20:26])
    np.testing.assert_array_equal(after['ctrl'],g[-1,:,20:26])
    memory=after['guard_memory'];np.testing.assert_array_equal(memory[:,0],1);np.testing.assert_array_equal(memory[:,15],40)
    for left,right in ((memory[:,1:5],g[-1,:,54:58]),(memory[:,5:11],g[-1,:,20:26]),(memory[:,11:15],g[-1,:,34:38])):
        np.testing.assert_array_equal(left,right)
    role=after['phase_1'];park=after['parking_2'];monitor=after['phase_2']
    np.testing.assert_array_equal(role[:,:,:2],g[:,:,:2]);np.testing.assert_array_equal(park[:,:,:2],g[:,:,:2])
    np.testing.assert_array_equal(park[:,:,5:29],0);np.testing.assert_array_equal(park[:,:,29],0)
    np.testing.assert_array_equal(role[:,:,7:9],c[:,:,1,27:29])
    np.testing.assert_array_equal(monitor[:,:,:2],g[:,:,:2]);np.testing.assert_array_equal(monitor[:,:,2],park[:,:,2])
    np.testing.assert_array_equal(monitor[:,:,3],monitor[:,:,2]==0);np.testing.assert_array_equal(monitor[:,:,5],role[:,:,5])
    s=after['state'];assert np.isfinite(s).all()
    np.testing.assert_array_equal(s[:,0],40);np.testing.assert_array_equal(s[:,37],40)
    assert s[:,31:33].min()>=HEIGHT_115_GEOMETRIC_MIN and s[:,33].min()>=0 and s[:,35:37].max()<=1e-6 and s[:,38].min()>=0
    cumulative=np.cumsum(g[:,:,20:26]*.0005,axis=0).transpose(1,0,2)
    # Existing368 interval reconstruction tolerance; no cross-trajectory tolerance.
    np.testing.assert_allclose(after['command_1'][:,1:41],cumulative,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(after['command_0'],after['command_1'][:,40])
    np.testing.assert_array_equal(after['command_1'][:,0],0)
    stamp=np.full((n,82),-1);stamp[:,:41]=np.arange(41)
    np.testing.assert_array_equal(after['command_2'],stamp);np.testing.assert_array_equal(after['previous_sensor_steps'],40)
    np.testing.assert_array_equal(after['param'][:,4],0)
    frames=np.repeat(initial['frames'][:,-1:, :],10,axis=1);frames[:,-1,:38]=after['public']
    old=initial['frames'][:,-1,:38];new=after['public']
    velocity=lambda x:np.sin(x[:,2])*x[:,6]+np.cos(x[:,2])*x[:,7]
    frames[:,-1,38]=(.5*(velocity(old)+velocity(new))).astype(float)*.02
    inputs=np.zeros((n,9,8),np.float32);inputs[:,-1,:6]=after['command_0']/.02/np.array([40.]*4+[4.5]*2)
    elapsed=np.zeros((n,9),np.float32);elapsed[:,-1]=1
    valid=np.zeros((n,10),np.float32);valid[:,-2:]=1
    for k,v in dict(frames=frames,inputs=inputs,elapsed=elapsed,valid=valid).items():np.testing.assert_array_equal(after[k],v)
    expected=np.concatenate((frames.reshape(n,390),inputs.reshape(n,72),elapsed,valid),axis=1)
    np.testing.assert_array_equal(after['returned'],expected)
    return dict(substeps=400,min_leg_m=float(s[:,31:33].min()),min_eight_margin_rad=float(s[:,33].min()),
        max_torque_excess_Nm=float(s[:,35:37].max()),min_design_margin_rad=float(s[:,38].min()),
        prefix_max_abs_error_Nms=float(abs(after['command_1'][:,1:41]-cumulative).max()))


def run():
    assert not (OUT/'independent_review.json').exists()
    receipt=json.loads((OUT/'execution_receipt.json').read_text());fail=json.loads((OUT/'runtime/failure.json').read_text())
    admit=json.loads((OUT/'source_admission.json').read_text());start=json.loads((OUT/'runtime/started.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in receipt['input_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in admit['source_sha256'].items()) and len(admit['source_sha256'])==105
    assert start['admission_sha256']==sha(OUT/'source_admission.json') and start['commit']==receipt['start_commit']
    assert fail['status']=='baseline_repeat_failure' and fail['graph_world_steps']==800 and len(fail['FD_calls'])==35
    assert fail['secondary_errors']==[] and not fail['implicit_retry'] and fail['new_learning_samples']==fail['new_model_forward_rows']==0
    assert [r['condition'] for r in fail['reports']]==['B0_repeat1','B0_repeat2']
    assert not (OUT/'runtime/CartLive3').exists() and not (OUT/'runtime/completion.json').exists()
    paths=[OUT/'runtime'/n for n in ('B0_repeat1','B0_repeat2')];models=[]
    for stage in (1,2,3):
        a,b=[load(p/f'constructor_stage{stage}.npz') for p in paths];same(a,b);assert len(a)==413
        assert json.loads((paths[0]/f'constructor_stage{stage}.json').read_text())==json.loads((paths[1]/f'constructor_stage{stage}.json').read_text())
        models.append(a)
    sensor_names=[e.attrib['name'] for e in ET.parse(ROOT/'wheelleg_ppo/xml/wheelleg.xml').getroot().find('sensor')]
    gyro=int(models[0]['model.sensor_adr'][sensor_names.index('body_gyro')]);assert gyro==4
    initials=[load(p/'initial.npz') for p in paths];same(*initials);assert len(initials[0])==28
    data=[];reports=[]
    for p,initial in zip(paths,initials):
        after=load(p/'zero_prefix.npz');trace=load(p/'input_trace.npz');terminal=load(p/'terminal_before_wait.npz')
        same(terminal,load(p/'preserved_terminal.npz'));same(trace,load(p/'preserved_observer.npz'))
        same({k:v for k,v in after.items() if k not in ('returned','reward','done')},load(p/'preserved_common.npz'))
        topology=json.loads((p/'topology.json').read_text())
        assert topology['counts']==dict(phase=40,Nom=40,clear=0,deliver=0) and topology['observer_calls_each_slot']==9 and not topology['cartesian']
        report=stream(initial,after,trace,terminal,gyro);reports.append(report);data.append((after,trace))
    pre0,pre1=data[0][1]['pre'],data[1][1]['pre'];first=np.argwhere(pre0!=pre1)[0].tolist();assert first==[4,6,25]
    vslots=np.flatnonzero(pre0[:,6,25]!=pre1[:,6,25]).tolist();gslots=np.flatnonzero(pre0[:,6,4]!=pre1[:,6,4]).tolist()
    assert vslots==list(range(4,10)) and gslots==list(range(5,11))
    b0,b1=data[0][1]['boundaries'],data[1][1]['boundaries']
    np.testing.assert_array_equal(b0[5,:,2,2],b1[5,:,2,2]);assert b0[5,6,2,3]!=b1[5,6,2,3] and b0[5,6,3,2]!=b1[5,6,3,2]
    np.testing.assert_array_equal(data[0][1]['control'][:,:,:,:6],data[1][1]['control'][:,:,:,:6])
    np.testing.assert_array_equal(data[0][0]['guard'],data[1][0]['guard'])
    # Actual saved-stream corruption tests exercise acceptance boundaries without replay.
    for key,index in (('pre',(0,0,4)),('boundaries',(2,0,2,2)),('control',(3,0,1,0))):
        bad={k:v.copy() for k,v in data[0][1].items()};bad[key][index]+=1
        try:stream(initials[0],data[0][0],bad,load(paths[0]/'terminal_before_wait.npz'),gyro)
        except AssertionError:pass
        else:raise AssertionError('Corrupt stream accepted: '+key)
    atomic_json(OUT/'independent_review.json',dict(round=372,verified=True,status='baseline_repeat_failure_independently_received',
        receipt_sha256=sha(OUT/'execution_receipt.json'),failure_sha256=sha(OUT/'runtime/failure.json'),
        admission_sha256=sha(OUT/'source_admission.json'),source_sha256=sha(__file__),input_sha256=receipt['input_sha256'],
        model_arrays_each_stage=413,common_initial_fields=28,reports=reports,total_substeps=800,FD_calls=35,
        gyro_address=gyro,first_recorded_difference=first,v_difference_slots=vslots,gyro_difference_slots=gslots,
        final_control_and_guard_exact=True,corrupt_stream_rejection_verified=True,
        limits='只接收已存基线短时重复性失败与输入链归属；原物理监测通过不等完整实机/长期安全，未保存逐步actuator_force/contact，未排他定位solver原因；Cart未执行。',
        new_physics_steps=0,new_FD=0,new_model_forward_rows=0,new_learning_samples=0,formal5_admitted=False,goal_complete=False))
    print('RECEIVED372 3x413model/28initial fields;800substeps/35FD;input-before-filter,guard/command/history/terminal preservation verified;0newphysics',flush=True)


if __name__=='__main__':run()
