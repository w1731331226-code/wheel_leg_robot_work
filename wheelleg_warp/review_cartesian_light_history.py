"""Independent stored command/history receipt; overwritten actual terminals stay unproven."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v2'
SCALES=np.array([40.]*4+[4.5]*2)


def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def lateral(packet):
    return np.sin(packet[:,2])*packet[:,6]+np.cos(packet[:,2])*packet[:,7]


def fresh(public):
    assert public.ndim==2 and public.shape[1]==38 and np.isfinite(public).all()
    n=len(public);frames=np.repeat(np.column_stack((public,np.zeros(n,np.float32)))[:,None,:],10,axis=1)
    valid=np.zeros((n,10),np.float32);valid[:,-1]=1
    return np.concatenate((frames.reshape(n,390),np.zeros((n,81),np.float32),valid),axis=1)


class Receipt:
    def __init__(self,public):
        n=len(public);self.count=np.zeros(n,np.int64);self.previous=self.count.copy()
        self.total=np.zeros((n,6));self.prefix=np.zeros((n,82,6));self.stamp=np.full((n,82),-1,np.int64);self.stamp[:,0]=0
        self.clock=np.zeros(n);self.y=self.clock.copy();self.velocity=lateral(public);self.actor=fresh(public)
        self.steps=0;self.done_rows=0;self.max_integral_error=0.

    def consume(self,r):
        assert all(np.isfinite(r[k]).all() for k in ('actor','returned','state','delay_steps','command_integral','sensor_elapsed_s','terminal_packet','public_after'))
        np.testing.assert_array_equal(r['actor'],self.actor)
        np.testing.assert_array_equal(r['previous_sensor_steps'],self.previous)
        g=r['guard'];n=len(self.count);assert g.shape==(40,n,62) and np.isfinite(g).all()
        done=r['done'];assert done.shape==(n,) and done.dtype==np.bool_
        for slot in range(40):
            w=np.flatnonzero(g[slot,:,0]==1)
            if not len(w):continue
            np.testing.assert_array_equal(g[slot,w,1],self.count[w]+1)
            self.count[w]+=1;self.steps+=len(w)
            self.total[w]+=g[slot,w,20:26]*.0005
            self.prefix[w,self.count[w]%82]=self.total[w]
            self.stamp[w,self.count[w]%82]=self.count[w]
        np.testing.assert_array_equal(self.count,r['state'][:,0])
        np.testing.assert_array_equal(self.stamp,r['command_stamp'])
        np.testing.assert_allclose(self.total,r['command_total'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(self.prefix,r['command_prefix'],rtol=0,atol=1e-10)
        delay=r['delay_steps'];assert np.all(delay==delay.astype(np.int64)) and np.all((delay>=0)&(delay<=40))
        end=np.maximum(self.count-delay.astype(np.int64),0);assert np.all((end>=self.previous)&(end-self.previous<=40))
        w=np.arange(n)
        np.testing.assert_array_equal(self.stamp[w,end%82],end)
        np.testing.assert_array_equal(self.stamp[w,self.previous%82],self.previous)
        impulse=self.prefix[w,end%82]-self.prefix[w,self.previous%82]
        dt=(end-self.previous)*.0005
        self.max_integral_error=max(self.max_integral_error,float(abs(impulse-r['command_integral']).max()))
        np.testing.assert_allclose(r['command_integral'],impulse,rtol=0,atol=1e-10)
        np.testing.assert_array_equal(r['sensor_elapsed_s'],dt)
        terminal_packet=r['terminal_packet'];public=r['public_after']
        np.testing.assert_array_equal(public[~done],terminal_packet[~done])
        np.testing.assert_array_equal(terminal_packet[dt==0,20:22],self.actor[dt==0,371:373])
        route_dt=np.full(n,.02);route_dt[done]=self.count[done]*.0005-self.clock[done]
        assert np.all(route_dt>0)
        velocity=lateral(terminal_packet);self.y+=.5*(self.velocity+velocity)*route_dt
        self.velocity=velocity;self.clock+=route_dt
        frames=self.actor[:,:390].reshape(n,10,39).copy();commands=self.actor[:,390:462].reshape(n,9,8).copy()
        elapsed=self.actor[:,462:471].copy();valid=self.actor[:,471:].copy()
        frames[:,:-1]=frames[:,1:].copy();frames[:,-1]=np.column_stack((terminal_packet,self.y.astype(np.float32)))
        commands[:,:-1]=commands[:,1:].copy();commands[:,-1]=0
        # Quantize the independently checked recorded interval in the frozen float32 history format.
        mean=np.divide(r['command_integral'],dt[:,None],out=np.zeros((n,6)),where=dt[:,None]>0)
        commands[:,-1,:6]=mean/SCALES
        elapsed[:,:-1]=elapsed[:,1:].copy();elapsed[:,-1]=dt/.02
        valid[:,:-1]=valid[:,1:].copy();valid[:,-1]=1
        expected_terminal=np.concatenate((frames.reshape(n,390),commands.reshape(n,72),elapsed,valid),axis=1)
        if 'terminal_history' in r:np.testing.assert_array_equal(r['terminal_history'],expected_terminal)
        frames[done]=np.column_stack((public,np.zeros(n,np.float32)))[done,None,:]
        commands[done]=0;elapsed[done]=0;valid[done]=0;valid[done,-1]=1
        self.actor=np.concatenate((frames.reshape(n,390),commands.reshape(n,72),elapsed,valid),axis=1)
        np.testing.assert_array_equal(r['returned'],self.actor)
        self.previous=end;self.previous[done]=0
        self.count[done]=0;self.total[done]=0;self.prefix[done]=0;self.stamp[done]=-1;self.stamp[done,0]=0
        self.clock[done]=0;self.y[done]=0;self.velocity[done]=lateral(public[done]);self.done_rows+=int(done.sum())

    def final(self,z):
        np.testing.assert_allclose(z['command_0'],self.total,rtol=0,atol=1e-10)
        np.testing.assert_allclose(z['command_1'],self.prefix,rtol=0,atol=1e-10)
        np.testing.assert_array_equal(z['command_2'],self.stamp)
        np.testing.assert_array_equal(z['previous_sensor_steps'],self.previous)
        encoded=np.concatenate((z['frames'].reshape(len(self.count),390),z['inputs'].reshape(len(self.count),72),z['elapsed'],z['valid']),axis=1)
        np.testing.assert_array_equal(encoded,self.actor)


def self_check():
    public=np.zeros((2,38),np.float32);receipt=Receipt(public);g=np.zeros((40,2,62));g[0,:,0]=1;g[0,:,1]=1
    state=np.zeros((2,39));state[:,0]=1;stamp=receipt.stamp.copy();stamp[:,1]=1
    endpoint=public.copy();endpoint[:,7]=1
    frames=fresh(public)[:,:390].reshape(2,10,39).copy();frames[:,-1,:38]=endpoint
    frames[:,-1,38]=.01
    elapsed=np.zeros((2,9),np.float32);elapsed[:,-1]=.025;valid=np.zeros((2,10),np.float32);valid[:,-2:]=1
    returned=np.concatenate((frames.reshape(2,390),np.zeros((2,72),np.float32),elapsed,valid),axis=1)
    r=dict(actor=fresh(public),guard=g,done=np.zeros(2,bool),previous_sensor_steps=np.zeros(2,np.int64),state=state,
        command_total=np.zeros((2,6)),command_prefix=np.zeros((2,82,6)),command_stamp=stamp,delay_steps=np.zeros(2),
        command_integral=np.zeros((2,6)),sensor_elapsed_s=np.full(2,.0005),terminal_packet=endpoint,public_after=endpoint,returned=returned)
    receipt.consume(r)
    for field in ('actor','command_integral','returned','command_stamp'):
        bad={k:v.copy() for k,v in r.items()};bad[field].flat[0]+=1
        try:Receipt(public).consume(bad)
        except AssertionError:pass
        else:raise AssertionError('Corrupt '+field+' accepted')
    bad={k:v.copy() for k,v in r.items()};bad['returned'].flat[0]=np.nan
    try:Receipt(public).consume(bad)
    except AssertionError:pass
    else:raise AssertionError('Nonfinite history accepted')
    episode_worlds([{'world':0},{'world':1}],np.ones(2,bool))
    try:episode_worlds([{'world':0},{'world':0}],np.ones(2,bool))
    except AssertionError:pass
    else:raise AssertionError('Duplicate episode and missing done accepted')
    print('PASS375 independent command/prefix/returned-history corruption rejected;0physics',flush=True)


def post_private(r,post):
    mapping=dict(guard_memory='guard_memory',parking_seen='parking_0',parking_effective='parking_1',phase_reference='phase_0',
        phase_role='phase_1',phase_monitor='phase_2',command_total='command_0',command_prefix='command_1',command_stamp='command_2')
    mask=~r['done']
    for name,key in mapping.items():
        a,b=r[name],post[key]
        if name in ('phase_role','phase_monitor'):np.testing.assert_array_equal(a[:,mask],b[:,mask])
        else:np.testing.assert_array_equal(a[mask],b[mask])


def episode_worlds(rows,done):
    assert sorted(row['world'] for row in rows)==np.flatnonzero(done).tolist()


def review(directory):
    initial=load(directory/'initial.npz');prefix=load(directory/'zero_prefix.npz');zero=load(directory/'zero_before_autoreset.npz')
    r={**zero,**{k:prefix[k] for k in ('actor','returned','command_integral','sensor_elapsed_s')}}
    r['public_after']=prefix['public'];first=Receipt(initial['public']);first.consume(r);first.final(prefix);post_private(r,prefix)
    cache=load(directory/'cache_stage1.npz');public=np.column_stack((cache['obs0'],np.zeros((10,6),np.float32)))
    receipt=Receipt(public);episodes=json.loads((directory/'episodes.json').read_text())['episodes'];by_packet={}
    for row in episodes:by_packet.setdefault(row['packet'],[]).append(row)
    keys=('actor','returned','guard','state','done','previous_sensor_steps','delay_steps','command_total','command_prefix','command_stamp',
        'command_integral','sensor_elapsed_s','terminal_packet','public_after','terminal_history','stage_before','stage_after',
        'guard_memory','parking_seen','parking_effective','phase_reference','phase_role','phase_monitor')
    stage_caches={s:load(directory/f'cache_stage{s}.npz') for s in (1,2,3)};packet=0
    for path in sorted(directory.glob('block_*.npz')):
        with np.load(path) as z:data={k:z[k] for k in keys}
        for i in range(len(data['actor'])):
            r={k:v[i] for k,v in data.items()};receipt.consume(r)
            episode_worlds(by_packet.get(packet,[]),r['done'])
            for row in by_packet.get(packet,[]):
                w=row['world'];assert r['done'][w] and row['curriculum_stage']==r['stage_before'][w]
                assert row['physical_steps']==r['state'][w,0] and row['duration_s']==r['state'][w,0]*.0005
            for s in (1,2,3):
                mask=r['done']&(r['stage_after']==s);expected=np.column_stack((stage_caches[s]['obs0'][mask],np.zeros((mask.sum(),6),np.float32)))
                np.testing.assert_array_equal(r['public_after'][mask],expected)
                np.testing.assert_array_equal(r['delay_steps'][r['stage_before']==s],stage_caches[s]['param'][r['stage_before']==s,4])
            packet+=1
    assert packet==1200 and receipt.done_rows==len(episodes)==26
    final=load(directory/'final_common.npz');receipt.final(final);post_private(r,final)
    return dict(packets=packet,zero_valid_steps=first.steps,rollout_valid_steps=receipt.steps,done_rows=receipt.done_rows,
        max_command_integral_error=receipt.max_integral_error,all_stored_returned481_command_rings_windows_and_public_resets_verified=True,
        stored_terminal_history_is_reconstructed_expectation_only=True,actual_done_info_terminal481_independently_verified=False,
        zero_and_final_non_done_post_private_directly_verified=True,
        intermediate_guard_parking_phase_collector_post_private_independently_verified=False,
        actual_per_transition_param_q0_obs0_native_reference_independently_verified=False)


if __name__=='__main__':
    self_check()
    for arm in ('CartLive3','CartReset3'):print(arm,review(OUT/'runtime'/arm),flush=True)
