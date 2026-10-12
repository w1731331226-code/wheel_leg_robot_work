"""Stored PPO command/history/RMS/buffer receipt; no simulation or network evaluation."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import review_cartesian_light_history as history

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'
OUT=BASE/'cartesian_ppo_interface_v1'
RMS_KEYS=('obs_rms_mean','obs_rms_var','obs_rms_count','ret_rms_mean','ret_rms_var','ret_rms_count','returns')
PRIVATE=('guard_memory','parking_seen','parking_effective','phase_reference','phase_role','phase_monitor','command_total','command_prefix','command_stamp')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def inputs():
    receipt=json.loads((OUT/'execution_receipt.json').read_text());a=json.loads((OUT/'source_admission.json').read_text())
    p=json.loads((OUT/'proposal.json').read_text());started=json.loads((OUT/'runtime/started.json').read_text())
    assert receipt['proposal_sha256']==a['proposal_sha256']==sha(OUT/'proposal.json')
    assert receipt['source_admission_sha256']==started['admission_sha256']==sha(OUT/'source_admission.json')
    assert receipt['completion_sha256']==sha(OUT/'runtime/completion.json') and receipt['started']==started
    assert p['ppo']['gamma']==.99 and p['ppo']['gae_lambda']==.95 and p['ppo']['n_steps']==50 and not p['ppo']['use_sde']
    assert p['normalization']['norm_obs'] and not p['normalization']['norm_reward'] and p['normalization']['clip_obs']==10 and p['normalization']['epsilon']==1e-8
    assert p['normalization'].get('gamma',.99)==.99 and p['bank_seed']==33531 and p['worlds']==10
    assert not (OUT/'runtime/failure.json').exists() and np.__version__==a['runtime']['versions']['numpy']
    for mapping in (receipt['outputs_sha256'],a['source_sha256'],p['parent_sha256'],p['old_failures_retained']):assert all(sha(ROOT/n)==h for n,h in mapping.items())
    for mapping in (a['runtime']['installed_sha256'],a['runtime']['PPO_source_sha256']):assert all(sha(Path(n))==h for n,h in mapping.items())
    expected={str((OUT/'source_admission.json').relative_to(ROOT)):sha(OUT/'source_admission.json'),**a['source_sha256']}
    for commit in ('HEAD',started['commit']):
        names=list(expected);data=subprocess.check_output(['git','cat-file','--batch'],input=''.join(commit+':'+n+'\n' for n in names).encode(),cwd=ROOT);position=0
        for n in names:
            end=data.index(b'\n',position);header=data[position:end].split();assert len(header)==3 and header[1]==b'blob'
            size=int(header[2]);assert hashlib.sha256(data[end+1:end+1+size]).hexdigest()==expected[n];position=end+size+2
        assert position==len(data)
    return receipt,a


def update(mean,var,count,x):
    batch_mean=np.mean(x,axis=0);batch_var=np.var(x,axis=0);n=len(x);delta=batch_mean-mean
    total=count+n
    return mean+delta*n/total,(var*count+batch_var*n+np.square(delta)*count*n/(count+n))/(count+n),n+count


def normalized(packet,state):
    return np.clip((packet-state['obs_rms_mean'])/np.sqrt(state['obs_rms_var']+1e-8),-10,10).astype(np.float32)


def check_rms(r,state):
    before={n:r['rms_before_'+n] for n in RMS_KEYS};after={n:r['rms_after_'+n] for n in RMS_KEYS}
    for n in RMS_KEYS:np.testing.assert_array_equal(before[n],state[n])
    mean,var,count=update(before['obs_rms_mean'],before['obs_rms_var'],before['obs_rms_count'],r['returned'])
    for name,value in zip(('mean','var','count'),(mean,var,count)):np.testing.assert_array_equal(after['obs_rms_'+name],value)
    returns=before['returns']*.99+r['reward']
    mean,var,count=update(before['ret_rms_mean'],before['ret_rms_var'],before['ret_rms_count'],returns)
    for name,value in zip(('mean','var','count'),(mean,var,count)):np.testing.assert_array_equal(after['ret_rms_'+name],value)
    returns[r['done']]=0;np.testing.assert_array_equal(after['returns'],returns)
    np.testing.assert_array_equal(r['normalized_actor'],normalized(r['actor'],before))
    np.testing.assert_array_equal(r['normalized_returned'],normalized(r['returned'],after))
    np.testing.assert_array_equal(r['normalized_terminal_history'][r['done']],normalized(r['terminal_info_history'][r['done']],after))
    np.testing.assert_array_equal(r['env_reward_before_bootstrap'],r['reward'].astype(np.float32))
    return {n:v.copy() for n,v in after.items()}


def check_private(r,receipt):
    mask=~r['done']
    for n in PRIVATE:
        a,b=r[n],r['post_'+n]
        if n in ('phase_role','phase_monitor'):np.testing.assert_array_equal(a[:,mask],b[:,mask])
        else:np.testing.assert_array_equal(a[mask],b[mask])
    np.testing.assert_array_equal(r['post_previous_sensor_steps'],receipt.previous)
    np.testing.assert_array_equal(r['post_command_stamp'],receipt.stamp)
    np.testing.assert_allclose(r['post_command_total'],receipt.total,rtol=0,atol=1e-10)
    np.testing.assert_allclose(r['post_command_prefix'],receipt.prefix,rtol=0,atol=1e-10)
    done=r['done']
    for n in ('guard_memory','parking_seen'):np.testing.assert_array_equal(r['post_'+n][done],0)
    np.testing.assert_array_equal(r['post_parking_effective'][done],r['parking_effective'][done])


def check_gaussian(r):
    a,mean,std=r['gaussian_action3'],r['gaussian_mean3'],r['gaussian_std3']
    assert a.shape==mean.shape==std.shape==(10,3) and np.isfinite(a).all()
    np.testing.assert_array_equal(mean,0);np.testing.assert_array_equal(std,.25)
    np.testing.assert_array_equal(r['clipped_action3'],np.clip(a,-1,1))
    paired=np.stack((r['clipped_action3'],-r['clipped_action3']),axis=-1).reshape(10,6)
    np.testing.assert_array_equal(r['action'],paired)
    terms=-((a-mean)**2)/(2*(std**2))-np.log(std)-np.float32(.5*np.log(2*np.pi))
    expected=terms.sum(axis=1)
    # Fixed float32 operation budget; independent CPU/CUDA log arithmetic need not be bitwise equal.
    tol=32*np.finfo(np.float32).eps
    np.testing.assert_allclose(r['log_probs'],expected,rtol=tol,atol=tol)
    return float(abs(r['log_probs']-expected).max()),int((a!=r['clipped_action3']).any(axis=1).sum())


def check_buffer(z,rows,starts):
    fields=dict(observations='normalized_actor',actions='gaussian_action3',values='values',log_probs='log_probs',rewards='env_reward_before_bootstrap')
    assert z['observations'].shape==(50,10,481) and z['actions'].shape==(50,10,3)
    assert all(np.isfinite(value).all() for value in z.values())
    assert not any(r['timeout'].any() for r in rows),'Only observed no-timeout buffers independently admitted'
    for n,key in fields.items():
        value=np.stack([r[key] for r in rows]);value=value.reshape(z[n].shape)
        np.testing.assert_array_equal(z[n],value)
    episode=np.stack([starts]+[r['done'] for r in rows[:-1]])
    np.testing.assert_array_equal(z['episode_starts'],episode)
    np.testing.assert_array_equal(z['last_dones'],rows[-1]['done'])
    advantage=np.zeros_like(z['advantages']);last=np.zeros(10,np.float32)
    for t in range(49,-1,-1):
        alive=1.0-z['last_dones'].astype(np.float32) if t==49 else 1.0-z['episode_starts'][t+1]
        value=z['last_values'].flatten() if t==49 else z['values'][t+1]
        delta=z['rewards'][t]+.99*value*alive-z['values'][t]
        last=delta+.99*.95*alive*last;advantage[t]=last
    np.testing.assert_array_equal(z['advantages'],advantage)
    np.testing.assert_array_equal(z['returns'],advantage+z['values'])
    return rows[-1]['done'].copy()


def self_check():
    history.self_check()
    receipt=history.Receipt(np.zeros((2,38),np.float32));done=np.array([True,False])
    private=dict(guard_memory=np.ones((2,16)),parking_seen=np.ones(2),parking_effective=np.ones((2,6)),phase_reference=np.zeros((2,16)),
        phase_role=np.zeros((40,2,11)),phase_monitor=np.zeros((40,2,6)),command_total=receipt.total.copy(),command_prefix=receipt.prefix.copy(),command_stamp=receipt.stamp.copy())
    state=dict(done=done,**private,**{'post_'+n:v.copy() for n,v in private.items()},post_previous_sensor_steps=receipt.previous.copy())
    state['post_guard_memory'][done]=0;state['post_parking_seen'][done]=0;check_private(state,receipt)
    for key in ('post_parking_seen','post_parking_effective'):
        bad={n:v.copy() for n,v in state.items()};bad[key][0]=1 if key=='post_parking_seen' else 0
        try:check_private(bad,receipt)
        except AssertionError:pass
        else:raise AssertionError('Incorrect parking reset semantics accepted')
    r=dict(gaussian_action3=np.zeros((10,3),np.float32),gaussian_mean3=np.zeros((10,3),np.float32),gaussian_std3=np.full((10,3),.25,np.float32),
        clipped_action3=np.zeros((10,3),np.float32),action=np.zeros((10,6),np.float32),log_probs=np.full(10,-3*(np.log(np.float32(.25))+np.float32(.5*np.log(2*np.pi))),np.float32))
    check_gaussian(r)
    for key in ('gaussian_std3','action','log_probs'):
        bad={n:v.copy() for n,v in r.items()};bad[key].flat[0]+=.1
        try:check_gaussian(bad)
        except AssertionError:pass
        else:raise AssertionError('Bad '+key+' accepted')
    rows=[dict(timeout=np.zeros(10,bool),done=np.zeros(10,bool),normalized_actor=np.zeros((10,481),np.float32),gaussian_action3=np.zeros((10,3),np.float32),values=np.zeros((10,1),np.float32),log_probs=np.zeros(10,np.float32),env_reward_before_bootstrap=np.zeros(10,np.float32)) for _ in range(50)]
    z=dict(observations=np.zeros((50,10,481),np.float32),actions=np.zeros((50,10,3),np.float32),values=np.zeros((50,10),np.float32),log_probs=np.zeros((50,10),np.float32),rewards=np.zeros((50,10),np.float32),advantages=np.zeros((50,10),np.float32),returns=np.zeros((50,10),np.float32),episode_starts=np.zeros((50,10),np.float32),last_dones=np.zeros(10,bool),last_values=np.zeros((10,1),np.float32))
    check_buffer(z,rows,np.zeros(10,bool));z['returns'][0,0]=1
    try:check_buffer(z,rows,np.zeros(10,bool))
    except AssertionError:pass
    else:raise AssertionError('Bad GAE accepted')


def review_arm(directory):
    initial=history.load(directory/'initial.npz');prefix=history.load(directory/'zero_prefix.npz');zero=history.load(directory/'zero_before_autoreset.npz')
    zr={**zero,**{n:prefix[n] for n in ('actor','returned','command_integral','sensor_elapsed_s')}};zr['public_after']=prefix['public']
    cold=history.Receipt(initial['public']);cold.consume(zr);cold.final(prefix);history.post_private(zr,prefix)
    start=history.load(directory/'initial_policy_observation.npz');public=start['raw'][:,351:389]
    receipt=history.Receipt(public);np.testing.assert_array_equal(start['raw'],receipt.actor)
    mean,var,count=update(np.zeros(481),np.ones(481),np.asarray(.0001),start['raw'])
    for n,v in zip(('mean','var','count'),(mean,var,count)):np.testing.assert_array_equal(start['obs_rms_'+n],v)
    np.testing.assert_array_equal(start['normalized'],normalized(start['raw'],start))
    state={n:start[n].copy() for n in RMS_KEYS};caches={s:history.load(directory/f'cache_stage{s}.npz') for s in (1,2,3)}
    np.testing.assert_array_equal(state['ret_rms_mean'],0);np.testing.assert_array_equal(state['ret_rms_var'],1)
    np.testing.assert_array_equal(state['ret_rms_count'],.0001);np.testing.assert_array_equal(state['returns'],0)
    episodes=json.loads((directory/'episodes.json').read_text())['episodes'];by_packet={}
    for row in episodes:by_packet.setdefault(row['packet'],[]).append(row)
    chunks=[];starts=np.ones(10,bool);packet=chunk=0;max_log_error=0.;clipped_rows=0
    for path in sorted(directory.glob('block_*.npz')):
        data=history.load(path);assert all(len(v)==100 and np.isfinite(v).all() for v in data.values())
        for k in range(100):
            r={n:v[k] for n,v in data.items()};assert r['terminal_expected_verified']
            r['terminal_history']=r['terminal_expected_history'];receipt.consume(r)
            np.testing.assert_array_equal(r['terminal_info_history'][r['done']],r['terminal_expected_history'][r['done']])
            np.testing.assert_array_equal(r['terminal_info_history'][~r['done']],0)
            np.testing.assert_array_equal(r['normalized_terminal_history'][~r['done']],0)
            check_private(r,receipt);state=check_rms(r,state)
            log_error,clipped=check_gaussian(r);max_log_error=max(max_log_error,log_error);clipped_rows+=clipped
            history.episode_worlds(by_packet.get(packet,[]),r['done'])
            for row in by_packet.get(packet,[]):
                w=row['world'];bank=caches[int(r['stage_before'][w])]
                assert row['curriculum_stage']==r['stage_before'][w] and row['duration_s']==r['state'][w,0]*.0005 and row['physical_steps']==r['state'][w,0]
            for s in (1,2,3):
                mask=r['stage_after']==s;cache=caches[s]
                for n,key in (('param','param'),('q0','q0'),('obs0','obs0'),('native_reference','reference')):np.testing.assert_array_equal(r['bank_'+n][mask],cache[key][mask])
                done=mask&r['done'];public_reset=np.column_stack((cache['obs0'][done],np.zeros((done.sum(),6),np.float32)))
                np.testing.assert_array_equal(r['public_after'][done],public_reset)
                before=r['stage_before']==s;np.testing.assert_array_equal(r['delay_steps'][before],cache['param'][before,4])
            chunks.append(r);packet+=1
            if len(chunks)==50:
                starts=check_buffer(history.load(directory/f'rollout_{chunk:02d}.npz'),chunks,starts);chunks=[];chunk+=1
    assert packet==1200 and chunk==24 and receipt.done_rows==len(episodes)==26
    final=history.load(directory/'final_common.npz');receipt.final(final)
    for n in PRIVATE:
        key=dict(parking_seen='parking_0',parking_effective='parking_1',phase_reference='phase_0',phase_role='phase_1',phase_monitor='phase_2',command_total='command_0',command_prefix='command_1',command_stamp='command_2').get(n,n)
        np.testing.assert_array_equal(r['post_'+n],final[key])
    return dict(packets=packet,rollout_buffers=chunk,done_rows=receipt.done_rows,zero_valid_steps=cold.steps,rollout_valid_steps=receipt.steps,
        max_command_integral_error=receipt.max_integral_error,max_Gaussian_logprob_error=max_log_error,clipped_Gaussian_rows=clipped_rows,
        actual_terminal_and_all_returned481_verified=True,same_packet_post_nine_and_previous_and_bank_four_verified=True,
        RMS_normalization_reward_and_returns_reconstructed=True,GAE_returns_and_buffer_packet_identity_verified=True,actual_timeout_bootstrap_rows=0)


def run():
    self_check();receipt,a=inputs()
    reports={arm:review_arm(OUT/'runtime'/arm) for arm in ('CartLive3','CartReset3')}
    result=dict(round=377,verified=True,reports=reports,source_admission_sha256=sha(OUT/'source_admission.json'),execution_receipt_sha256=sha(OUT/'execution_receipt.json'),
        sources_unchanged=True,original_artifacts_unchanged=True,scope='Actual terminal/command/history/post/bank/RMS/Gaussian/GAE receipt;wheel/map/routing separately reviewed.',
        new_physics_steps=0,new_FD_calls=0,new_model_forward_rows=0,new_optimizer_updates=0,actual_timeout_bootstrap_exercised=False)
    (OUT/'history_PPO_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS377 stored histories/PPO/RMS',reports,flush=True)


if __name__=='__main__':run()
