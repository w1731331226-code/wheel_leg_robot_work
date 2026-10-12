"""Synthetic CPU evidence fixtures; never qualify a real original-parking run."""
import copy
import numpy as np
from cartesian_classical_history_receipt import review_case, PRIVATE, fresh, pairs


def fixture(delay=20):
    n = 65; ids = np.array([7, 10, 12, 15, 6, 9, 11, 14, 8, 13] + [0]*9)
    public = np.zeros(38, np.float32); public[22:24] = .3
    q = np.zeros(17, np.float32); q[2:4] = [.4, 1]
    native_reference = np.zeros(10); native_reference[2] = .3
    native = np.zeros(30); native[1] = .3; native[12] = -1
    reference = np.array([0., -.3, 0., -.3])
    state = np.zeros(39); state[[1, 30, 31, 32, 33, 38]] = [-1, 1, 1, 1, 1e30, 1e30]
    param = np.zeros(17); param[4] = delay
    initial = dict(public=public[None, :], q=q[None, :], param=param[None, :],
        native_reference=native_reference[None, :], cart_reference=reference[None, :],
        cart_latent=np.zeros((1, 3)), cart_shadow=np.zeros((1, 22)), raw481=fresh(public[None, :]))
    requested = np.array([.1, .2, .3], np.float32); action = pairs(requested[None, :])[0]
    c = np.zeros((n, 38)); park = np.zeros((n, 30)); g = np.zeros((n, 62))
    filtered = np.zeros(3)
    ctrl = np.array([2., 2., 2., 2., .28125, .28125], np.float32)
    for step in range(n):
        before = filtered.copy(); filtered += np.clip(requested.astype(float)-filtered, -.01, .01)
        c[step, 0] = 1; c[step, 1:4] = before; c[step, 4:7] = filtered
        c[step, 7:13] = pairs(filtered[None, :])[0]
        park[step, :2] = [1, step+1]; park[step, 5:11] = action; park[step, 11:17] = action
        park[step, 17:23] = pairs(before[None, :])[0]; park[step, 23:29] = pairs(filtered[None, :])[0]
        g[step, :2] = [1, step+1]; g[step, 20:26] = ctrl
    pre = np.zeros((n, 39)); post = pre.copy()
    pre[:, :17] = q; post[:, :17] = q; pre[:, 33:39] = ctrl
    post[:, 18] = np.where(np.arange(n) < 40, 1, 2)
    p1 = public.copy(); p1[7] = 1; p1[32:38] = pairs(c[39:40, 4:7])[0].astype(np.float32)
    p2 = public.copy(); p2[7] = 2; p2[32:38] = pairs(c[-1:, 4:7])[0].astype(np.float32)
    first = initial['raw481'][0].copy()
    a = first[:390].reshape(10, 39).copy(); a[:-1] = a[1:].copy(); a[-1] = np.r_[p1, np.float32(.01)]
    inputs = np.zeros((9, 8), np.float32); inputs[-1, :6] = [.05]*4 + [.0625]*2
    elapsed = np.zeros(9, np.float32); elapsed[-1] = (40-delay)/40
    valid = np.zeros(10, np.float32); valid[-2:] = 1
    incoming2 = np.r_[a.ravel(), inputs.ravel(), elapsed, valid].astype(np.float32)
    a[:-1] = a[1:].copy(); a[-1] = np.r_[p2, np.float32(.02875)]
    inputs[:-1] = inputs[1:].copy(); inputs[-1, :6] = [.05]*4 + [.0625]*2
    elapsed[:-1] = elapsed[1:].copy(); elapsed[-1] = .625
    valid[-3:] = 1
    terminal = np.r_[a.ravel(), inputs.ravel(), elapsed, valid].astype(np.float32)
    reset_raw = fresh(public[None, :])[0]
    after = dict(public=public.copy(), param=param.copy(), q=q.copy(), v=np.zeros(16), ctrl=ctrl.copy(),
        controller_memory=native.copy(), native_reference=native_reference.copy(), guard_memory=np.zeros(16),
        parking_seen=np.array(0, np.int32), parking_effective=action.copy(), phase_reference=np.zeros(16),
        cart_latent=np.zeros(3), cart_shadow=np.zeros(22), cart_reference=reference.copy(),
        stopped_q=post[-1, :17].copy(), stopped_v=post[-1, 17:33].copy(),
        total=np.zeros(6), prefix=np.zeros((82, 6)), stamp=np.r_[0, np.full(81, -1, np.int32)],
        previous_sensor_steps=np.array(0, np.int64), active=np.array(1), done=np.array(0),
        targets=np.zeros(6), state=state.copy(), frames=reset_raw[:390].reshape(10, 39),
        inputs=reset_raw[390:462].reshape(9, 8), elapsed=reset_raw[462:471], valid=reset_raw[471:])
    total = np.zeros(6); prefix = np.zeros((82, 6)); stamp = np.full(82, -1, np.int32); stamp[0] = 0
    records = []; previous = 0
    for k, end in enumerate((40, 65)):
        start = 40*k
        for step in range(start, end):
            total += ctrl.astype(float)*.0005; prefix[(step+1)%82] = total; stamp[(step+1)%82] = step+1
        sensor_end = max(end-delay, 0); impulse = prefix[sensor_end%82]-prefix[previous%82]
        native_before = native.copy(); native_before[16:22] = c[end-1, 7:13]
        shadow = np.zeros(22); shadow[16:22] = pairs(c[end-1:end, 4:7])[0]
        guard_before = np.zeros(16); guard_before[0] = 1; guard_before[5:11] = ctrl; guard_before[15] = end
        private = dict(guard_memory=guard_before, parking_seen=np.array(0), parking_effective=action.copy(),
                       native_controller=native_before, phase_reference=np.zeros(16),
                       cart_latent=c[end-1, 4:7].copy(), cart_shadow=shadow, cart_reference=reference.copy())
        r = dict(prewait_public38=(p1 if k==0 else p2).copy(), prewait_count=np.array(end),
            previous_sensor_steps_before=np.array(previous), delay_steps=np.array(delay),
            command_total_before=total.copy(), command_prefix_before=prefix.copy(), command_stamp_before=stamp.copy(),
            interval_integral=impulse, interval_elapsed_s=np.array((sensor_end-previous)*.0005),
            terminal_worlds=np.array(k==1), postwait_public38=(p1 if k==0 else public).copy(),
            returned_raw481=(incoming2 if k==0 else reset_raw).copy())
        for name in PRIVATE:
            r[name+'_before'] = private[name].copy()
            if k==0:r[name+'_after'] = private[name].copy()
            else:
                key='controller_memory' if name=='native_controller' else name
                r[name+'_after'] = after[key].copy()
        if k==1:
            total[:]=0;prefix[:]=0;stamp[:]=-1;stamp[0]=0
        r.update(command_total_after=total.copy(), command_prefix_after=prefix.copy(), command_stamp_after=stamp.copy(),
                 post_previous_sensor_steps=np.array(0 if k==1 else sensor_end))
        records.append(r);previous=sensor_end
    delivery={name:np.stack([r[name] for r in records]) for name in records[0]}
    actor=np.stack([np.r_[x,x,action] for x in (first,incoming2)]).astype(np.float32)
    case=dict(seed='synthetic_H1_CPU_only',scenario=dict(delay_ms=delay*.5))
    clock=np.column_stack((np.arange(1,n+1),np.arange(n)*.0005,np.arange(1,n+1)*.0005))
    trace=np.zeros((n,112));trace[:,0]=1;trace[:,1:4]=clock
    return dict(case=case,row=dict(**case,physical_steps=n,duration_s=n*.0005),world=0,initial=initial,actor=actor,
        full=dict(pre=pre,post=post,trace=trace),cart=dict(trace=c,clock=clock,
            terminal=dict(latent=c[-1,4:7].copy(),shadow=records[-1]['cart_shadow_before'].copy(),
                          reference=reference.copy(),canonical=c[-1,7:13].copy())),
        parking=park,guard=g,actual_terminal=terminal,after_reset=after,ids=ids,delivery=delivery)


def run():
    data=fixture();result=review_case(**data)
    assert result['history_rows']==2 and result['steps']==65 and result['actual_terminal_and_all_returned481_verified']
    review_case(**fixture(0))
    corruptions=(('actor',lambda x:x['actor'].__setitem__((0,0),1)),
        ('clock',lambda x:x['cart']['clock'].__setitem__((0,0),2)),
        ('terminal',lambda x:x['actual_terminal'].__setitem__(389,.5)),
        ('integral',lambda x:x['delivery']['interval_integral'].__setitem__((0,0),1)),
        ('stamp',lambda x:x['delivery']['command_stamp_before'].__setitem__((0,20),-1)),
        ('delay',lambda x:x['delivery']['delay_steps'].__setitem__(0,19)),
        ('guard_post',lambda x:x['delivery']['guard_memory_after'].__setitem__((0,15),0)),
        ('original_parking_clear',lambda x:x['delivery']['parking_effective_after'].__setitem__((1,0),0)),
        ('native_reset',lambda x:x['after_reset']['controller_memory'].__setitem__(5,1)),
        ('latent_leak',lambda x:x['delivery']['prewait_public38'].__setitem__((0,32),1)))
    for name,change in corruptions:
        bad=copy.deepcopy(data);change(bad)
        try:review_case(**bad)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Corrupt '+name+' accepted')
    for missing in ('interval_integral','returned_raw481','cart_shadow_after'):
        bad=copy.deepcopy(data);del bad['delivery'][missing]
        try:review_case(**bad)
        except ValueError:pass
        else:raise AssertionError('Missing '+missing+' silently accepted')
    print('PASS381 CPU H1: positive-delay/partial terminal/original parking hold/actual private reset and corrupt/missing evidence rejected;synthetic only,0physics/FD/GPU/NN')


if __name__=='__main__':run()
