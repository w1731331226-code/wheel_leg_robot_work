"""延迟传播的闭式核验，以及旧预测/LP接口的归档回归。"""
import json
import numpy as np
from probe_height_115_recovery_feedback import predict_active
from probe_height_115_action_predict_loow import ROOT, DT, forecast, forecast_acceleration, ACTIVE
from probe_height_115_live_braking import solve_current, solve_from_acceleration
from native.terrain import model, HeightTerrainScenario
from audit_height_115_acceleration_trend import propagate


def run():
    q=np.array([.5,-.7,.52,-.72]);v=np.array([.01,-.02,.03,-.04]);g=np.arange(12).reshape(4,3)/7
    a0=np.array([1.,-2.,3.,-4.]);pa=np.array([.2,-.3,.4]);pv=v-DT*(a0+g@pa)
    actions=np.array([[1.,0,0],[0,1.,0],[0,0,1.],[-1.,2.,-3.]])
    for n in (0,1,4):
        qh,vh,ah=predict_active(q,v,pv,pa,g,actions[:n])
        acc=a0[None,:]+actions[:n]@g.T
        expected_q=q+n*DT*v+DT**2*np.sum((n-np.arange(n)-.5)[:,None]*acc,axis=0)
        assert np.allclose(qh,expected_q,atol=1e-12,rtol=0)
        assert np.allclose(vh,v+DT*acc.sum(axis=0),atol=1e-12,rtol=0)
        assert np.allclose(ah,a0,atol=1e-10,rtol=0)
    assert np.array_equal(q,[.5,-.7,.52,-.72]) and np.array_equal(v,[.01,-.02,.03,-.04])
    ppv=pv-DT*(a0-np.array([2.,-3.,4.,-5.])*DT)
    previous_a=(pv-ppv)/DT
    jerk=(a0-previous_a)/DT
    qh,vh,ah=propagate(q,v,pv,ppv,pa,np.zeros(3),g,actions,True)
    acc=a0[None,:]+np.arange(1,5)[:,None]*DT*jerk+actions@g.T
    assert np.allclose(qh,q+4*DT*v+DT**2*np.sum(np.array([3.5,2.5,1.5,.5])[:,None]*acc,axis=0),atol=1e-12,rtol=0)
    assert np.allclose(vh,v+DT*acc.sum(axis=0),atol=1e-12,rtol=0)
    assert np.allclose(ah,a0+4*DT*jerk,atol=1e-9,rtol=0)
    zero=np.zeros(3);held=np.zeros((4,3));vv=np.zeros(4)
    vp=vv-DT*np.full(4,5.);vpp=vp-DT*np.full(4,3.);vppp=vpp-DT*np.full(4,1.)
    limited=propagate(q,vv,vp,vpp,zero,zero,g,held,2,older_v=vppp,older_action=zero)
    linear=propagate(q,vv,vp,vpp,zero,zero,g,held,1)
    assert all(np.allclose(a,b,atol=1e-10,rtol=0) for a,b in zip(limited,linear))
    for older_velocity,commands in ((vpp-DT*np.full(4,5.),held),(vppp,actions)):
        limited=propagate(q,vv,vp,vpp,zero,zero,g,commands,2,older_v=older_velocity,older_action=zero)
        constant=propagate(q,vv,vp,vpp,zero,zero,g,commands,0)
        assert all(np.array_equal(a,b) for a,b in zip(limited,constant))
    history_only=propagate(q,vv,vp,vpp,zero,zero,g,actions,3,older_v=vppp,older_action=zero)
    linear_with_input=propagate(q,vv,vp,vpp,zero,zero,g,actions,1)
    assert all(np.allclose(a,b,atol=1e-10,rtol=0) for a,b in zip(history_only,linear_with_input))
    history_changed=propagate(q,vv,vp,vpp,zero,np.ones(3),g,actions,3,older_v=vppp,older_action=zero)
    constant_changed=propagate(q,vv,vp,vpp,zero,np.ones(3),g,actions,0)
    assert all(np.array_equal(a,b) for a,b in zip(history_changed,constant_changed))
    folder=ROOT/'wheelleg_warp/results'
    fit=json.loads((folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json').read_text())
    arc=json.loads((folder/'height_115_local_states_20260928/verification.json').read_text())
    m=model(HeightTerrainScenario(**arc['events'][fit['selected_event_ids'][3]]['scenario']))
    qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    gain=np.array(fit['folds'][3]['gain']);r=fit['folds'][3]['reserves'];reserve=(r['actual_A_length_m'],r['eight_joint_margin_rad'])
    for name,age in [('height_115_recovery_feedback_delay05_20260929',1),('height_115_recovery_feedback_delay2_20260929',4)]:
        d=json.loads((folder/name/'verification.json').read_text())
        row=next(x for x in d['rows'] if x['step']>=41+age and np.any(x['action']))
        with np.load(folder/name/'trace.npz') as raw:z={k:raw[k] for k in raw.files}
        k=row['step']-age;q=z['pre_q'][k,1];v=z['pre_v'][k,1];pv=z['pre_v'][k-1,1,va];pa=z['actions'][k-1,1]
        a0=(v[va]-pv)/DT-gain@pa;nom=z['nominal'][k,1]
        x=solve_current(m,q,v,pv,pa,gain,reserve,nom);y=solve_from_acceleration(m,q,v,a0,gain,reserve,nom)
        assert x[3] is None and y[3] is None
        assert np.array_equal(x[0],y[0]) and np.allclose(x[0],row['action'],atol=1e-10,rtol=0)
        limits=np.array([m.jnt_range[m.joint(n).id] for n in ACTIVE])
        s=dict(q0=q[qa],v0=v[va],vprev=pv,coeff=np.array([[-pa[0],-pa[1],-pa[2]]]),active_joint_limits=limits)
        old=forecast(s,gain);new=forecast_acceleration(q[qa],v[va],a0[None,:],limits)
        assert all(np.array_equal(a,b) for a,b in zip(old,new))
    print('PASS closed-form delayed propagation and archived forecast/LP interface equivalence')


if __name__=='__main__':run()
