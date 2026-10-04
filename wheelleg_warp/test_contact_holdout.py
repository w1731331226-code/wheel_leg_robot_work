"""Kernel event selection and actual factory CUDA-buffer lifetime regression."""
import gc,json,weakref
import numpy as np
import warp as wp
import collect_contact_holdout as c


def run():
    q=wp.zeros((1,17));v=wp.zeros((1,16));pre=wp.zeros((1,101),dtype=c.D)
    ever=wp.zeros(1,dtype=int);filled=wp.zeros((1,3),dtype=int);snap=wp.zeros((1,3,103),dtype=c.D)
    active=wp.array(np.ones(1,np.int32));flags=wp.zeros((1,2),dtype=int)
    for time,mask in [(0,0),(.0005,4),(1.4995,4),(1.5,4),(3.5,4)]:
        a=np.zeros((1,101));a[0,49]=time;pre.assign(a);flags.assign(np.array([[0,mask]],np.int32))
        wp.launch(c.save_holdout,1,[q,v,active,flags,pre,ever,filled,snap])
    np.testing.assert_array_equal(filled.numpy(),[[1,1,1]])
    a=snap.numpy();np.testing.assert_array_equal(a[0,:,49],[1.5,3.5,.0005])
    np.testing.assert_array_equal(a[0,2,101:],[0,4]);before=a.copy()
    wp.launch(c.save_holdout,1,[q,v,active,flags,pre,ever,filled,snap]);np.testing.assert_array_equal(snap.numpy(),before)
    np.testing.assert_array_equal(q.numpy(),0);np.testing.assert_array_equal(v.numpy(),0)
    old=c.experiment.evaluate;proof={}
    def inspect(cases,*args):
        env=c.experiment.raw_env(cases,'diff3')
        refs=[weakref.ref(b) for b in env._holdout_buffers];gc.collect()
        assert all(r() is not None for r in refs)
        env.reset();assert all(r() is not None for r in refs)
        proof.update(kernel_events=True,graph_buffer_owners_alive=True,factory_reset_checked=True,rollout_steps=0)
        env.close();raise RuntimeError('self-check complete')
    c.experiment.evaluate=inspect
    try:
        p=json.loads((c.OUT/'registration.json').read_text());k=json.loads((c.OUT/'collector_contract.json').read_text())
        try:c.capture(p['cases'],None,None,k['classical_B1'],c.OUT,'self-check')
        except RuntimeError as e:assert str(e)=='self-check complete'
    finally:c.experiment.evaluate=old
    assert proof
    c.atomic_json(c.OUT/'collector_self_check.json',proof)
    print('PASS',proof)


if __name__=='__main__':run()
