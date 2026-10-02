"""Native schedule timing, public interpolation, original domain and resets."""
from dataclasses import replace
import numpy as np
import warp as wp
from native.environment import NativeEnv,reset_rows
from native.braking_reference import load_reference,update_reference
from probe_height_115_margin import cases


def check():
    for invalid in (dict(braking_reference=True,nominal_correction=True),dict(braking_reference=True,nominal_correction=None)):
        try:NativeEnv.height115_candidate(n=1,**invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid reference mode accepted')
    base=cases()[4]
    scenes=[base,replace(base,speed=-1.),replace(base,speed=.75,stand_height_m=.1375),replace(base,stand_height_m=.16)]
    env=NativeEnv.height115_candidate(n=4,scenario=scenes,residual_scale=0,braking_reference=True)
    plan=load_reference()
    try:
        env.reset();s=env.state.numpy();s[:,0]=4000;env.state.assign(s)
        args=[env.state,env.param,env.active,env.braking_plan,env.nominal_correction]
        wp.launch(update_reference,4,args);assert not env.nominal_correction.numpy().any()
        s[:,1]=2.0005;s[:,0]=4010;env.state.assign(s);wp.launch(update_reference,4,args)
        expected=np.stack([plan[0,0],plan[0,1],.25*plan[0,0],np.zeros(6)])
        np.testing.assert_allclose(env.nominal_correction.numpy(),expected,rtol=0,atol=1e-14)
        s[:,0]=4020;env.state.assign(s);wp.launch(update_reference,4,args)
        expected=np.stack([plan[1,0],plan[1,1],.25*plan[1,0],np.zeros(6)])
        np.testing.assert_allclose(env.nominal_correction.numpy(),expected,rtol=0,atol=1e-14)
        try:env.set_nominal_correction(np.zeros((4,6)))
        except ValueError:pass
        else:raise AssertionError('Manual overwrite of automatic reference accepted')
        before=env.nominal_correction.numpy().copy()
        env.mask.assign(np.array([1,0,0,0],np.int32));wp.launch(reset_rows,4,env.reset_args)
        assert not env.nominal_correction.numpy()[0].any()
        np.testing.assert_array_equal(env.nominal_correction.numpy()[1:],before[1:])
        env.reset();assert not env.nominal_correction.numpy().any()
        # The actual40-step graph must execute four reference slots itself.
        s=env.state.numpy();s[:,0]=4010;s[:,1]=2.0005;env.state.assign(s)
        env.step(np.zeros((4,3),np.float32))
        expected=np.stack([plan[3,0],plan[3,1],.25*plan[3,0],np.zeros(6)])
        np.testing.assert_allclose(env.nominal_correction.numpy(),expected,rtol=0,atol=1e-14)
        assert np.all(env.state.numpy()[:,0]==4050)
    finally:env.close()
    print('PASS native reference:5ms timing inside40-step graph, both signs/interpolation, zero regions, manual rejection and resets; fixture not full task')


if __name__=='__main__':check()
