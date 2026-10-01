"""Check the shared execution packet in virtual6; no learning or full-task claim."""
from dataclasses import replace
import numpy as np
from native.environment import NativeEnv
from probe_height_115_margin import cases
from training_contract import observation_spec


def check():
    assert observation_spec('request_state_v1',3)==observation_spec('request_state_v1',6)
    try:NativeEnv(n=1,observation_contract='unknown',bank_factory=lambda *args:(_ for _ in ()).throw(AssertionError('physics constructed')))
    except ValueError:pass
    else:raise AssertionError('Unknown observation contract accepted')
    env=NativeEnv.height115_candidate(n=1,scenario=[replace(cases()[0],delay_ms=20.)],residual_mode='virtual6')
    try:
        assert env.observation_space.shape==(38,) and env.action_space.shape==(6,)
        initial=env.reset();obs,reward,done,_=env.step(np.array([[.8,-.6,.2,-.3,.1,-.1]],np.float32))
        assert not done.any() and np.isfinite(reward).all()
        np.testing.assert_array_equal(obs[:,:32],initial[:,:32])
        np.testing.assert_allclose(obs[:,32:],[[.4,-.4,.2,-.3,.1,-.1]],rtol=0,atol=1e-7)
        np.testing.assert_allclose(obs[:,32:],env.k['state'].numpy()[:,16:22],rtol=0,atol=1e-7)
        assert np.all(env.state.numpy()[:,37]==40)
        assert np.all(env.reset()[:,32:]==0)
    finally:env.close()
    print('PASS virtual6: same38D packet, current six requests,20ms physical delay, native40-step evidence/reset and invalid-mode rejection')


if __name__=='__main__':check()
