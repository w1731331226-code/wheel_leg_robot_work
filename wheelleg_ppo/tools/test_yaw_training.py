"""Small checks for the scientific score and rollout-boundary checkpoint schedule."""
from types import SimpleNamespace
from unittest.mock import patch
from pretrain_yaw import ROOT, self_check, verify_sources
from train_yaw import SelectionCallback


if __name__ == '__main__':
    self_check()
    config = dict(evaluation_interval=20000, curriculum=[dict(start=0,stage=1),
                  dict(start=20000,stage=2),dict(start=100000,stage=3)])
    changes, saves = [], []
    env = SimpleNamespace(set_attr=lambda key,value: changes.append((key,value)))
    callback = SelectionCallback(config, {}, None, False)
    callback.model = SimpleNamespace(get_env=lambda:env)
    callback.save_and_evaluate = lambda: saves.append(callback.num_timesteps)
    for step in (0, 2000, 20000, 100000):
        callback.num_timesteps = step
        callback._on_rollout_start()
    assert saves == [20000, 100000]
    assert changes == [('stage',1),('stage',1),('stage',2),('stage',3)]
    callback.smoke = True
    callback.num_timesteps = 20000
    callback._on_rollout_start()
    assert saves == [20000, 100000]
    print('PASS: checkpoints after updates, curriculum boundaries, smoke does not select checkpoints')
    frozen = ROOT/'tools/results/yaw_precision_v2_2026-09-21'
    manifest = verify_sources(frozen)
    assert manifest['protocol'] == 'yaw-precision-v2'
    # A changed dependency must invalidate the selected version's manifest.
    with patch('pretrain_yaw.hashlib.sha256', return_value=SimpleNamespace(hexdigest=lambda:'changed')):
        try: verify_sources(frozen)
        except AssertionError: pass
        else: raise AssertionError('Changed source was accepted')
    print('PASS: v2 manifest selected explicitly; changed dependencies rejected')
