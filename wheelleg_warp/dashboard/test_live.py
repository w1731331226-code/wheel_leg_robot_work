"""记录器不能改变真实轨迹/RNG；归档可读且标明训练原始画面。"""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from ppo_env import WheelLegEnv,Scenario
from dashboard.live_env import LiveRecorder

with tempfile.TemporaryDirectory() as directory:
    plain=WheelLegEnv(scenario=Scenario(),max_seconds=.08)
    recorded=LiveRecorder(WheelLegEnv(scenario=Scenario(),max_seconds=.08),directory,'cpu')
    a,_=plain.reset(seed=42);b,_=recorded.reset(seed=42)
    np.testing.assert_array_equal(a,b)
    for _ in range(4):
        action=np.array([.1,-.1,.1])
        a=plain.step(action);b=recorded.step(action)
        np.testing.assert_array_equal(a[0],b[0]);assert a[1:4]==b[1:4]
        np.testing.assert_array_equal(plain.data.qpos,recorded.unwrapped.data.qpos)
        assert plain.np_random.bit_generator.state==recorded.unwrapped.np_random.bit_generator.state
    root=Path(directory)
    with np.load(root/'episodes/000001/trajectory.npz',allow_pickle=False) as trace:
        assert trace['qpos'].shape[0]==4
        np.testing.assert_array_equal(trace['qpos'][-1],plain.data.qpos)
    meta=json.loads((root/'episodes/000001/metadata.json').read_text())
    assert meta['status']=='completed' and meta['kind']=='actual_training_frames'
    latest=json.loads((root/'latest.json').read_text())
    with np.load(root/'latest.npz',allow_pickle=False) as data:
        assert int(data['frame'])==latest['frame']
    plain.close();recorded.close()
print('PASS：采集前后状态/奖励/RNG逐值一致，真实训练轨迹保存与来源标记正确')
