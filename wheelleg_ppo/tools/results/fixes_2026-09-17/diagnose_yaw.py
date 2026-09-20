"""Reproduce four development yaw failures and the fixed response, without test-set use."""
import contextlib
import io
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import wheelleg_sim as sim
from ppo_env import WheelLegEnv, sample_scenario

OUT = Path(__file__).resolve().parent

def run(case):
    version, seed = case
    sim.YAW6_DAMPING = .6 if version == 'before' else 2.
    sim.YAW6_LIMIT = .04 if version == 'before' else .12
    original = sim.control
    trace = []
    def observe(m, d, st, residual=None):
        original(m, d, st, residual)
        yaw = sim.euler(d)[2]
        err = math.atan2(math.sin(yaw-st.yaw_target), math.cos(yaw-st.yaw_target))
        raw = -.4*err-sim.YAW6_DAMPING*st.yw_f
        limit = sim.YAW6_LIMIT*st.lqr6.mass_scale*sim.hw.WHEEL_RADIUS/.025
        trace.append([d.time, yaw, st.yw_f, raw, limit, st.yaw_torque,
                      d.ctrl[4], d.ctrl[5]])
    sim.control = observe
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            e = WheelLegEnv(scenario=sample_scenario('validation', seed)); e.reset(seed=0)
            while not e.done:
                e.step(np.zeros(3))
        a = np.asarray(trace)
        np.savez_compressed(OUT/f'yaw_{version}_{seed}.npz',trace=a,
                           columns=['time','yaw','filtered_yaw_rate','raw_pd','limit','yaw_torque','wheel_l','wheel_r'])
        result = dict(version=version,seed=seed,metrics=e.metrics(),
                      yaw_limited_steps=int(np.sum(abs(a[:,3])>a[:,4])),
                      peak_yaw_rate=float(max(abs(a[:,2]))))
        if version == 'after':
            assert result['metrics']['success'], result
        return result
    finally:
        sim.control = original

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run, [(v,s) for v in ('before','after') for s in (3,4,8,9)]))
    (OUT/'yaw_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
