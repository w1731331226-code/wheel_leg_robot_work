"""独立的 1 Nm 单图九臂、六世界留出试验入口；先预检台阶晚窗。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from probe_height_115_action_predict_single_graph import ROOT, run, sha

OUTPUT = ROOT / 'wheelleg_warp/results/height_115_action_predict_1nm_single_graph_20260928'
ROLE = 'public_115m_single_graph_common3_1Nm_5ms_world_holdout_prediction'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    output = parser.parse_args().output
    try:
        run(output, peak_nm=1.)
    except Exception as exc:
        if not (output / 'verification.json').exists():
            output.mkdir(parents=True, exist_ok=True)
            sources = ('wheelleg_warp/probe_height_115_action_predict_1nm.py',
                       'wheelleg_warp/probe_height_115_action_predict_single_graph.py',
                       'wheelleg_warp/probe_height_115_action_predict_loow.py',
                       'wheelleg_warp/probe_height_115_live_common_local.py',
                       'wheelleg_warp/native/controller.py')
            previous = subprocess.check_output(
                ['git', 'show', '666716a:wheelleg_warp/probe_height_115_action_predict_single_graph.py'], cwd=ROOT)
            (output / 'verification.json').write_text(json.dumps(dict(
                role=ROLE, status='failed_pre_registered_gate_or_execution',
                error=f'{type(exc).__name__}: {exc}', training=False, final_holdout_opened=False,
                preflight='world1 step late; any first-10-step motor clipping, command mismatch, inactive step, or unequal same-graph start stops collection',
                initial_motor_peak_Nm=1.,
                previous_single_graph_git_revision='666716a',
                previous_single_graph_source_sha256=hashlib.sha256(previous).hexdigest(),
                source_sha256={name: sha(ROOT / name) for name in sources},
                source_state_verification_sha256=sha(ROOT / 'wheelleg_warp/results/height_115_local_states_20260928/verification.json'),
                source_windows_sha256=sha(ROOT / 'wheelleg_warp/results/height_115_local_states_20260928/windows.npz')),
                ensure_ascii=False, indent=2) + '\n')
        raise
