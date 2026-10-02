"""One registered five-node coverage check of the executable baseline; no learning."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
from native.environment import NativeEnv
from probe_height_115_margin import cases

HEIGHTS = (.115, .16, .25, .30, .38)


def check():
    registration = json.loads((OUT / 'registered_cases.json').read_text())
    result = json.loads((OUT / 'height_panel.json').read_text())
    rows = result['episodes']
    assert len(rows) == len(registration['scenarios']) == 30
    assert not result['learning'] and not result['full_admission']
    for row, scene in zip(rows, registration['scenarios']):
        assert row['physical_steps'] == row['physical_evidence_steps'] > 0
        assert row['target_leg_m'] == scene['stand_height_m']
        assert row['baseline_version'] == 'height115-current-vmc-v2-request-state-candidate'
    groups = {}
    for i, height in enumerate(HEIGHTS):
        subset = rows[i * 6:(i + 1) * 6]
        groups[str(height)] = dict(physical=sum(r['physical_safety_passed'] for r in subset),
            task=sum(r['success'] for r in subset), completed=sum(r['reason'] == 'completed' for r in subset))
    assert groups == result['groups']
    for name, digest in result['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    print('CHECKED', groups, flush=True)


def run():
    assert not (OUT / 'height_panel.json').exists()
    scenes = [replace(s, stand_height_m=h) for h in HEIGHTS for s in cases()[:6]]
    registration = dict(heights=HEIGHTS, scenarios=[asdict(s) for s in scenes],
        scope='Existing six normal cases crossed with five existing design nodes, nominal parameters, zero Actor. Public engineering coverage; no holdout, no gain search. Discrete nodes do not certify the continuous interval or active 1.4rad design gate.')
    (OUT / 'registered_cases.json').write_text(json.dumps(registration, indent=2) + '\n')
    env = NativeEnv.height115_candidate(n=len(scenes), scenario=scenes, residual_scale=0)
    try:
        env.reset()
        rows = [None] * len(scenes)
        for _ in range(700):
            _, _, done, infos = env.step(np.zeros((len(scenes), 3), np.float32))
            for w in np.flatnonzero(done):
                if rows[w] is None:
                    rows[w] = {k: v for k, v in infos[w].items() if k != 'terminal_observation'}
            if all(row is not None for row in rows):
                break
        assert all(row is not None for row in rows)
        groups = {str(h): dict(physical=sum(r['physical_safety_passed'] for r in rows[i*6:(i+1)*6]),
            task=sum(r['success'] for r in rows[i*6:(i+1)*6]),
            completed=sum(r['reason'] == 'completed' for r in rows[i*6:(i+1)*6])) for i, h in enumerate(HEIGHTS)}
        sources = [Path(__file__), OUT / 'registered_cases.json'] + [ROOT / 'wheelleg_warp' / name for name in
            ('native/controller.py', 'native/environment.py', 'native/design.py', 'native/terrain.py', 'native/models.py', 'training_contract.py', 'probe_height_115_margin.py')]
        result = dict(episodes=rows, groups=groups, learning=False, full_admission=False,
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        (OUT / 'height_panel.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        env.close()
    check()


if __name__ == '__main__':
    check() if '--check' in sys.argv else run()
