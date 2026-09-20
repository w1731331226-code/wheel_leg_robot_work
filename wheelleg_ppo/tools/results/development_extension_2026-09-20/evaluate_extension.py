"""Frozen development-only B0 assessment; run from any working directory."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ppo_env import sample_scenario, heldout_combination
from prepare_ppo import evaluate


def hashes():
    old = json.loads((ROOT / 'tools/results/fixes_2026-09-17/final/source_sha256.json').read_text())
    current = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in old}
    assert current == old, 'Frozen baseline sources changed; review before running'
    return current


if __name__ == '__main__':
    sources = hashes()
    cases = [(f'development_{seed}', sample_scenario('validation', seed)) for seed in range(1000, 1024)]
    assert len(cases) == 24 and not any(heldout_combination(s) for _, s in cases)
    manifest = dict(protocol='preppo-v2-ramp', split='development_extension_only',
                    generator_split='validation', stage=3, seeds=list(range(1000, 1024)),
                    test_set_evaluated=False, source_sha256=sources,
                    cases=[dict(name=n, scenario=asdict(s)) for n, s in cases])
    path = OUT / 'preregistered_scenarios.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    assert not (OUT / 'baseline.json').exists(), 'Do not overwrite completed evidence'
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(evaluate, cases))
    assert len(rows) == 24 and hashes() == sources
    result = dict(method='B0', protocol=manifest['protocol'], test_set_evaluated=False,
                  success_count=sum(r['success'] for r in rows), total=len(rows), runs=rows)
    (OUT / 'baseline.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('SUMMARY', result['success_count'], '/', result['total'], flush=True)
