"""Reuse round25 standard PPO engineering probe for torque6; no long training."""
from pathlib import Path
import importlib.util,json
HERE=Path(__file__).resolve().parent
source=HERE.parent/'twentyfifth_round_native_admission_20261003/short_ppo.py'
spec=importlib.util.spec_from_file_location('round25_short_ppo',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.OUT=HERE
module.run('torque6')
from training_contract import digest
(HERE/'ppo_driver_sha256.json').write_text(json.dumps({str(p.relative_to(HERE.parents[2])):digest(p) for p in (Path(__file__),source)},indent=2)+'\n')
