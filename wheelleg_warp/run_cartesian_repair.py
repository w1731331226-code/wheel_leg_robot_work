"""One v2 repair queue using the frozen364 engine with scoped, verified bindings."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import run_cartesian_delivery as engine
import cartesian_pair_runtime as cart
from cartesian_pair_kernel_v2 import deliver_accepted_nominal
from native.environment import PUBLIC_ACTUATOR_GAIN_UPPER
from review_yaw_sector import ROOT,sha
from run_wheel_interference import runtime
from dashboard.live_env import atomic_json

PARENT=engine.OUT
OUT=PARENT.parent/'cartesian_pair_delivery_repair_v2'
ORIGINAL_INSTRUMENT=cart.instrument


def require_accepted_nominal(raw):
    if not raw.nominal_correction_enabled or not raw.shared_reference_enabled or raw.control_kernel is not cart.NOMINAL or not np.array_equal(raw.actuator_gain_upper,PUBLIC_ACTUATOR_GAIN_UPPER):
        raise ValueError('V2 requires the qualified shared accepted-Nom interface and public gain bounds')


def repaired_instrument(factory,cases,directory,mode):
    result=ORIGINAL_INSTRUMENT(factory,cases,directory,mode);raw=result[0]
    require_accepted_nominal(raw)
    raw._cartesian_topology.update(accepted_nominal_mode=1,candidate_kernel='cartesian_pair_kernel_v2.deliver_accepted_nominal',
        candidate_kernel_sha256=sha(ROOT/'wheelleg_warp/cartesian_pair_kernel_v2.py'))
    atomic_json(directory/'cartesian_topology.json',raw._cartesian_topology)
    return result


def inputs():
    p=json.loads((OUT/'runtime_proposal.json').read_text());old=json.loads((PARENT/'runtime_proposal.json').read_text())
    for key in ('cases','conditions','evaluations','actual_graph_world_step_budget','constructor_FD_budget','per_job_first_episode_steps_cap','actor_row_budget','profile'):
        assert p[key]==old[key],key
    assert p['parent_failed_proposal_sha256']==sha(PARENT/'runtime_proposal.json') and p['parent_failure_sha256']==sha(PARENT/'runtime/failure.json')
    assert not (PARENT/'runtime/completion.json').exists()
    repair=json.loads((PARENT/'nominal_mode_repair_review.json').read_text());assert repair['verified'] and not repair['installed']
    for mapping in (p['source_sha256'],p['evidence_sha256'],repair['source_sha256']):
        assert all(sha(ROOT/f)==h for f,h in mapping.items())
    return p


def self_check():
    from test_cartesian_pair_runtime import run as routing
    with patch.object(cart,'deliver',deliver_accepted_nominal):routing()
    good=dict(nominal_correction_enabled=True,shared_reference_enabled=True,control_kernel=cart.NOMINAL,actuator_gain_upper=np.array(PUBLIC_ACTUATOR_GAIN_UPPER))
    require_accepted_nominal(SimpleNamespace(**good))
    for field,value in [('nominal_correction_enabled',False),('shared_reference_enabled',False),('control_kernel',None),('actuator_gain_upper',np.ones(6))]:
        try:require_accepted_nominal(SimpleNamespace(**{**good,field:value}))
        except ValueError:pass
        else:raise AssertionError('Unqualified nominal interface admitted')
    assert cart.deliver is not deliver_accepted_nominal
    print('PASS365 v2 routing, scoped restoration and accepted-Nom rejection cases;0physics',flush=True)


def freeze():
    inputs();assert not (OUT/'runtime_admission.json').exists();self_check()
    old=json.loads((PARENT/'runtime_admission.json').read_text());source=old['source_sha256'].copy()
    assert all(sha(ROOT/f)==h for f,h in source.items())
    for name in ('cartesian_pair_kernel_v2.py','test_cartesian_nominal_mode.py','run_cartesian_repair.py'):
        file=ROOT/'wheelleg_warp'/name;source[str(file.relative_to(ROOT))]=sha(file)
    atomic_json(OUT/'runtime_admission.json',dict(round=365,verified=True,proposal_sha256=sha(OUT/'runtime_proposal.json'),
        repair_review_sha256=sha(PARENT/'nominal_mode_repair_review.json'),failed_admission_sha256=sha(PARENT/'runtime_admission.json'),
        source_sha256=source,runtime=runtime(),new_physics_steps=0,new_learning_samples=0,full_delivery_not_yet_verified=True,
        original_engine_round=364,accepted_nominal_check_before_first_graph_launch=True))


def verify():
    p=inputs();a=json.loads((OUT/'runtime_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'runtime_proposal.json')
    assert a['repair_review_sha256']==sha(PARENT/'nominal_mode_repair_review.json')
    assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items()) and a['runtime']==runtime()
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Need2GiB diagnostic archive space')
    return p,a


def run():
    verify();assert not (OUT/'runtime').exists() and not (OUT/'run_receipt.json').exists()
    error=None
    try:
        with patch.object(engine,'OUT',OUT),patch.object(engine,'QUEUE',OUT/'runtime'),patch.object(engine,'verify',verify),\
             patch.object(cart,'deliver',deliver_accepted_nominal),patch.object(cart,'instrument',repaired_instrument):
            engine.run()
    except BaseException as failure:
        error=repr(failure);raise
    finally:
        complete=OUT/'runtime/completion.json';failed=OUT/'runtime/failure.json'
        result=complete if complete.exists() else failed if failed.exists() else None
        atomic_json(OUT/'run_receipt.json',dict(round=365,engine_generation_round=364,engine_sha256=sha(ROOT/'wheelleg_warp/run_cartesian_delivery.py'),
            status='complete' if complete.exists() else 'failed' if failed.exists() else 'not_started',error=error,
            result_path=None if result is None else str(result.relative_to(ROOT)),result_sha256=None if result is None else sha(result),
            scoped_bindings_restored=engine.OUT==PARENT and cart.instrument is ORIGINAL_INSTRUMENT and cart.deliver is not deliver_accepted_nominal,
            original_failure_sha256=sha(PARENT/'runtime/failure.json'),new_learning_samples=0,formal5_admitted=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.command]()
