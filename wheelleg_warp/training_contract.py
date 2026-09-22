"""Small shared guards for terrain selection and frozen training inputs."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASK_CONTRACT_VERSION=2


def yaw_total(*summaries):
    values=[s.get('mean_yaw_score_deg') for s in summaries]
    return sum(values) if all(v is not None and math.isfinite(v) for v in values) else math.inf


def promotion_allowed(candidate,reference):
    """An incomplete run or loss of any already-successful legacy case vetoes promotion."""
    if not all(candidate[k]['complete'] for k in ('terrain','legacy')):return False
    new=candidate['legacy_runs'];old=reference['legacy_runs']
    identity=lambda r:(r['seed'],json.dumps(r['scenario'],sort_keys=True))
    if [identity(r) for r in new]!=[identity(r) for r in old]:
        raise ValueError('原回归场景不同，不能配对晋升')
    if any(r.get('task_contract_version')!=TASK_CONTRACT_VERSION for r in new+old):
        raise ValueError('旧判据成绩不能用于当前选模，必须重新评估')
    return all(not r['success'] or n['success'] for n,r in zip(new,old))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_hashes(prefix):
    return dict(checkpoint_sha256=digest(str(prefix)+'.zip'),normalization_sha256=digest(str(prefix)+'.pkl'))


def verify_checkpoint(prefix,record):
    for key,value in checkpoint_hashes(prefix).items():
        if record.get(key)!=value:raise ValueError(f'检查点或归一化文件未冻结/已改变: {prefix} ({key})')


def source_files(entrypoint):
    paths=['wheelleg_warp/'+name for name in (
        'training_contract.py','terrain_eval.py','benchmark_parallel.py','native/terrain.py',
        'native/terrain_env.py','native/models.py','native/environment.py','native/controller.py',
        'native/live.py','dashboard/live_env.py')]
    paths+=['wheelleg_ppo/tools/'+name+'.py' for name in (
        'ppo_env','pretrain_yaw','wheelleg_sim','hardware_profile','rm_controller','model_lqr','state_estimation')]
    paths.append('wheelleg_ppo/xml/wheelleg.xml')
    paths.append(str(Path(entrypoint).resolve().relative_to(ROOT)))
    return paths


def source_hashes(entrypoint):return {name:digest(ROOT/name) for name in source_files(entrypoint)}


def verify_frozen(protocol,entrypoint):
    if protocol.get('task_contract_version')!=TASK_CONTRACT_VERSION:
        raise ValueError('训练协议使用旧任务判据，必须在独立目录重新冻结及评估')
    hashes=protocol.get('source_sha256',{})
    if not set(source_files(entrypoint)).issubset(hashes):raise ValueError('训练协议缺少必需源码冻结')
    for name,value in hashes.items():
        if digest(ROOT/name)!=value:raise ValueError(f'冻结源码已改变: {name}')
    verify_checkpoint(protocol['checkpoint'],protocol)


def verify_protocol(output,entrypoint,expected=None):
    protocol=json.loads((Path(output)/'protocol.json').read_text())
    if expected is not None and protocol!=expected:raise ValueError('运行期间训练协议已改变')
    verify_frozen(protocol,entrypoint)
    return protocol
