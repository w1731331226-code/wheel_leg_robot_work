"""Dispatch guard checks using real immutable checkpoints, no training launch."""
from pathlib import Path
import tempfile,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_queue as q
p,registration=q.protocol();run=q.P/'runs/M3/1609'
try:q.complete(run,p)
except AssertionError as e:assert 'Budget incomplete' in str(e)
else:raise AssertionError('Live incomplete budget accepted')
with tempfile.TemporaryDirectory(prefix='wheelleg_queue_guard_',dir='/tmp') as directory:
    d=Path(directory);fixture=dict(p,policy_steps_per_seed=40000);records=[]
    for steps in (20000,40000):
        row=q.read(run/f'step_{steps}.json');row['path']=str(d/f'step_{steps}')
        for suffix in ('.zip','.pkl'):(d/f'step_{steps}{suffix}').symlink_to(run/f'step_{steps}{suffix}')
        (d/f'step_{steps}.json').write_text(json.dumps(row));records.append(row)
    best=min(records,key=lambda r:(r['summary']['total']-r['summary']['success_count'],r['summary']['mean_yaw_score_deg'],r['policy_steps']))
    selection=dict(protocol_sha256=q.sha(q.P/'protocol.json'),consumed_policy_steps=40000,evaluation_count=2,best=best)
    last=dict(records[-1],protocol_sha256=q.sha(q.P/'protocol.json'),consumed_policy_steps=40000)
    for name,value in [('selection.json',selection),('last_checkpoint.json',last),('progress.json',dict(executed_policy_steps=40000))]:(d/name).write_text(json.dumps(value))
    assert q.complete(d,fixture)==selection
    (d/'progress.json').write_text('{"executed_policy_steps":40100}')
    try:q.complete(d,fixture)
    except AssertionError:pass
    else:raise AssertionError('Unsaved budget accepted')
    (d/'progress.json').write_text('{"executed_policy_steps":40000}')
    broken=dict(last,checkpoint_sha256='0'*64);(d/'last_checkpoint.json').write_text(json.dumps(broken))
    try:q.complete(d,fixture)
    except AssertionError:pass
    else:raise AssertionError('Changed checkpoint hash accepted')
assert len(registration['queue'])==9
print('PASS exact-budget/metric/hash/finite guard; real completed40k fixture accepted; live partial and unsaved/tampered rejected; no dispatch')
