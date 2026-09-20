"""不读真实终评集：用合成已完成结果核对配对汇总与不完整训练拒绝。"""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import train_compare as task


def check():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        protocol = dict(name='synthetic', backends=['cpu', 'warp'], seeds=[1609,1610,1611],
                        config=dict(policy_steps=2000000), final_cases=[], known_limitations=['合成测试'])
        for seed in protocol['seeds']:
            for backend in protocol['backends']:
                run = root / f'{backend}_{seed}'
                run.mkdir()
                task.write(run/'completed.json', dict(passed=True, smoke=False, policy_steps=2000000,
                           train_seconds=10.,total_seconds=12.,initial_policy_sha256=str(seed)))
                task.write(run/'selection.json', dict(best=None))
                for kind in ('selected', 'last'):
                    task.write(run/f'final_{kind}.json', dict(summary=dict(total=64,success_count=50 if backend=='cpu' else 48,
                               complete=True,mean_yaw_score_deg=.5 if backend=='cpu' else .6)))
        with patch.object(task, 'load_protocol', return_value=protocol):
            task.compare(root)
            result = json.loads((root/'comparison.json').read_text())
            assert len(result['rows']) == 12 and len(result['aggregates']) == 4
            assert result['aggregates'][0]['success_mean'] == 50
            assert result['aggregates'][2]['success_mean'] == 48
            assert result['aggregates'][0]['success_std'] == 0
            assert (root/'COMPARISON.md').exists()
            bad = root/'cpu_1609/completed.json'
            data=json.loads(bad.read_text()); data['policy_steps']=4000;task.write(bad,data)
            try:
                task.compare(root)
            except AssertionError:
                pass
            else:
                raise AssertionError('不完整预算不能生成正式比较')
    print('PASS：合成配对汇总、两类检查点、未完成预算拒绝；未接触真实终评集')


if __name__ == '__main__':
    check()
