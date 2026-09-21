"""独立核对原生候选报告的回合回报与实际逐步奖励总和，不修改候选代码。"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from native.environment import NativeEnv

if __name__ == '__main__':
    env = NativeEnv(8, stage=3, seed=730000)
    env.reset()
    totals = np.zeros(8)
    rows = []
    for step in range(1200):
        obs, rewards, done, infos = env.step(np.tile([.1, -.1, .1], (8, 1)))
        totals += rewards
        for i in np.flatnonzero(done):
            reported = float(infos[i]['episode']['r'])
            rows.append(dict(world=int(i), step=step + 1, reason=infos[i]['reason'],
                             reward_sum=float(totals[i]), reported=reported,
                             error=reported-float(totals[i])))
            totals[i] = 0
        if len(rows) >= 8:
            break
    env.close()
    result = dict(passed=bool(rows) and all(abs(r['error']) < 1e-5 for r in rows), episodes=rows)
    output = Path(sys.argv[1])
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    assert result['passed'], '回合回报与实际奖励累计不一致，候选不可作为已验证正式基线'
