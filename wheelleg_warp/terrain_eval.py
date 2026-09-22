"""Deterministic evaluation for fixed terrain-v1 cases."""
from dataclasses import asdict
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from native.terrain_env import TerrainEnv
from training_contract import TASK_CONTRACT_VERSION


def validate_terrain_rows(rows):
    if not rows:raise ValueError('评价场景不能为空')
    for row in rows:
        if row.get('task_contract_version')!=TASK_CONTRACT_VERSION:raise ValueError('旧评价记录不能当作新任务契约成绩；请重新评估')
        goal=row.get('task_goal_progress_m')
        if goal is None or not np.isfinite(goal) or goal<=0:raise ValueError('任务目标必须为有限正数')
        evidence=bool(row['terrain_passed'] and row['terrain_exit_passed'])
        if bool(row['terrain_evidence_passed'])!=evidence:raise ValueError('通过证据字段不一致')
        if row['success'] and (row['reason']!='completed' or not evidence):raise ValueError('成功却没有完整地形通过证据')
    return rows


def summarize_terrain(rows,expected_seeds):
    from pretrain_yaw import yaw_score
    validate_terrain_rows(rows)
    if [row['seed'] for row in rows]!=list(expected_seeds):raise ValueError('Missing, duplicated or reordered scenarios')
    scores=[]
    for row in rows:
        score=yaw_score(row)  # Reuse complete-trajectory validation; CPU metric stays untouched.
        if score is not None:
            scenario=row['scenario'];speed=abs(scenario['speed'])
            old_ref=3.5+1.5*(scenario['center']+abs(scenario['offset'])/2+.75)/speed
            new_ref=3.5+1.5*row['task_goal_progress_m']/speed
            score*=np.sqrt(old_ref/new_ref)
        scores.append(score)
    complete=all(value is not None for value in scores)
    return dict(total=len(rows),success_count=sum(bool(row['success']) for row in rows),complete=complete,
        mean_yaw_score_deg=float(np.mean(scores)) if complete else None)


def evaluate_terrain(model,normalization,cases,env_kwargs=None):
    if (env_kwargs or {}).get('terminate_on_attitude_failure',False):raise ValueError('评估必须保留完整物理轨迹，不能启用训练姿态终止')
    raw=TerrainEnv(len(cases),scenario=cases,**(env_kwargs or {}))
    env=VecNormalize.load(str(normalization),raw);env.training=False;env.norm_reward=False
    stats=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count);rows=[]
    try:
        obs=env.reset();pending=set(range(len(cases)));found={}
        while pending:
            obs,_,done,infos=env.step(model.predict(obs,deterministic=True)[0])
            for i in list(pending):
                if done[i]:
                    row={k:v for k,v in infos[i].items() if k!='terminal_observation'}
                    found[i]=dict(seed=cases[i].terrain_seed,scenario=asdict(cases[i]),**row);pending.remove(i)
        rows=[found[i] for i in range(len(cases))]
        np.testing.assert_array_equal(stats[0],env.obs_rms.mean);np.testing.assert_array_equal(stats[1],env.obs_rms.var)
        assert stats[2]==env.obs_rms.count
    finally:env.close()
    return validate_terrain_rows(rows)
