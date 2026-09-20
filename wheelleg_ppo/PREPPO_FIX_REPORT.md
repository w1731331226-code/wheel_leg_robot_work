# PPO前置问题修复报告

日期：2026-09-17。控制与工程回归通过；三篇优先全文查新完成2/3，残差信赖域一篇因全文访问权限未关闭。因此不能称全部研究前置问题已清零。正式训练与最终保留集测试未启动。

验收条件在 [ACCEPTANCE.md](tools/results/fixes_2026-09-17/ACCEPTANCE.md)；机器可读汇总为 [acceptance_summary.json](tools/results/fixes_2026-09-17/acceptance_summary.json)。

## 证据核对与修改范围

开始时报告仍写24/28和六状态频响失败，但工作区已有9月16日未收尾的慢模态降阶代码、严格设计准入，以及两篇PDF。此次保留这些工作，复核公式、调用路径、回归结果并补齐验算与报告，没有把已有修改记为新写代码。原始BASELINE_MANIFEST.json及其17个冻结文件哈希全部通过，旧证据未改写。

本轮初始baseline.json复现模态模型配旧航向环24/28；它不同于9月16日静态模型的历史基线。该初始目录的source_sha256是在运行结束采集、期间发生候选编辑，故不作为初始代码版本证明；旧航向函数留存在wheelleg_sim_before.py，diagnose_yaw.py又在相同最终模态模型上成对复现旧/新航向环。最终final/和training_gate/源码哈希均已对当前文件逐一验证。

## 四个偏航失败

根因：左右轮非同步越障触发较大瞬态角速度，旧航向阻尼不足；在validation_3/9，旧差矩限幅又截断PD请求。逐0.5ms日志统计限幅58/425步；validation_4/8旧限幅0步，主要表现为阻尼不足。不能把四例统一解释成电机饱和。初始诊断、候选失败和成对轨迹均保留。

修复在共享control()航向环：六状态名义阻尼0.60→2.0 Nm/(rad/s)，归一化差矩上限0.04→0.12；按冻结名义质量和轮径约为每侧1.65Nm。比例增益0.4、死区和最终电机限制保持原值。只增加阻尼到2.0为27/28，进一步增加到4.0不能修好validation_3；联合增加航向余量后28/28。新参数保留为具名校准常量，不读取随机化质量来调控制器。

| 场景 | 原报告偏航峰值 | 本轮模态模型+旧航向 | 最终峰值 | 最终速度RMSE m/s |
|---|---:|---:|---:|---:|
| validation_3 | 6.152° | 6.154° | 4.798° | 0.12973 |
| validation_4 | 5.079° | 5.071° | 3.566° | 0.09170 |
| validation_8 | 5.718° | 5.733° | 1.937° | 0.06973 |
| validation_9 | 16.012° | 16.075° | 1.800° | 0.05088 |

全部28开发场景、含完整12例困难组，按原preppo-v2-ramp阈值通过；未挑选失败子集作为总体成绩，也未放宽阈值。四例停车距离0.137～0.255m、末段速度0.00146～0.00305m/s。成对诊断中新航向请求限幅均0步。validation_3仅约0.20°余量，这只是固定开发场景通过，不是OOD或实机保证。

证据：[最终28例](tools/results/fixes_2026-09-17/final/baseline.json)、[四例成对诊断](tools/results/fixes_2026-09-17/yaw_comparison.json)、[可运行诊断脚本及逐步NPZ](tools/results/fixes_2026-09-17/diagnose_yaw.py)。

## 六状态频响

根因：静态凝聚用被省略状态的准静态响应替代其频率相关动态，0.2～5Hz内不能满足10%精度。密集网格下静态候选误差比旧稀疏5点报告更高，不能混用这两种最大误差。

复核并保留现有慢模态法：在已闭合VMC支撑环的15状态模型上保留6个慢模态，再映射回原物理六状态；不增加在线状态或观测器。rm_controller从该模型生成冻结名义设计，降阶频响/省略动态/完整闭环不过门即拒绝加载。本轮进一步修正中点检查：插值物理坐标增益后按当前构型映射，符合在线实现，不能直接平均不同构型的电机坐标增益。

| 腿长m | 静态凝聚密集最大误差 | 模态密集最大误差 | 完整闭环谱半径 |
|---|---:|---:|---:|
| 0.16 | 22.757% | 0.260% | 0.998923 |
| 0.25 | 30.308% | 0.284% | 0.998877 |
| 0.30 | 37.628% | 0.292% | 0.998862 |
| 0.38 | 32.141% | 0.346% | 0.998852 |

四点省略动态谱半径均<1；0.205/0.275/0.340m实际插值中点闭环均<1。7个工作点的正负0.2Ns脉冲共14/14通过，无非轮接触。网格为201个对数点并保留旧5频点，门槛仍10%；这不是连续频域上界、全参数稳定性或实机证明。

证据：[模态及实际插值](tools/results/fixes_2026-09-17/modal_actual_interpolation.json)、[静态反例](tools/results/fixes_2026-09-17/static_comparison.json)。

## 文献与集成回归

两篇全文的版本、页码、动作/观测、约束、预算、对照及复现缺口已写入 [全文核对](LITERATURE_FULLTEXT_REVIEW.md)，并更新文献矩阵。第三篇IEEE明确无PDF访问权限，等待合法全文路径/链接；其动作细节和与本机方法的排他性比较仍未确认。

工程回归：12组接口检查通过，含原始哈希、当前Gym/直接控制6000物理步逐值一致、未知质量隔离、子步积分/峰值、输出限幅、重置和5类动作接口。新旧控制器的24000步轨迹保留差异，不再错误要求模型修改后逐值相同；差异JSON的最大值混合位置/速度/力矩/计时字段，仅是变更检测，不是性能指标。

旧动作站立、停车、静止跳、前进跳、后退跳5/5通过。CPU、8环境、M与B2-V各4096策略步，2次更新×10epoch均通过参数更新、有限性及保存重载/归一化检查。短更新与其他检查有重叠，耗时不作吞吐结论；未再次验收CUDA或运行正式学习。

证据：[接口](tools/results/fixes_2026-09-17/interface/acceptance.json)、[旧动作](tools/results/fixes_2026-09-17/regression/10s_dynamics.json)、[CPU短更新](tools/results/fixes_2026-09-17/training_gate/training_gate.json)。

## 复验命令

在wheelleg_ppo目录执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python tools/test_preppo.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python tools/prepare_ppo.py baseline --require-success --output tools/results/recheck_baseline
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python tools/model_lqr.py --hardware --reduction 6 --heights .16 .25 .30 .38 --check-midpoints --report tools/results/recheck_model.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python tools/test_dm8009_dynamics.py --modes stand stop jump jumpfwd jumpback --output tools/results/recheck_actions
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python tools/prepare_ppo.py pulses --output tools/results/recheck_pulses
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python tools/prepare_ppo.py training_gate --output tools/results/recheck_training
```

新增--require-success使baseline只有28例全通过才返回0；不带该参数仍保持原研究评估行为。修订后的经典基线必须作为后续M/B2-V及其他对照的共同底座，不能把本轮经典控制改进算作PPO收益。旧短更新策略及旧脉冲/探索分布不能混充新模型证据；本轮已重新生成 [18组脉冲及3类状态×5种动作初始分布](tools/results/fixes_2026-09-17/pulses/pulses.json)，源文件哈希与当前控制器一致。
