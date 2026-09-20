# CPU / Warp GPU 正式训练与最终比较

本轮用户明确要求同时正式训练两个基线并比较效果。当前带真实采集的实验根目录为 `results/formal_cpu_warp_live_v1_20260921/`，冻结协议在其中的 `protocol.json`。此前GPU全状态对齐失败保留；本实验比较两个物理后端训练所得策略，不声称物理后端等价，不替换历史CPU代码及实验。

## 已冻结的公平性条件

- 两个后端均为M3/diff3，种子1609、1610、1611，每种子200万策略步；两侧各一个8环境队列同时运行，每队列内部种子串行，共1200万新策略步。
- 新两组从相同种子重新初始化，不加载短更新权重，不将已有CPU分段续训计入配对。
- 沿用v2的网络、PPO超参数、课程、观测归一化、奖励、动作映射、VMC/六状态控制、物理步长和最终限幅。每2万步保存并在同一CPU选择集评估，按原成功数/Jψ/较早步数规则选模。
- CPU组的物理步由原MuJoCo执行；GPU组由MuJoCo Warp在CUDA:0执行。两组小型PPO网络均使用CPU以隔离物理后端变量。GPU控制器/奖励仍在CPU，逐步回读；不能称为GPU原生批量闭环，也不保证提速。
- GPU环境在独立模块命名空间复用原环境，只替换物理调用；原CPU源码不改。GPU对外episode时间统一使用整数物理步数乘0.5ms，避免float32时间累加污染严格时长核对，积分器/控制顺序不变。
- 终评为提前生成的64例 `test_iid` 新场景，种子91000～91063，仅全部训练完成后使用。原研究的门控/最终集不在本实验使用。
- 两侧选定检查点和末次检查点均在共同CPU终评环境评估。主比较是成功数和Jψ，报告3个配对种子的单独值与均值/标准差；原始输出同时记录速度RMSE、停车、姿态、饱和及失败原因。任何未完成轨迹导致Jψ总体不完整，不用剔除失败后的平均值掩盖它。
- 记录训练、周期选择评估和全程耗时；两组并行、历史CPU训练仍运行，存在资源竞争。耗时只是本次并行安排下的实测，不能推成独占硬件倍率。

## 运行与产物

初始化协议与短更新入口（新目录只创建一次）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python wheelleg_warp/train_compare.py init --output 新实验目录
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python wheelleg_warp/train_compare.py smoke --backend cpu --output 实验目录
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python wheelleg_warp/train_compare.py smoke --backend warp --output 实验目录
```

正式监督入口必须两侧smoke已通过，随后启动CPU/Warp双队列，三种子全部完成后自动共同终评并生成 `COMPARISON.md`、`comparison.json`。各队列日志为 `cpu.log`、`warp.log`，各run的 `progress.json` 在每次2000步rollout边界更新，`completed.json` 才代表完整预算完成。

```bash
systemctl --user status wheelleg-cpu-warp-formal.service
journalctl --user -u wheelleg-cpu-warp-formal.service -n 30 --no-pager
```

服务退出不会自动从零重跑，失败/中断产物保留。中断后要显式审核恢复方案并披露随机流/环境状态是否恢复，禁止把分段恢复冒称无中断训练。某一队列失败时，另一队列可继续完成自身工作，但不能据此生成完整比较。当前监督器等待两队列退出后写最终失败状态；日常排错同时查看各队列日志与run中的 `failed.json`，不能只看监督器active。

源文件和依赖哈希在每次运行前后核验；正式队列期间不要改冻结入口，必要修复应停队列、保留证据、重新注册。Git同步守护继续归档稳定文件并推送；最终比较尚未生成时，不把短更新或在跑进度当成最终效果。

## 验证入口

- `test_training.py`：GPU非零动作、观测/奖励、时间、结束重置与原CPU模块未受污染。
- `test_comparison.py`：仅合成结果，验证配对汇总、选定/末次双检查点和不完整预算拒绝，不访问真实终评集。
- 两侧4000步smoke：真实PPO参数更新、损失有限、保存重载、CPU共同评估、stage2/3重置与非零输入。正式运行重新随机初始化。

## 真实训练画面版本

用户随后明确选择训练进程原始逐帧画面。旧 `formal_cpu_warp_v1_20260921` 已停止、标为superseded并保留全部产物，不与当前成绩混合。新版本两侧从相同种子重新开始，仅环境0增加只读状态采集，采集版两侧4000步检查通过；CPU最终策略权重与无采集版逐值一致。新正式服务于2026-09-21 07:39:50启动，仍按双方3种子×200万步预算自动终评。前端与录像说明见 [训练观测台](dashboard/README.md)。
