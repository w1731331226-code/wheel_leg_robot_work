# 长期项目记忆

维护规则：每轮大的对话结束前更新当前状态、决策理由、变更及验收证据、未关闭项；每次提交必须包含本文件的更新。此文件是跨对话入口，原始实验数据与历史计划继续保留在各自目录。自动快照条目只证明归档，不证明代码或研究结论通过。

## 当前约定（2026-09-21）

- 用户要求：新增 MuJoCo Warp GPU 基线；原 CPU 仿真继续作为原始效果校对；运行 GitHub 同步守护；为每轮实质工作更新长期记忆并及时写中文提交说明。
- 项目约束入口：`AGENTS.md`、`.agents/skills/wheelleg-project-guard/SKILL.md`。GPU 基线独立放在 `wheelleg_warp/`；CPU 控制、物理模型、环境和历史证据不覆盖。
- Git 远程：`origin` → `https://github.com/w1731331226-code/wheel_leg_robot_work.git`，工作分支 `main`。禁止 force push；远程分叉时保留本地提交并人工处理。
- 同步覆盖 Git 已跟踪文件及未忽略的新文件，连续两次30秒扫描保持稳定才自动快照；缓存、凭据及 `.gitignore` 排除内容不上传。超过95MiB的单文件拒绝提交。后台训练产生的稳定检查点属于用户授权同步范围，记录为独立自动归档提交。
- 编辑任务开始暂停 `wheelleg-git-sync.service`，结束更新此记忆、提交后恢复；可用 `.git/project-write.lock` 排他锁代替。用户暂存内容优先，不自动提交已有暂存区。
- `.githooks/commit-msg` 在本地强制检查中文标题、随提交更新本记忆、凭据路径和文件大小。新克隆须执行 `git config core.hooksPath .githooks`；本地钩子不是 GitHub 服务端规则，不允许代理主动绕过。

## 当前任务：暂停训练与多环境并行实测（2026-09-21）

- 用户要求先停止旧基线及当前CPU/GPU训练，再全力测试多环境并行并给出最优配置。两训练服务及其全部子进程已停止；渲染服务为独占测量暂时停止，训练不自动恢复。
- 用户补充：必须验证收敛效果，随后明确最多使用1024批量，更多会卡死。2048测试无完整结果，09:xx～10:02之间发生重启；4096未运行，测试队列已不存在。不得重启大于1024的实验，测试进程需限制内存。
- 新实测目录 `wheelleg_warp/results/parallel_sweep_20260921/`；停止记录含原正式status，历史日志/检查点保留。
- 正在比较CPU多进程、现有Warp多进程及工作区已有原生GPU批量候选。保持250步轨迹、原PPO超参，区分采样与实际梯度更新；原生候选未获正式等价准入。

### 1024候选工程检查与收敛协议

- CPU扫描1/4/8/12/16/20/24/32，峰值20环境229.7策略步/秒；原Warp 1/4/8/12峰值8环境82.6。原生32/64/128/256/512/1024中1024达到11657.0策略步/秒（含PPO更新），不含评估/录像，固定场景库；只作为性能候选。
- 原生回合累计日志多扣10分，独立8回合测试复现并修复，返回给PPO的实际奖励未变；误差由10降至4.6e-14以内。正/反向、质量/延迟两场景同GPU后端控制奖励重置配对通过，未改变CPU物理等价门。
- 已在NativeEnv和扫描入口限制最多1024环境。2048中断、重启观察及4096未运行记入 `large_batch_interrupted.json`；原因未确认，不伪称通过或OOM已证实。
- 收敛实验目录 `wheelleg_warp/results/convergence_1024_20260921/`，服务 `wheelleg-native-convergence-1024.service`，24GiB内存上限且MemorySwapMax=0。三个配对种子，1024世界下n_steps=16/250、相同其余PPO配置，各2,048,000步，125/8次PPO更新，末次共同CPU开发32例。中间评估报告实际完成更新步数，不用最终保留集。
- 仅运行这个新收敛实验，不恢复旧正式训练；原生候选固定stage3训练场景库，原课程/每回合重采样仍未接入。学习曲线与效果结论尚待实验完成。

## 最新任务：GPU闭环与动画优化（2026-09-21）

- 用户指出GIF不流畅、Warp GPU较慢并要求尽可能优化。已分开处理：旧GIF25fps、直播5Hz限频/同步编码卡顿；GPU每0.5ms完整回读大量不需要字段。
- 新 `fast_physics.py` 打包已审计的控制/奖励必需字段，复用固定页内存，CUDA图中合并上传、原物理步、打包与回传，只同步当前流。原CPU和完整GPU回读类保持不变。
- 同一GPU状态回读逐值一致；非对称越障12000物理步实际接触与0/2/3接触数量变化检查通过。单环境800物理步三次中位数5.271→1.242秒，观测4.24倍。4000策略步GPU采样/更新167.882→58.331秒，观测2.88倍；CPU同预算34.890秒，GPU仍更慢，不扩大为已实现端到端GPU优势。
- CPU4000步最终权重逐值一致；GPU非逐位一致，最大参数差0.01030；已记录接触附近首次qpos数值差5.12e-9，不宣称跨运行训练轨迹逐位一致。全部证据见 `wheelleg_warp/results/packed_transfer_20260921/`。
- 动画已用完整50Hz状态生成50FPS GIF/WebP，旧25fps文件保留；编码在独立子进程，直播循环及发送上限约60Hz，实际FPS仍受单环境样本供应限制。前端明确区分真实实时与50FPS录像，无伪造插帧。WebP样例约为GIF的28%。
- 当前正式目录 `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/`；CPU从60000、GPU从20000更新检查点分段恢复策略/优化器/归一化，不用探针模型，不从零重练。两侧4000步smoke和各2000步恢复探针均通过。
- 环境及随机流重置，恢复步数不同，最终比较必须披露分段运行；当前段耗时不是无中断全程耗时。总策略预算仍是每种子200万、每后端3种子，原目录及历史成果全保留。
- 新正式服务于08:18:06实际启动，PID49625。前端已切到新目录；ETA扣除继承步数，等待新段实测再估计，不沿用旧4.4天结论。状态以run_config/progress/selection和服务为准。
- 详情及后续瓶颈见 `wheelleg_warp/OPTIMIZATION.md`；CPU控制/奖励及8个单世界GPU上下文仍限制吞吐，256世界开环704k物理步/秒不能冒充当前闭环/PPO吞吐。

## 真实训练画面第一版（原目录保留）

- 用户明确选择“训练进程原始逐帧画面”，不是另跑检查点回放。旧 `formal_cpu_warp_v1_20260921` 已停止并标为superseded，入口旧源码和全部产物保留于原目录；旧进度不混入新比较。
- 当前正式目录为 `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/`，新协议增加只读LiveRecorder，CPU和GPU均从相同种子重新初始化。双方4000步采集版真实PPO检查通过、初始权重一致；CPU最终4000步策略权重与无采集版逐值相同。
- 新正式监督服务于北京时间07:39:50实际启动，PID 41042；预算仍为两后端各3种子×200万步，各8环境，全部完成后自动共同CPU终评。物理后端差异与GPU当前较慢限制仍保留。
- 前端地址 `http://127.0.0.1:8765/`。`wheelleg-dashboard.service` 和 `wheelleg-training-render.service` 已启动、enabled且用户退出登录后持续运行。渲染器与训练进程独立。
- 画面来自每侧真实训练环境0，每50Hz策略步保存qpos/qvel/ctrl/action/obs/reward，物理仍2kHz。状态原子写入，渲染前匹配帧号/回合号；画面标明正式/预检、种子、环境、回合、仿真时间和数据年龄。评估期间采样暂停，显示最后真实帧。
- 所有完整回合NPZ在当前运行 `<backend_seed>/live/episodes/`；约每2万采样步选一个完整回合生成GIF，位于 `wheelleg_warp/dashboard/local_data/captures/`，前端可下载视频、轨迹和配置。大体积原始轨迹/GIF按用户本地保存要求保留在磁盘并被Git忽略，未远程备份；代码/协议/指标/模型检查点仍同步GitHub。
- 本轮验证：记录不改变状态/奖励/RNG；两侧4000步通过；CPU最终策略逐值一致；ETA边界和路径穿越/符号链接访问拒绝；浏览器真实画面、暂停/继续和下载链接；保存167帧轨迹可离线重放，首末图像一致，中间取样像素MAE约0.001/255，非跨GPU逐像素承诺。
- 07:44左右观测CPU约18000、GPU4000策略步，早期总体ETA约4.4天（粗区间2.2～8.9天）。此数值仅为当时吞吐外推，后续以前端/API实测为准；完整评估周期出现后纳入其开销，停机/失败/过期暂停ETA。
- 说明与重放命令见 `wheelleg_warp/dashboard/README.md`。回放可重现环境0已记录的运动状态，不等于保存全部8环境完整训练RNG和精确续训状态；强制中断时尚未结束回合可能只有最新状态落盘。

## CPU / GPU 对照第一版（无逐帧采集，已中止保留）

- 用户最新明确授权同时进行CPU/GPU两个基线的正式训练，最后比较效果；此前“本轮只做工程验收”的范围限制由本次授权更新。GPU对齐失败仍保留，不能重标为等价CPU基线。
- 冻结新协议 `wheelleg_warp/results/formal_cpu_warp_v1_20260921/protocol.json`：M3，CPU/Warp各3种子1609/1610/1611，每种子200万策略步，合计1200万新策略步；每侧8环境，双队列并行，队列内串行种子。
- 新两侧重新同种子初始化；历史CPU续训继续保留且不计入配对。PPO网络两侧均用CPU，以隔离物理后端变量；Warp物理使用GPU，控制与奖励仍在CPU，不承诺GPU端到端加速。
- 新增 `gpu_env.py` 独立环境适配，无需修改冻结CPU源码；修正GPU对外episode时钟的float32累加偏差，使用整数物理步时钟。真实非零动作/重置/结束与CPU模块隔离检查通过。
- 两侧4000步短更新均已通过：CPU采样/更新35.31秒、全程59.49秒；GPU采样/更新186.56秒、全程215.49秒。初始策略权重SHA256完全一致。短更新证据仅作管线准入，正式权重从同种子重新初始化。
- 两侧使用同一32例CPU选择集；全部训练完成后自动在提前固定的64例新IID场景比较selected和last检查点，输出 `COMPARISON.md` 与 `comparison.json`。原研究门控/最终集不用于本次比较。
- 合成汇总、真实GPU环境适配、两侧PPO短更新均通过，准入记录为实验目录 `readiness.json`。用户服务 `wheelleg-cpu-warp-formal.service` 已于北京时间2026-09-21 07:16:55实际启动（监督PID 35212），自动启动CPU/Warp队列并在双方全部完成后终评。实时状态见 `status.json` 和各run的 `progress.json`；不能把active或中途进度称为完整预算完成。

## CPU / PPO 已知状态

- GPU 分支的原始 CPU 源码基点为 `0f70b834bd8071f485bc81b1d6b1f99cf266d5de`；本轮开始时未提交变更均为在跑训练的日志/检查点。本轮未将这些训练结果归为 GPU 实验。
- `wheelleg_ppo/PAPER_PLAN.md` 第12节为 v2 研究协议；28例开发场景基线已有通过记录，困难组12/12，不能仍用不可达的“再提高10个百分点”主门。研究门沿已有v2协议，不在本轮重调。
- CPU 实验服务 `wheelleg-m3-1609-v2-resume1.service` 本轮开始时仍运行。输出目录 `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/`；当前进度以其中完整指标JSON和 selection/completed 为准，不用本文件中的静态数字替代实时状态。
- 原15分钟自动跟进任务已取消，不代表训练服务已停止。没有自动调度后续18项队列的承诺。
- 之前只把 PPO 网络置于 CUDA 的比较没有提速，见计划12.10。新 Warp 基线改变物理执行后端，不能直接继承旧实验训练成绩或协议源码哈希。

## 本轮实施与验证

- 新增 `wheelleg_warp/baseline.py`、固定版本依赖及运行文档；复用 CPU 模型构建、VMC＋六状态LQR＋零残差和最终电机限幅。GPU 使用 `cuda:0`、MuJoCo Warp 3.12.0、Warp 1.17.0、MuJoCo 3.12.0。只增加两个GPU依赖，原CPU依赖版本保持不变。
- CPU/GPU 分别运行独立控制器状态，测试站立、已见参数范围的非对称接触、跳跃；CPU模型原参数不改。固定工程校对阈值 qpos 0.02、qvel 0.5、ctrl 2.0，接触障碍集合一致。阈值不是论文成功门，也不是逐位一致承诺。
- 批量吞吐为相同初态、同一CPU控制轨迹的开环物理回放，CUDA graph 执行；CPU对照采用8线程原生 `mujoco.rollout`，不以Python循环人为压低CPU速度。
- 冒烟首轮因 Python tuple 使用 abs 报错；修正为 NumPy abs，失败记录保留于 `wheelleg_warp/results/smoke_20260921/summary.json`。
- 修正后冒烟通过，见 `wheelleg_warp/results/smoke_fix_20260921/summary.json`：0.1秒轨迹 qpos 最大误差2.51e-7；256环境×200步物理回放单次GPU/CPU速度比2.496，末态qpos误差7.09e-7。与原CPU训练同时运行，非独占机器性能结论。
- 完整校对首轮站立6秒通过；非对称接触暴露MJWarp 3.12回读未同步geom1/geom2的问题，失败保留于 `wheelleg_warp/results/paired_20260921/summary.json`。已在GPU唯一回读入口同步contact.geom到旧字段，原CPU控制器不改。针对真实接触、主动污染旧字段的 `wheelleg_warp/test_contact_bridge.py` 已通过。修正后的完整6秒×3场景运行于 `wheelleg_warp/results/paired_contact_fix_20260921/summary.json`；三项均已完整运行6秒：站立与跳跃通过，非对称接触全状态校对失败，总状态alignment_failed（入口按设计退出1）。非对称场景 qpos最大误差0.17224、qvel最大误差11.33987、ctrl最大误差1.31268；双方均接触bump_L，末态车体位置差约0.28mm，但不能据此绕过已设全状态门。GPU保留为候选基线，原CPU仍为正式效果参照。
- `python3 tools/test_git_sync.py` 已通过：自动快照/推送、中文和记忆钩子拒绝、保留既有暂存内容、远程分叉不强推。服务已安装并enable，systemd配置检查通过；登录驻留 `Linger=yes` 已开启，收尾提交时启动并核对远程。

## 未关闭项与后续边界

- CPU参考标签 `cpu-reference-pre-warp-20260921` 和 `wheelleg_warp/CPU_REFERENCE.json` 已建立；新入口运行前验证27个CPU源码/模型哈希，变化时拒绝混用原基线。
- 新基线的 Python 控制器仍在CPU，校对每物理步回读完整状态；该路径预计慢于原CPU。批量物理加速不代表完整闭环或PPO训练加速；尚未迁移 Gym/PPO 采样到 GPU。
- 只有工程对齐和受控吞吐实测可在本轮验收；新后端长期训练需另行冻结协议及验证，不能直接替换在跑 CPU 训练。
- 完整校对若失败，保留误差与失败证据，不放宽门槛，不替换原CPU。
- 同步失败用 `journalctl --user -u wheelleg-git-sync.service -n 30 --no-pager` 排查；后台同步不等于会自动发送聊天通知。身份/网络/分叉问题不能通过强推解决。

### 本轮维护检查

- 依赖一致性 `pip check` 通过；新增代码编译检查、`git diff --check`、systemd服务配置检查通过。
- 原CPU参考标签已实际推送远程：`cpu-reference-pre-warp-20260921`。
- 全状态对齐失败属于已记录的基线限制，不能写成“全校对通过”；后续只在用户要求时诊断碰撞/精度差异或迁移GPU原生闭环。

### 最终GPU验收结果（2026-09-21）

- `paired_contact_fix_20260921/summary.json`：站立最大qpos误差2.26e-6；跳跃最大qpos误差0.001499、qvel误差0.01183、ctrl误差0.00358，均通过预设门。非对称接触未通过，原CPU不替换。
- 256世界×200步回放：GPU 0.07270秒，8线程CPU 0.18369秒，单次倍率2.5266，末态qpos误差7.09e-7。只代表所述物理执行路径，存在输出量及后台负载差异，未验证PPO端到端加速。
- CPU控制器逐步回读路径约47～50秒完成6秒仿真，CPU配对路径约5.8秒；GPU原生控制/环境/PPO仍未迁移。
- 所有失败JSON保留，未调整原先门槛；原CPU27项哈希最后核对仍一致。同步脚本的回归、接触字段回归、依赖与配置检查通过。

### 同步服务上线

- `wheelleg-git-sync.service` 已实际启动且active，enabled和Linger=yes已核验；本轮提交由守护进程推送，最终同步结果以远程HEAD校验为准。

### 自动同步快照 2026-09-21T06:51:23+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_320000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_320000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_320000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_340000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_340000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_340000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_360000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_360000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_360000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_380000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_380000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_380000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_400000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_400000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_400000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_420000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_420000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_420000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_440000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_440000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_440000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_460000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_460000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_460000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_480000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_480000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_480000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_500000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_500000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_500000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_520000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_520000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_520000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_540000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_540000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_540000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_560000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_560000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_560000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_580000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_580000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_580000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_600000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_600000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_600000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_620000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_620000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_620000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_640000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_640000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_640000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_660000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_660000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_660000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_680000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_680000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_680000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_700000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_700000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_700000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_720000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_720000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_720000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_740000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_740000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_740000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_760000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_760000.zip`

### 自动同步快照 2026-09-21T06:52:36+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_760000.json`

### 自动同步快照 2026-09-21T06:54:44+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_780000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_780000.zip`

### 自动同步快照 2026-09-21T06:56:55+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_780000.json`

### 自动同步快照 2026-09-21T06:58:32+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_800000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_800000.zip`

### 自动同步快照 2026-09-21T07:00:41+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_800000.json`

### 自动同步快照 2026-09-21T07:02:19+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_820000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_820000.zip`

### 双后端正式训练交付说明

- 正式监督服务单次运行，不自动重启覆盖中断目录；用户退出登录后继续运行（Linger已启用）。系统关机/崩溃后需显式审核断点恢复，不能默认为精确续训。
- 两队列完成后自动共同终评并写比较报告，Git同步守护归档推送；本轮结束时尚无最终效果结论。
- 已有CPU历史实验继续运行，资源竞争已在协议披露。GPU接触数值差异和当前逐步回读较慢均未隐瞒，不为求提速改变预算、控制、奖励或评估。

### 自动同步快照 2026-09-21T07:18:43+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_820000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_840000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_840000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_840000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_860000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_860000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_860000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_880000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_880000.zip`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/status.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp_1609/run_config.json`

### 自动同步快照 2026-09-21T07:19:51+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp_1609/progress.json`

### 自动同步快照 2026-09-21T07:47:24+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_880000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_900000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_900000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_900000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_920000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_920000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_920000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_940000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_940000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_940000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_960000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_960000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_960000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_20000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_20000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/status.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/step_20000.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/step_20000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/step_20000.zip`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp_1609/progress.json`

### 优化后真实续训核验

- 08:24左右已确认CPU从60000继续至68000、GPU从20000继续至26000，均有新PPO更新，非仅进程active。当前段早期有效采样更新速率约CPU76/GPU63策略步每秒（8环境合计），仍待完整周期评估稳定后判断全程倍率。
- 4000步CPU权重不变、GPU非逐位一致以及初次Warp内核参数编译错误均保留证据；前者不扩大为整个GPU训练精确可复现。
- 续训ETA另剔除恢复点一次性补评估，不把继承旧步数或启动补评估误算成新采样吞吐，专项测试通过。

### 自动同步快照 2026-09-21T08:27:47+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1000000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1000000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1000000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1020000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1020000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1020000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1040000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1040000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1040000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1060000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1060000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1060000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1080000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1080000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1080000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_980000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_980000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_980000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_60000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_60000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_60000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_80000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_80000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/status.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_20000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_20000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_20000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_20000.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_40000.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_40000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_40000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_60000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_60000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/step_20000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/step_20000.zip`

### 自动同步快照 2026-09-21T08:28:26+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_40000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_40000.zip`

- 录像连续刷新时同步切换图像资源与来源标签，避免新录像标签仍对应旧图像；该前端修正不改训练入口或物理。

### 自动同步快照 2026-09-21T08:31:12+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1100000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1100000.zip`

### 自动同步快照 2026-09-21T08:31:47+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_80000.json`

### 自动同步快照 2026-09-21T08:32:21+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_40000.json`

### 自动同步快照 2026-09-21T09:54:13+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1100000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1120000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1120000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1120000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1140000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1140000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1140000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1160000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1160000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1160000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1180000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1180000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1180000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1200000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1200000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1200000.zip`
- `wheelleg_warp/check_native_episode_return.py`
- `wheelleg_warp/native/benchmark.py`
- `wheelleg_warp/native/check_controller.py`
- `wheelleg_warp/native/controller.py`
- `wheelleg_warp/native/environment.py`
- `wheelleg_warp/native/models.py`
- `wheelleg_warp/native/test_native.py`
- `wheelleg_warp/profile_gpu_path.py`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_100000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_100000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_100000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_120000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_120000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_120000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_140000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_140000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_140000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_160000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_160000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_160000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_180000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_180000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_180000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_100000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_100000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_100000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_120000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_120000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_120000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_140000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_140000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_60000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_60000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_60000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_80000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_80000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_80000.zip`
- `wheelleg_warp/results/gpu_saturation_20260921/profile.json`
- `wheelleg_warp/results/native_gpu_20260921/bench_128.json`
- `wheelleg_warp/results/native_gpu_20260921/bench_32.json`
- `wheelleg_warp/results/native_gpu_20260921/engineering.json`
- `wheelleg_warp/results/native_gpu_20260921/engineering_initial_short.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_1.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_1.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_12.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_12.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_16.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_16.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_20.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_20.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_24.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_24.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_32.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_32.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_4.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_4.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_8.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_8.log`
- `wheelleg_warp/results/parallel_sweep_20260921/machine.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_1024.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_1024.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_128.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_128.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_2048.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_256.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_256.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_32.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_32.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_512.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_512.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_64.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_64.log`
- `wheelleg_warp/results/parallel_sweep_20260921/refine.log`
- `wheelleg_warp/results/parallel_sweep_20260921/sweep.log`
- `wheelleg_warp/results/parallel_sweep_20260921/sweep_events.jsonl`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_1.json`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_1.log`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_12.json`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_12.log`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_4.json`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_4.log`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_8.json`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_8.log`
