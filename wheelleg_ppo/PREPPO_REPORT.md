# PPO训练启动准入报告

最新修复状态见 [2026-09-17修复报告](PREPPO_FIX_REPORT.md)：开发组28/28、六状态频响门通过，两篇全文已核对；第三篇仍待全文。以下保留2026-09-16历史准入记录，其24/28及旧频响失败不代表当前版本。

日期：2026-09-16。结论：**CPU与CUDA小规模训练工程准入均通过；当前默认采用8环境CPU。正式训练尚未启动。**（驱动重装后的更新见第7节。）

本次只完成训练前的软件、环境、协议与数值检查。短更新生成的模型不是可用控制策略；四个困难开发场景的失败保留，不构成程序验收失败，也不代表PPO已经解决这些问题。

## 1. 已完成项与证据

| 检查 | 结果 | 证据 |
|---|---|---|
| 原始基线冻结 | 原17个文件逐一核对原SHA256；未改写原清单 | `BASELINE_MANIFEST.json`；结果目录 `baseline/` |
| 零残差控制回归 | 前进/后退与停车24,000个物理步逐值一致 | `acceptance.json`、`trace_old/new_*.npy` |
| Gym包装层回归 | 名义模型惯量及6,000个物理步逐值一致 | `acceptance.json` |
| 未知质量隔离 | 同状态命令一致；增益、前馈、平衡角、mass_scale只读冻结 | `test_preppo.py` |
| 最终输出链 | 六状态覆盖之后加残差；每0.5 ms重算映射/余量；最终计时一次 | `wheelleg_sim.py`、`ppo_env.py` |
| 动作接口 | M1/M2/M3、B2-V和原始六维力矩残差均通过接口检查 | `acceptance.json` |
| 环境契约 | Gym/SB3检查、重置复现、状态隔离、终止/截断、无效动作拒绝、40子步积分/峰值检查通过 | `acceptance.json`，共11组检查 |
| 摩擦和延迟 | 实际分侧接触摩擦核对；刚度/阻尼保持名义混合；Actor延迟缓冲核对 | `acceptance.json` |
| 旧动作回归 | 站立、停车、静止跳、前进跳、后退跳5/5通过 | `regression/10s_dynamics.json` |
| 开发场景B0 | 24/28；包括名义4/4、单侧10/15 mm凸台8/8 | `baseline.json` |
| 机理脉冲 | 平地、不同腿高、单侧接触；三通道正负小脉冲18组完成 | `pulses.json`、`pulse_*.npz` |
| 初始探索分布 | 各动作模式在同三类状态各256样本；已记录实际力矩RMS与限幅比例 | `initial_distribution_*.json` |
| PPO链路 | M/B2-V全随机化、8环境、两轮更新；参数/损失有限、保存重载一致 | `training_gate.json` |
| 文献矩阵 | 12篇已建立；全文未取得项逐项标注 | [LITERATURE_MATRIX.md](LITERATURE_MATRIX.md) |

结果统一位于 [tools/results/preppo_2026-09-16](tools/results/preppo_2026-09-16)。正式测试集没有运行。

## 2. 协议校核及失败结果

最初候选协议为阶跃起步及到终点后1 s停车观察，28个场景均未通过；无障碍±1 m/s的速度RMSE约0.235～0.236 m/s，末段最高速度约0.102～0.104 m/s。这暴露的是参考指令/观察窗口问题，不是新控制器性能退化。原结果保留在 `candidate_v1_baseline.json`，不会拿它与修订协议直接比较算法提升。

冻结协议 `preppo-v2-ramp`：准备1 s，1 s线性速度斜坡，之后巡航；斜坡全部计入速度误差；到终点后停车观察2 s，统计末0.5 s。保持5°姿态/航向、20%巡航速度RMSE、0.60 m停车距离、0.03 m/s末段速度四个门槛。控制增益未调整。只用开发场景校核，所有RL及经典对照必须使用相同协议。

最终保留的失败均为偏航超5°，没有倾倒或非法状态：

| 开发场景 | yaw峰值 | 速度RMSE m/s |
|---|---|---|
| validation_3 | 6.152° | 0.1307 |
| validation_4 | 5.079° | 0.0923 |
| validation_8 | 5.718° | 0.0704 |
| validation_9 | 16.012° | 0.0549 |

困难验证组固定为 `validation_0..11` 的全部12个场景，不能只挑上表失败场景；当前8/12通过。它是开发组，不替代正式200 episode测试。组合留出区域为障碍高度≥16 mm且左右摩擦差≥0.25，训练/验证排除该区域；独立测试命名空间及种子列表已记录在 `protocol.json`。

左右驱动参数定义为轮电机力矩增益 `1+δ` 与 `1-δ`，训练δ∈[-0.03,0.03]；这是增益失配，两侧增益之差为2δ。日志分别记录命令和实际执行器力矩，不把名义命令约束称为未知电机的实测安全边界。

初始分布检查只证明没有明显的初始饱和偏置，不证明各模式探索分布相同；B2-V和M可达集合不同。脉冲结果显示平地轮差矩对yaw的影响明显，构型/接触改变后有交叉耦合；是否删除ΔF/ΔT仍须递进训练消融，不能凭脉冲作性能结论。

实现QA中还修复了两个容易隐藏的混杂因素：摩擦优先级不能顺带改变solref/solimp；7.0与浮点求和6.999999999999999不能触发名义载荷修改。前者的旧结果保留为 `superseded_contact_mixing_*`，后者已有严格逐值回归。

## 3. 主机与依赖

- CPU：Intel Core i7-14650HX，24逻辑线程；本进程可用24线程。
- 内存：约31 GiB，总可用约26 GiB；工作盘剩余约158 GB。足够当前8进程、小型MLP和检查点。
- 系统：Ubuntu 22.04系列内核6.8.0-138，Python 3.10.12。
- MuJoCo 3.12.0，NumPy 2.2.6，SciPy 1.15.3，Gymnasium 1.3.0，SB3 2.9.0；初次验收使用PyTorch 2.14.0+cpu，驱动重装后已更新为2.14.0+cu130。
- `pip check`无依赖冲突，CPU自动微分通过；初次版本锁在 `preppo_2026-09-16/environment.lock.txt`，当前版本锁在 `cuda_preflight_2026-09-16/environment.lock.txt`（均位于 `tools/results/`）。
- GPU：用户重装驱动后，主机已识别NVIDIA GeForce RTX 5060 Laptop GPU，约8 GB显存，驱动580.178.04；PyTorch CUDA 13.0运算与PPO更新通过。旧 `gpu_host_check.json` 为修复前记录，不代表当前状态。Codex沙箱仍不可访问显卡，CUDA检查通过授权的沙箱外运行；见第7节。

运行设置固定 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，PyTorch线程数1，避免8个环境各自抢占全部CPU。当前无需迁移MJX或等待GPU修复。

## 4. 实测吞吐与短更新

采样基准使用2×64 MLP、每环境256策略步、两轮优化epoch；包含观测、奖励、接触和PPO更新，进程启动不计入持续吞吐。表中200万步耗时只是短基准外推，正式验证、检查点、较难地形和长时降频会增加时间。

| CPU环境数 | 总策略步/秒 | 本轮优化耗时 s | 200万步粗估 h |
|---|---|---|---|
| 1 | 34.7 | 0.010 | 16.00 |
| 4 | 129.0 | 0.034 | 4.31 |
| 8 | 212.1 | 0.067 | 2.62 |

首次reset（含名义设计）0.284 s；设计缓存后的reset中位数0.025 s。完整名义单环境零动作采样约44.7策略步/s，不能与多进程总吞吐混用。

更接近拟训练配置的检查使用8环境、完整随机化、每环境256步、batch=256、每次10个epoch：

| 方法 | 策略步 | 更新次数×epoch | 总耗时 s | 数值/重载 |
|---|---|---|---|---|
| M（diff3） | 4096 | 2 × 10 | 20.68 | 通过 |
| B2-V（virtual6） | 4096 | 2 × 10 | 21.45 | 通过 |

检查过策略参数确实发生更新、全部参数及训练标量有限、确定性预测重载一致；观测归一化保存/重载以及测试冻结也通过。`smoke_*`、`gate_*`模型仅为链路验收产物。

建议从CPU、8环境、2×64网络开始，先按共同课程做M/B2-V各20万步初始学习检查，再评估200万步训练。不要用这些短更新损失或未训练策略的成功率论证方法有效。

## 5. 仍未关闭的研究项

1. 动作空间设计、残差信赖域和中文TD3-PID-VMC三篇的全文细节尚未取得；矩阵明确保留未核实项，创新结论待确认。
2. 六状态降阶频响误差仍约19%～32%，没有通过旧10%近似精度门。完整线性闭环谱半径<1与本次回归支持名义使用，不等价于全局稳定证明。
3. B1调度整定、正式学习曲线、多训练种子、消融、最终保留集/OOD和论文写作属于后续工作。模型接触体不完整及理想电源的限制不变。
4. CPU与CUDA均已通过工程准入；当前小网络未观察到GPU加速优势。这不是实机部署、发表创新性或学习收敛验收。

## 6. 复验

在 `wheelleg_ppo/` 下运行，完整命令亦见 [README.md](README.md)：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/test_preppo.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py baseline
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py pulses
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py benchmark
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py training_gate
```

耗时测量单独执行；回归失败会返回非零退出码。B0困难场景失败是研究记录，`baseline`成功退出只表示完成评估，具体任务通过率必须读取JSON，不能用退出码冒充28/28通过。


## 7. 驱动重装后的CUDA复验

用户重装驱动后完成，证据保存在 [cuda_preflight_2026-09-16](tools/results/cuda_preflight_2026-09-16)。主机nvidia-smi成功，驱动580.178.04；PyTorch 2.14.0+cu130，CUDA运行时13.0，RTX 5060 Laptop的计算能力为12.0。安装只改项目虚拟环境，未再次修改系统驱动。

- 11组程序验收复跑通过，依赖检查无冲突；控制、模型、场景生成器和冻结协议均未改变。
- CUDA基本运算/自动微分通过；M与B2-V均实际在 `cuda:0` 上执行两轮PPO更新，未回退CPU。
- 参数和损失有限；同设备确定性重载逐值一致；CUDA检查点在CPU上预测也在 `rtol=1e-5, atol=1e-6` 内一致；归一化冻结检查通过。
- 隐藏GPU的失败路径返回非零退出码，明确报“CUDA不可用”，不会误报GPU验收成功。
- 本轮为设备链路检查，未启动20万/200万步学习，也未运行最终测试集。

以下同配置短测量使用8环境、2×64 MLP、完整参数随机化，每种方法4096步、2次更新、每次10 epoch。CPU与最终GPU复测串行运行；CUDA计时前后同步。首轮与接口检查并行的CUDA数据保留在 `cuda/`，最终比较使用 `cuda_solo/`。

| 方法 | CPU耗时 s | CUDA耗时 s | CPU策略步/s | CUDA策略步/s |
|---|---|---|---|---|
| diff3 | 24.33 | 25.26 | 168.3 | 162.2 |
| virtual6 | 23.75 | 25.15 | 172.5 | 162.9 |

CPU与CUDA耗时接近，当前单次短测量没有显示GPU收益，不能据此宣称稳定的百分比速度差。默认继续CPU、8环境；CUDA保持可用，后续网络或任务计算量变化时再测。MuJoCo物理仿真仍在CPU，GPU只负责策略网络；[SB3官方说明](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)也建议非CNN PPO优先评估CPU。

GPU复验命令（在可访问显卡的主机终端执行）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py training_gate --device cuda --output tools/results/cuda_check
```

本轮未解决第5节的三篇全文查新缺口；工程可开始初始学习检查，研究创新性仍未确认。
