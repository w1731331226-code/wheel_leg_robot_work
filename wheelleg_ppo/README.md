# wheelleg_ppo

论文研究，优先开发。Gymnasium环境、差模/六维残差接口及训练准入检查已实现，v2预实验已于2026-09-21启动，已有M3/1609的20万步阶段核验；尚无方法优势或稳定收敛结论。独立复制2026-09-15基线，无运行时跨项目源码依赖。部署、目录结构与实现说明见[项目总文档](../README.md)，最新研究状态见[PAPER_PLAN.md](PAPER_PLAN.md)。

2026-09-17历史前置修复：开发组28/28，六状态密集频响误差<0.35%，旧动作5/5通过；当时三篇全文查新已完成两篇，残差信赖域一篇仍缺全文。详见 [修复报告](PREPPO_FIX_REPORT.md) 与 [全文核对](LITERATURE_FULLTEXT_REVIEW.md)。当时尚未启动训练；后续状态以论文计划为准。

当前基线：300 mm轮距、160 mm中央箱体、220 mm平台、7 kg名义模型；VMC＋六状态LQR＋状态机。论文项目后续改动不自动同步到毕设项目。

从本目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/wheelleg_sim.py --test stand
../.venv/bin/python3 tools/wheelleg_sim.py
```

当前使用父目录虚拟环境，已安装Gymnasium、SB3和PyTorch 2.14.0+cu130；驱动重装后RTX 5060 Laptop已通过CUDA短更新检查。CUDA构建也支持CPU训练。独立环境安装：

```bash
python3 -m pip install torch==2.14.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
python3 -m pip install -r requirements.txt
```

从本目录复验：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/test_preppo.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py baseline --require-success --output tools/results/current_baseline
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py pulses
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py benchmark
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py training_gate
```

最后两项只执行短PPO更新来验证训练链路，产生的 `smoke_*` / `gate_*` 不能用作正式策略或论文学习结果。吞吐测量须单独运行，避免其他回归占用CPU。

GPU验收使用同一脚本，显式指定设备与独立输出目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/prepare_ppo.py training_gate --device cuda --output tools/results/cuda_check
```

此命令须在能访问GPU的环境执行。CUDA不可用时脚本会报错，禁止把自动回退CPU当作GPU通过。普通CPU实验也可使用CPU构建，但冻结v2训练入口严格核对`torch==2.14.0+cu130`；替换构建不属于原配置复现。物理仿真仍在CPU运行，当前v2预实验根据本机实测使用CPU。

环境入口为 `tools/ppo_env.py` 的 `WheelLegEnv`。模式：`diff1`、`diff2`、`diff3`、`virtual6`、`torque6`、`mixed3_plus`、`mixed3_minus`；物理参数随机化只作用于被控对象，控制器默认固定名义设计。`design_controller()` 仅供明确的离线/oracle研究，不能在随机episode中调用。

源基线与哈希见 `BASELINE_MANIFEST.json`；原始17个文件已保存在 `tools/results/preppo_2026-09-16/baseline/`，供零残差对照，不是运行时依赖。不要覆盖或更新旧基线哈希来掩盖后续修改。

详见 [论文方案](PAPER_PLAN.md)、[训练准入报告](PREPPO_REPORT.md) 和 [文献矩阵](LITERATURE_MATRIX.md)。
