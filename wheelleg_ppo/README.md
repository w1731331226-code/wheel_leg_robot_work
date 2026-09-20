# wheelleg_ppo

论文研究，优先开发。Gymnasium环境、差模/六维残差接口及训练准入检查已实现，正式PPO训练尚未开始。独立复制2026-09-15基线，无运行时跨项目源码依赖。

2026-09-17前置修复：开发组28/28，六状态密集频响误差<0.35%，旧动作5/5通过；三篇全文查新已完成两篇，残差信赖域一篇仍缺全文。详见 [修复报告](PREPPO_FIX_REPORT.md) 与 [全文核对](LITERATURE_FULLTEXT_REVIEW.md)。正式训练未启动。

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

此命令须在能访问GPU的主机终端或已授权的沙箱外执行；当前Codex沙箱仍屏蔽设备。CUDA不可用时脚本会报错，禁止把自动回退CPU当作GPU通过。仅需CPU时，也可安装官方CPU构建；物理仿真仍在CPU运行。

环境入口为 `tools/ppo_env.py` 的 `WheelLegEnv`。模式：`diff1`、`diff2`、`diff3`、`virtual6`、`torque6`；物理参数随机化只作用于被控对象，控制器默认固定名义设计。`design_controller()` 仅供明确的离线/oracle研究，不能在随机episode中调用。

源基线与哈希见 `BASELINE_MANIFEST.json`；原始17个文件已保存在 `tools/results/preppo_2026-09-16/baseline/`，供零残差对照，不是运行时依赖。不要覆盖或更新旧基线哈希来掩盖后续修改。

详见 [论文方案](PAPER_PLAN.md)、[训练准入报告](PREPPO_REPORT.md) 和 [文献矩阵](LITERATURE_MATRIX.md)。
