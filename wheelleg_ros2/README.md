# wheelleg_ros2

毕设工程，暂缓开发；ROS2尚未接入。独立复制2026-09-15基线，无运行时跨项目源码依赖。旧项目保留历史证据。

当前基线：300 mm轮距、160 mm中央箱体、220 mm平台、7 kg名义模型；VMC＋六状态LQR＋状态机。论文项目后续改动不自动同步到毕设项目。

从本目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python3 tools/wheelleg_sim.py --test stand
../.venv/bin/python3 tools/wheelleg_sim.py
```

当前可复用父目录虚拟环境；独立环境按requirements.txt安装。尚未安装PPO/ROS2依赖。来源和代码哈希见BASELINE_MANIFEST.json。

详见 GRADUATION_PLAN.md。
