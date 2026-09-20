# MuJoCo Warp GPU 基线

这是独立物理后端基线。原 `wheelleg_ppo/` CPU 仿真仍为效果参照，原训练继续运行。采用官方 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)；`mujoco-wrap` 按 `mujoco-warp` 实现。

## 运行

从项目根目录运行：

```bash
.venv/bin/pip install -r wheelleg_warp/requirements.txt
.venv/bin/python wheelleg_warp/baseline.py --output wheelleg_warp/results/新实验目录
```

默认 CUDA:0，256个并行世界，200步物理回放，并运行站立/非对称地面/跳跃各6秒成对校对。输出目录必须不存在，防止覆盖历史。无CUDA、容量溢出、非有限状态或校对超阈值会失败。短检查：

```bash
.venv/bin/python wheelleg_warp/baseline.py --seconds .1 --cases stand --output wheelleg_warp/results/新冒烟目录
```

模型和控制器直接复用原CPU代码，维持0.5ms物理步、六状态覆盖→残差→最终限幅→物理步的顺序。配对闭环使用两个独立控制器状态，零残差；GPU物理通过CUDA graph运行并回传原控制器需要的状态。非对称开发场景1秒后加速，4.5秒后停车；跳跃2秒触发。不使用最终保留集。

`CPU_REFERENCE.json` 固定原CPU源码/模型的27个哈希及 `cpu-reference-pre-warp-20260921` 标签；运行前检查，变化时拒绝混用原基线。

`summary.json` 保存原CPU Git基点、源码哈希、版本、场景、误差、接触集合和耗时。qpos全向量混合位置/角度/四元数，仅为工程数值比较，不能当成米制位置精度。工程误差阈值为 qpos≤0.02、qvel≤0.5、ctrl≤2.0 且接触障碍集合相同；原论文成功门不变。`aligned` 只表示此校对，不表示任务成功率或PPO优势。

## 性能口径

批量测试将原CPU生成的控制轨迹同时回放到GPU的256个世界，与8线程原生CPU rollout比较，预热/编译不计入耗时，初始状态保持一致。CPU计时包含rollout状态输出，GPU计时只含图启动和完成同步、末态回读在计时外，因此是各自API物理执行路径的测量，并非完全相同数据输出工作量。结果是短窗口单次测量，且原CPU训练仍并行运行；不作为稳定的端到端提速承诺。

闭环校对保留原Python控制器，每步完整回读，通常比CPU慢。GPU原生批量控制/奖励/终止和PPO采样尚未实现。不得把物理回放倍率写成训练倍率；迁移训练前需完成该链路及新的配对准入。

本轮原始结果位于 `results/smoke_20260921/`（初次失败）、`results/smoke_fix_20260921/`（修正通过）、`results/paired_20260921/`（首次完整校对发现接触回读兼容问题）、`results/paired_contact_fix_20260921/`（修正后完整校对）。最终状态以各目录JSON为准。

## GitHub 守护与长期记忆

根目录 `PROJECT_MEMORY.md` 保存跨对话状态；`AGENTS.md` 为执行约束。服务模板为 `tools/wheelleg-git-sync.service`，安装到 `~/.config/systemd/user/`，执行：

```bash
git config core.hooksPath .githooks
cp tools/wheelleg-git-sync.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now wheelleg-git-sync.service
systemctl --user status wheelleg-git-sync.service
journalctl --user -u wheelleg-git-sync.service -n 30 --no-pager
```

服务路径针对本机；新路径部署需改服务模板。30秒扫描，文件连续两轮稳定才生成中文快照提交；每次显式提交也会推送到当前上游。只同步Git非忽略内容，超95MiB文件拒绝提交。断网/认证/冲突留在journal，保留本地提交并重试，不强推、不自动合并。用户已有暂存时只推送已有提交，不自动打包暂存区。每轮编辑先停服务，写记忆、中文提交后重启。

用户会话退出后持续运行需要启用 `loginctl enable-linger wmt`；系统关机/休眠时不会同步。重启后服务会继续归档稳定的最终文件状态，无法重建离线期间每一次瞬时编辑。停止自动同步用 `systemctl --user disable --now wheelleg-git-sync.service`。

针对接触回读兼容的独立回归：`.venv/bin/python wheelleg_warp/test_contact_bridge.py`。该测试主动破坏旧接触字段，验证GPU回读会恢复原CPU控制器需要的geom1/geom2。

## 当前实测状态（2026-09-21）

GPU候选基线已可运行，尚未通过全部CPU对齐门，不能替换原CPU。

| 6秒闭环场景 | 最大qpos误差 | 最大qvel误差 | 最大ctrl误差 | 校对 |
|---|---:|---:|---:|---|
| 站立 | 2.26e-6 | 5.35e-5 | 1.04e-4 | 通过 |
| 非对称接触 | 0.17224 | 11.33987 | 1.31268 | 未通过 |
| 跳跃 | 0.001499 | 0.01183 | 0.00358 | 通过 |

非对称场景双方均接触bump_L，末态车体位置差约0.28mm，但全状态误差仍超限，不能用末态指标替换预注册门。总状态为 `alignment_failed`，入口退出1是对齐门的预期拒绝。

256世界×200步开环物理短测GPU耗时0.07270秒、8线程CPU耗时0.18369秒，倍率2.5266；按前述计时范围和后台负载限制解释。闭环回读路径更慢（GPU约47～50秒，CPU约5.8秒，均运行6秒仿真），不宣称PPO加速。

后续用户已授权双后端正式训练，方案和入口见 [正式训练比较说明](FORMAL_TRAINING.md)。此前对齐失败的限制仍保留，训练对照不等于宣布两个物理后端已等价。
