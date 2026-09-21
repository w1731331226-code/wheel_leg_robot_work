# 实时训练观测台

访问 <http://127.0.0.1:8765/>。前端、接口和渲染均在本地，无外部前端依赖，不上传视频。用户明确选择了**训练进程原始画面**，因此旧无采集对照已停止并保留，当前为 `formal_cpu_warp_fast_v1_20260921` 的重新初始化配对训练。

## 画面来源与保存

- 每侧显示8个并行环境中的环境0。训练采样进程在每个50Hz策略步后保存真实qpos/qvel、控制力矩、动作、观测、奖励；物理仍按2kHz步进。没有用另外一个策略回放环境冒充实时训练，没有预设动画。
- 独立渲染服务读取状态，使用同场景模型及MuJoCo离屏渲染。它只做渲染用的前向计算，不向训练环境施加动作、不额外推进训练物理。暂停画面按钮只暂停浏览器观看。
- 当前JPEG与原始状态的帧号/回合号先匹配，再显示；画面标明仿真时间、采样步、环境、种子、回合和数据年龄。预检阶段和正式训练明确区分。GPU画面的墙钟速度会受实际训练吞吐限制，不能假装实时速度运行。
- 每个完整回合的原始轨迹与场景配置位于 `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/<cpu或warp>_<seed>/live/episodes/<回合>/`。
- 约每2万采样步选择一个完整回合在后台生成50fps GIF/WebP，位于 `wheelleg_warp/dashboard/local_data/captures/`。前端可下载GIF、NPZ轨迹、JSON配置。GIF不覆盖旧录像，原始完整回合轨迹全保留。
- 大体积原始轨迹和录像在本地保存并被Git忽略；代码、协议、准入、状态和科研检查点继续由原Git守护同步。视频/逐帧轨迹没有远程备份。训练被强制终止时，最后尚未结束回合只有最新状态保证已落盘，不能称所有未完成帧均已归档。

离线重放任意完整回合（无需再运行策略或GPU物理；模型源码指纹须匹配）：

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  wheelleg_warp/dashboard/record.py \
  --replay /完整回合或录像目录 \
  --output /输出目录/reproduced.gif
```

这能复现已记录的环境0运动状态与画面，不等于恢复全部8环境的训练随机流/优化器瞬时状态；正式策略和归一化检查点仍按2万步保存。渲染的抗锯齿/颜色量化可能有极小像素差，不承诺跨GPU逐像素一致。

## 状态与预计完成时间

`/api/status` 每10秒刷新真实progress/selection/completed/failed文件，前端每5秒读取。正式进度按PPO更新边界记录，画面采样步可以略领先。已完成检查点评估周期存在后，ETA优先使用包含评估开销的周期吞吐；早期只根据已完成策略更新估计，并显示约0.5～2倍粗区间。总体时间还预留共同终评开销。区间是工程外推，不是统计置信区间，不是完成时间保证。

队列失败、退出或15分钟无新progress时暂停ETA；后者也可能是长评估，需要结合日志判断。CPU/GPU共享机器、课程阶段、渲染与评估竞争都会改变速度。最终比较完成之前，不从动画或短曲线宣布方法优势。

## 服务

```bash
systemctl --user status wheelleg-cpu-warp-formal.service
systemctl --user status wheelleg-dashboard.service wheelleg-training-render.service
journalctl --user -u wheelleg-training-render.service -n 30 --no-pager
```

两项前端服务已配置用户服务自启及失败重启；监听仅127.0.0.1。停止前端不会停止正式训练：

```bash
systemctl --user stop wheelleg-dashboard.service wheelleg-training-render.service
```

`test_live.py`验证记录不改变状态/奖励/RNG，`test_server.py`验证ETA边界和本地文件访问限制；采集版4000步CPU策略最终权重与无采集版逐值一致，两后端采集版4000步均通过真实PPO检查。浏览器已验证实时图像、暂停/恢复及下载链接。离线录像重建167帧，首/末采样图像一致，中间采样图像平均绝对像素差约0.001/255，原始运动状态来自同一NPZ。

2026-09-21追加优化：直播传输限频由5Hz提高到约60Hz，编码改为独立进程；完整原始50Hz状态用于录像，无插帧。页面“50 FPS录像”优先用WebP播放，并明确与原始实时采样区分。原25fpsGIF仍保留。GPU回读及续训说明见 [优化记录](../OPTIMIZATION.md)。当前ETA从恢复步数扣除旧进度后计算，不能把检查点继承步数当成新速度。

续训ETA还剔除恢复点一次性补评估的启动耗时，保留未来正常周期评估影响；检查点继承步数不计新段吞吐。界面吞吐是8环境合计，不能直接当作单车直播FPS。
