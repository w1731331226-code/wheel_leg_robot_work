# WheelLeg 原生GPU并行基线

当前只运行1024个MuJoCo Warp并行环境。已验证迁移优胜策略后启动正式分轮训练；旧CPU训练实例和CPU/Warp对照实例已退役，公共控制、物理和评估代码保留。

- [正式训练协议、准入与停止规则](NATIVE_TRAINING.md)
- [验证、收敛曲线与清理报告](results/formal_native_1024_20260921/REPORT.md)
- [全环境观测台说明](dashboard/README.md)
- [本机训练页面](http://127.0.0.1:8765/)

唯一正式目录：`results/formal_native_1024_20260921/`。`status.json` 是当前进度，`selection.json` 指向最佳策略；最多10轮、连续3轮无实质改善停止，随后只做一次64例独立终评。

训练服务：`wheelleg-native-formal-1024.service`；前端：`wheelleg-dashboard.service`；独立渲染：`wheelleg-training-render.service`。训练内存上限24GiB、渲染3GiB，进程组均禁止换页；不同时运行第二组1024世界，不再尝试2048/4096。

全部1024世界的姿态总览来自训练GPU状态，点击任意编号查看该世界的真实3D训练帧。详情400Hz记录、显示目标50FPS，实际帧率及延迟显示在页面；评估期间暂停，不生成替代运动。

基线是纯仿真地面M3任务。历史CPU/GPU物理全状态等价门仍有失败；工程可用、开发集效果、最终泛化和实机安全不是同一结论。“最佳”是此预算和停止规则下的最佳检查点，不保证全局最优。
