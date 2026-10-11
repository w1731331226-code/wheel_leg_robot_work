# WheelLeg 原生GPU并行基线

## 当前论文计划（第362轮，2026-10-12）

原目标仍是：在原完整不对称接触任务上，得到一个保持强经典能力并有明确增量的可执行PPO方法。当前没有通过收益资格的新方法，正式五种子、独立ID/OOD、统计与成本分析、新方法稿件均未完成；CPU历史基线继续保留，不将物理吞吐当PPO端到端加速。

第362轮[有限滤波顺序/尺度检查](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_pair_action_candidate_v1/delivery_review.json)发现候选不能直接塞入旧virtual6接口：55296个固定几何单步算术测试中，先latent滤波再映射有19230组canonical增量超过旧.01，最大.01590；先映射再独立六路滤波则合力泄漏最大.10914N。正确顺序仍保命令代数性质，但没有执行Native/关节保护/电机包络后的实际交付或闭环。

[362决定](results/paper_recovery_20261004/joint_reference_candidate_v1/round362_delivery_decision.json)明确主要比较为CartLive3与CartReset3：同潜变量/滤波/公式，只切换当前几何与公开reset时冻结几何，864组reset映射逐值相同。原D3/V6及强经典仍保留；固定网格live/D3电机RMS相差很大，不能把坐标/滤波效应归因几何更新。两Cart臂须使用同一潜变量记忆观察版本，实际canonical与alpha只留诊断，避免额外无延迟几何信息。363须形成完整、有限的原Native链交付协议或拒绝候选；当前不准入PPO，旧canonical速率保持和探索等价均未成立。

第361轮提出了一个[完整定义的命令坐标映射](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_pair_action_candidate_v1/proposal.json)：三维潜变量指定相反的虚拟端点力与轮差矩，由当前腿几何换算回原F/H通道，再统一缩放以满足原动作盒及0.1Nm的H总和上限。[离线检查](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_pair_action_candidate_v1/review.json)覆盖512个已存姿态×27请求，合力抵消及虚功恒等通过；它只保证命令层性质，不代表真实支持力、动态解耦或任务收益。严格要求F/H/轮所有共同量为零的版本在非对称姿态仅剩二维且投影可突变，因此未采用。

[361方法决定](results/paper_recovery_20261004/joint_reference_candidate_v1/round361_method_decision.json)保留这一明确映射供362有限信息/权限/交付方案判断，尚不准入仿真或PPO。新映射改变动作方向、缩幅和探索分布，必须与原D3/V6及固定参考坐标作公平比较；须先潜变量滤波再逐控制步映射，不能直接塞进旧六路独立滤波后冒称恒等。新颖性和收益未成立，旧预测QP分支继续停止。

第360轮已完成[五轮方向审查](results/paper_recovery_20261004/joint_reference_candidate_v1/round360_direction_review.json)、[十轮框架核查](results/paper_recovery_20261004/joint_reference_candidate_v1/round360_framework_review.json)和[冗余清理](results/paper_recovery_20261004/joint_reference_candidate_v1/round360_cleanup.json)。**停止当前尚未定义预测器的联合姿态QP资格链，不新增仿真或PPO。** 第359轮的0.424542°是同输入预测误差的必要下界，未超过1°不能证明存在合格预测器，也未检验六维动作响应；再开没有明确模型的脉冲试验缺少可证伪对象。

| 分支 | 当前结论 | 主要证据 |
| --- | --- | --- |
| 固定D3差模残差与V6，三种子各200k | 训练和全164评估完成，D3收益/保持资格失败，扩展关闭 | [独立资格报告](results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_prequalification_v1/evaluation_review.json) |
| 平地轮请求、半幅与冻结反射 | 局部改善不等学习优势，反射仍有动态干扰，继续平地调优已关闭 | [13回合接收](results/paper_recovery_20261004/joint_reference_candidate_v1/frozen_wheel_interference_diagnostic_v1/review.json)、[19回合接收](results/paper_recovery_20261004/joint_reference_candidate_v1/frozen_reflection_diagnostic_v1/review.json) |
| 跨案例统一冻结反射 | 八原成功全部复现，反射八例全失败，扩展关闭 | [16回合接收与关闭](results/paper_recovery_20261004/joint_reference_candidate_v1/reflection_retention_falsification_v1/closure.json) |
| B1-route固定0/±0.05m路线参考 | 三臂各0/6成功，18例物理设计通过，六zero复现，固定偏移分支关闭 | [18回合接收](results/paper_recovery_20261004/joint_reference_candidate_v1/classical_route_reference_diagnostic_v1/review.json) |
| 联合姿态预测与分配 | 只有概念和必要条件，没有具体预测器、误差上界或可行修正证明，不准入控制器 | [方法筛选](results/paper_recovery_20261004/joint_reference_candidate_v1/round358_method_selection.json)、[40例必要条件检验](results/paper_recovery_20261004/joint_reference_candidate_v1/public_history_prediction_qualification_v1/review.json) |

原接触任务成功仍有效，但不等于居中承载完整越障或同外扰抗扰优势；路径、接触和控制共变的边界见[第354轮主张决定](results/paper_recovery_20261004/joint_reference_candidate_v1/round354_claims_and_scope.json)。八条已知学习成功在目标接触前已存在请求与横向偏移，不能无据强制其全归零；[时序证据](results/paper_recovery_20261004/joint_reference_candidate_v1/round358_gain_precontact.json)也不证明这些预动作必需。所有旧门、失败、模型、原始数据和历史文稿保留。

下一候选须先给出完整的公共输入到动作映射、模型/学习定义、同权限无学习强对照和区别性预测，再考虑预注册实验。普通QP、DOB、对称化、安全残差或PPO组合本身不是已证创新，见[文献矩阵](../wheelleg_ppo/LITERATURE_MATRIX.md)。不因当前无合格候选把目标缩为负审计论文，不重复已有失败分支或无限添加源资格回合。下一次方向/清理为365，整体框架为370。

第360轮[同步修复](results/paper_recovery_20261004/joint_reference_candidate_v1/round360_sync_repair.json)已实证：80批有界对象整理与位图生成后，旧快照最后包仅926553字节，已普通快进到远端并清除辅助引用。后续大归档及最新HEAD仍需有界传输，尚非全部同步。主历史和科学数据保留；实时状态见`.git/sync-transfer.json`，以remote与local HEAD一致为最终同步验收。

## 给老师展示：115～380mm手动控制

在项目根目录的桌面终端运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wmt/wheel_leg_robot_work/.venv/bin/python /home/wmt/wheel_leg_robot_work/wheelleg_warp/manual_demo.py --height 0.115
```

地面演示直接复用当前v8五节点固定7kg VMC/六状态LQR、径向支持、实际扭矩包络及共享停车参考。地面和跳跃物理都在MuJoCo Warp GPU积分；跳跃复用原主机状态机与PackedPhysics逐步同步，窗口标明GPU physics/host jump control，没有声称跳跃控制已完全GPU化或比CPU更快。没有复制控制增益或挂载未训练PPO权重。窗口显示的高度是目标腿长与两腿平均FK腿长，不是机身离地高度。

| 按键 | 操作 |
| --- | --- |
| W/S | 按住前进/后退，松开停车，默认1m/s |
| A/D | 按住左/右转向，转速目标0.6rad/s（约34°/s），松开回航向保持；窗口显示指令与实际角速度 |
| ↑/↓ | 以20mm/s调节目标腿长，115～380mm |
| 1/2/3/4/5（含小键盘） | 切换115/160/250/300/380mm预设并重置姿态 |
| 7（含小键盘） | 请求GPU物理跳跃；蹲伏/空中再按可排队一次。起跳时保存原目标高度，落地后自动恢复，115mm也恢复115mm。内部预备姿态仍遵守原范围；飞行中高度键不改写已保存目标。窗口显示等待、执行、恢复或超时；排队最多10秒仿真时间，原安全起跳门保持 |
| F1～F12 | 切换下表场景并重置；保留当前目标高度 |
| T | 从当前目标高度重新初始化，执行原行驶→到达→停车保持任务 |
| B | 切换下一次T任务的前进/后退方向 |
| R | 回到当前目标高度的手动初态 |
| 鼠标滚轮 | 缩放视角 |
| 空格 / Esc | 暂停继续 / 退出 |

展示建议：依次按1～5和T，观察PASS、停车距离、停车后速度及高度RMSE；B后再按T展示反向任务。手动运动不评分，任务模式内目标高度固定，判据沿用原环境。五高度正/反共10任务全门通过，手控前进停车、300→115→380→300mm渐变与双向转向检查通过，证据[演示检查](results/manual_demo_v2_20261003/verification.json)。重置预设用于展示各高度初态，不把重置跳变称作连续调高。

115mm操作复核：点击窗口，按主键盘或小键盘1，Target和Mean FK leg均可显示115.0mm；普通长按↓以20mm/s调高。跳跃落地后自动恢复起跳前目标，恢复轨迹80mm/s，位置速度不重置。再次按高度键可接管恢复轨迹。[高度入口检查](results/manual_height_return_20261003/verification.json)

当前跳跃响应：演示单独将原预备轨迹由35提升至200mm/s，保留原平衡、接触、同步和电机限幅。五预设高度115/160/250/300/380mm按键到真实腾空分别为0.58/0.50/0.62/0.96/1.36秒仿真时间；380mm原为4.38秒。带1m/s前进的300mm也通过。进出跳跃复用预热的GPU模型、CPU状态缓冲与执行图；实测进入7～17ms、落地切换12～13ms，原落地重建约823ms。五高度自动恢复误差<1mm，交接q/v逐值一致，真实窗口两次115mm跳跃均自动恢复。[本轮检查](results/manual_jump_latency_20261003/verification.json)

输入修复：A/D由缓慢累积航向目标改为通过原yaw PD跟踪转速，使用相同增益、力矩及转速包络。300mm按住2秒实际转速由0.0985提高至0.5955rad/s；五高度左右转向/松键、真实腿链/关节/实际扭矩门通过。7键请求不再在SQUAT/空中被一次性消费；松开W/S导致原动态请求取消时会在10秒期限内以当前速度重新请求，持续转向不满足原门则明确提示超时，不强行起跳。真实窗口主7→SQUAT中小键盘7→两次腾空→地面COMPLETED检查通过。[本轮证据](results/manual_input_turn_20261003/verification.json)

| 场景键 | 实际碰撞场景 |
| --- | --- |
| F1 / F2 | 平地 / 左右15、8mm且错位60mm的障碍 |
| F3 / F4 | 2°坡道 / 2°横坡（过渡入口） |
| F5 / F6 | 4mm粗糙路 / 15mm台阶 |
| F7 / F8 | 不对称障碍＋粗糙路 / 连续起伏坡 |
| F9 / F10 | 6/12/18mm连续台阶 / 左右异高约10.5mm |
| F11 / F12 | 单侧坡道 / 左右不对称粗糙路 |

障碍来自现有地形生成器，橙色显示其真实碰撞几何，尺寸、接触、摩擦及控制门保持原值。可用`--scene step`等名字直接启动场景。12场景×5高度×正反共120例，物理/关节设计120/120，完整任务113/120；不对称障碍115/160mm、坡道115mm及起伏坡115mm仍有7例失败，窗口如实显示FAIL。GPU跳跃已验证115/160/250/300/380mm交接及300mm带前进速度起跳，经历蹲伏→起跳→腾空→落地→地面；跳跃与地面任务评分分开。[本轮完整证据](results/manual_scenes_jump_check_20261003/verification.json)

## 历史训练前状态（第30轮）

第30轮height-115统一入口为`NativeEnv.height115_candidate(shared_reference=True)`，v8-design-floor-support、38维观测，固定7kg的J设计位于`native/design.py`。当时CUDA PPO工程准入已通过，协议和readiness见[训练说明](NATIVE_TRAINING.md)。原28全部成功，当时公共148例仍有一例航向失败；工程准入不证明学习优势或高鲁棒性。下面是历史0.30m训练记录。

唯一1024个MuJoCo Warp并行环境基线已经完成正式分轮训练；旧CPU训练实例和CPU/Warp对照实例已退役，公共控制、物理和评估代码保留。

- [正式训练协议、准入与停止规则](NATIVE_TRAINING.md)
- [验证、收敛曲线与清理报告](results/formal_native_1024_20260921/REPORT.md)
- [全环境观测台说明](dashboard/README.md)
- [本机训练页面](http://127.0.0.1:8765/)
- [非结构化地形训练计划](TERRAIN_PLAN.md)
- [terrain-v2最终报告](results/terrain_v2_1024_20260922/REPORT.md)

唯一正式目录：`results/formal_native_1024_20260921/`。训练在第6轮因连续3轮无实质改善停止；最佳为第3轮中点，开发集32/32、Jψ=0.939924°。停止后的独立64例终评为60/64、Jψ=0.937169°，未用于重新选模或续训。

训练服务 `wheelleg-native-formal-1024.service` 已正常退出；前端 `wheelleg-dashboard.service` 保持运行。独立渲染服务在训练结束后停止并禁用，已有总览、详情快照、50FPS录像和轨迹仍可查看。不再运行第二组1024世界，也不再尝试2048/4096。

全部1024世界的姿态总览来自训练GPU状态，点击任意编号查看该世界的真实3D训练帧。详情400Hz记录、显示目标50FPS，实际帧率及延迟显示在页面；评估期间暂停，不生成替代运动。

基线是纯仿真地面M3任务。历史CPU/GPU物理全状态等价门仍有失败；工程可用、开发集效果、最终泛化和实机安全不是同一结论。“最佳”是此预算和停止规则下的最佳检查点，不保证全局最优。

非结构化地形扩展已经完成：terrain-v2新64例OOD为55/64，坡道与20～30mm全宽台阶均13/13；横坡、粗糙路各12/13，组合地形5/12。原基线、terrain-v1、terrain-v2动画分别保留在8765、8766、8767。
