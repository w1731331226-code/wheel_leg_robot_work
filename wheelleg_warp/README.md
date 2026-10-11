# WheelLeg 原生GPU并行基线

## 当前论文计划（第352轮，2026-10-11）

当前冻结方案是 [fixed_force_prequalification_v1](results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_prequalification_v1/proposal.json)：共享名义设计、公开481维输入与关节保护，D3差模力残差与同信息V6残差各三种子、各200k策略样本，只评价最终模型。492回合经典对照、六模型训练和984回合固定评估均已完成；第338轮[独立接收与资格报告](results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_prequalification_v1/evaluation_review.json)通过记录核验，但D3的主任务成功、航向收益和旧能力保持门未通过，本候选收益扩展关闭，正式五种子未准入。

第339轮[已有失败轨迹审计](results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_prequalification_v1/failure_audit.json)核对1476条完整任务结果和378条配对稠密轨迹。D3_33532常规53次失败中48次缺地形接触，主任务24次失败中20次缺指定障碍接触；8个主任务案例有记录姿态下的横向几何分离证据。无障碍反向0.5 m/s也在运动中偏航越5°，首次越界时轮残差与名义纠偏半差力矩约−0.06284/+0.05988 Nm；这是通道干扰的线索，未做因果干预。当前路线偏移已在公开输入内，不能仅据失败断言缺感知或扩大权限。[两案例图](results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_prequalification_v1/failure_witnesses.png)

第340轮已完成[五轮方向深审](results/paper_recovery_20261004/joint_reference_candidate_v1/round340_direction_review.json)、[十轮整体框架核查](results/paper_recovery_20261004/joint_reference_candidate_v1/round340_framework_review.json)及确认冗余清理。当前固定预算候选不值得继续长训；运行链与工程守护可复用，方法优势、独立泛化、稳定性证明和新稿仍缺。V6辅助报告与自身比较的逻辑已修正，D3资格项不变；338原报告和对应源码快照保留。80份冻结运行源未改。

第341轮完成[13回合冻结轮请求诊断](results/paper_recovery_20261004/joint_reference_candidate_v1/frozen_wheel_interference_diagnostic_v1/completion.json)，第342轮[独立接收](results/paper_recovery_20261004/joint_reference_candidate_v1/frozen_wheel_interference_diagnostic_v1/review.json)通过：13初态、全部实际物理子步、模型/RMS重放、原始/掩码请求及原臂历史任务标志一致。D3_33532原输出偏航峰5.44446°且任务失败，关闭轮请求后0.73825°并成功，六对峰值均降；所有关闭轮请求模型的偏航仍高于新B0。支持该固定平地反例中的轮请求干预效应，不证明学习优势或瞬时纯轮力学因果；旧候选收益扩展仍关闭。

第343轮[奖励重建与排序核对](results/paper_recovery_20261004/joint_reference_candidate_v1/frozen_wheel_interference_diagnostic_v1/reward_audit.json)通过，13回合总奖励最大重建误差3.91e-14。按原20ms策略时钟gamma=.99折扣，六组关闭轮请求后回报均提高；D3_33532为1.46835→1.97181，B0为1.97910且仍高于所有学习臂。本批不支持“原奖励把该失败排在改善前面”，不据此改奖励或追加训练；也不能推广为全分布奖励正确或PPO收敛证明。

第344轮[训练覆盖与名义响应审计](results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_prequalification_v1/training_scope_audit.json)核对900个阶段世界、2876个实际完成训练回合及60份更新统计。登记银行没有完整无障碍平地场景，但仍有平地片段和0.3m高度样本，不能据此判定缺训练是根因。三个D3模型在相同、左右测量相等的reset输入上均发出非零差模请求，说明动作反号结构没有自动约束策略的名义响应。

第345轮[方向深审与清理](results/paper_recovery_20261004/joint_reference_candidate_v1/round345_direction_review.json)完成。[有限反射审计](results/paper_recovery_20261004/joint_reference_candidate_v1/policy_reflection_audit_v1/review.json)核384个已存公开输入、768行冻结推理：需区分电机残差整组交换与虚拟请求成对交换，并在冻结归一化之前变换；六模型均有均值非等变偏差。群平均仅离线验证代数和有界性，未施加于机器人；静态名义形态对照不等于全接触/地形/控制动力学证明。

第346轮[19回合冻结反射诊断](results/paper_recovery_20261004/joint_reference_candidate_v1/frozen_reflection_diagnostic_v1/completion.json)完成，第347轮[独立接收](results/paper_recovery_20261004/joint_reference_candidate_v1/frozen_reflection_diagnostic_v1/review.json)通过：初态、351340物理子步、8794策略记录及11108网络查询核过，raw反射/冻结RMS/操作公式正确，六原臂任务标志复现。主要D3_33532反射比半幅yaw峰低.40830°、横向峰低.02532m，但roll峰等辅助指标不全改善；两个V6反射yaw差于半幅，全部反射仍差于B0。单案例主要比较不等于普遍优势。

第348轮[保存轨迹时序核对](results/paper_recovery_20261004/joint_reference_candidate_v1/frozen_reflection_diagnostic_v1/timing_audit.json)确认：主要反射模型在1.0005s起步时yaw约−.00156°，运动中增至−3.241°，7.2505s停车后仅再增至峰3.24488°。2s处名义轮差为0、残差半差−.01226Nm；运动中两者均非零的10446子步里9884步异号。初始零响应消除了直接起步偏置，但不约束后续反馈增益/历史响应；不能仅归因停车，也未证明死区或数值误差是唯一原因。

第349轮[当前任务机会核对](results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_prequalification_v1/task_opportunity_audit.json)保留全40×9结果：经典成功并集32例，学习有8次成功覆盖其余6例，但各模型均丢既有经典成功。26条匹配轨迹显示这8次成功均有正向目标接触，6次轮心全程在障碍横向投影外、2次部分在内；原接触任务成功保留，不等于居中承载完整越障。

[第350轮方向深审](results/paper_recovery_20261004/joint_reference_candidate_v1/round350_direction_review.json)、[整体框架核查](results/paper_recovery_20261004/joint_reference_candidate_v1/round350_framework_review.json)和清理已完成。第352轮[16回合独立接收](results/paper_recovery_20261004/joint_reference_candidate_v1/reflection_retention_falsification_v1/review.json)确认原输出8/8成功复现、反射8/8任务失败，物理/设计16/16通过；跨案例初态与238051物理子步、8937网络查询核过。[统一反射扩展已关闭](results/paper_recovery_20261004/joint_reference_candidate_v1/reflection_retention_falsification_v1/closure.json)，不调幅或重训救分。原接触任务成功保留，不因路径描述事后改判，也不推广为所有对称RL无效。

第353轮仅综合已接受的路径/接触暴露变化与原任务含义，形成下一研究问题的有限依据；不开展新的反射队列。355方向/清理、360整体框架复审保持。

同步方面已修复超时残留子进程，并启动历史保持的64MiB分批对象传输；临时辅助引用仅用于传输，主分支普通快进后清除。运行状态在`.git/sync-transfer.json`，尚不能声称远端追平；不改主历史或删除研究数据。

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
