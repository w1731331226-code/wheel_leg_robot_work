# WheelLeg 原生GPU并行基线

## 当前论文计划（376匹配PPO无更新接口已准入，待唯一队列，2026-10-13）

原目标仍是：在原完整不对称接触任务上，得到一个保持强经典能力并有明确增量的可执行PPO方法。当前没有通过收益资格的新方法，正式五种子、独立ID/OOD、统计与成本分析、新方法稿件均未完成；CPU历史基线继续保留，不将物理吞吐当PPO端到端加速。

第376轮新[匹配PPO工程协议](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_ppo_interface_v1/proposal.json)与[114源码准入](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_ppo_interface_v1/source_admission.json)已准备，提交并核HEAD后唯一运行。两Cart各10world、1coldzero后fresh Norm.reset保10课程样本、24×50真实SB3 rollout包，总960800图world步/FD80；policy24000rows，value-only另限24480rows，optimizer/train/learn禁止。新[worker](run_cartesian_ppo_interface.py)补actual/norm/expected终态分名、同期九私有+previous和bank四缓冲、轮速/Nom flags与原指针路由；全新actor/critic/RMS/全RNG/σ=.25和初态逐行匹配。CPU/mock及GPU纯复制自检过，0机器人物理/FD/网络forward；当前尚未真实运行，不晋升PPO/方法收益。

第373轮[独立v2合同](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v2/proposal.json)已注册，两Cart各10world、1coldzero+1200固定正弦包，总960800图步/FD80，0学习。[同输入语义接受](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v2/semantic_acceptance.json)已完成：20个已存完整输入各运行B0/CartLive/CartReset真实phase/parking/Nom/candidate/guard组合，60 CUDA控制行的16公共状态与最终命令严格同，0solver/FD。仅覆盖残差/latent/shadow全零的冷中性输入，不包含旧latent撤回或长期物理路径等价；原失败封存，不按浮点差设容差。

374已完成[新单轨迹worker](run_cartesian_interface_v2.py)、[108源/runtime准入](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v2/source_admission.json)及提交后唯一队列。[实际回执](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v2/execution_receipt.json)记录两Cart各1200正弦包/10world，原graph各1201调用，总960800图world步、35FD、113.21秒含构造与录证，0学习/retry，secondary空。各臂25次部分reset、stage2/3各10行切换、1929行正delay窗口；26个完成回合任务/物理/设计报告均通过。只是原seen训练bank固定脚本描述，不证明PPO增量。所有原单轨迹门实际由producer执行通过，完整独立命令/历史/reset/map数值接收仍待完成，未晋升100world/PPO。

[只读执行接收](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v2/readonly_execution_review.json)已核来源/runtime/预算、24块SHA与事件计数；有效计分物理子步959775，与960800图world步分开。原物理/design摘要通过，访问delay最高10ms；完整命令/历史/reset/map数值接收仍未完成。

375的[独立数值接收](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v2/independent_numerical_review.json)已核959775有效子步的CPU映射、latent/停车、实际returned481、命令环/旧delay窗口、公开reset及Cart三项post状态；误差沿原363/366门，原工件与108冻结源未变。**完整接口仍未通过**：实际done infos481被期望覆盖，中间其它九私有状态post/previous、换bank四缓冲、每slot轮速度与Nom历史flag exposure没有独立同期录证；producer断言与下一pre不代原值。仅有zero/final的九私有缓冲实际post可直接核。

[375方向深审](results/paper_recovery_20261004/joint_reference_candidate_v1/round375_direction_review.json)保留Live/Reset几何更新假设；376将最小录证修复并入必要的三维匹配PPO无更新集成验证，不重复1200×2原脚本补绿。新合同先定world/旧bank/原自然课程/采样与graph/FD/forward预算，再源码/runtime/HEAD准入并提交后唯一执行；当前没有新运行预算或学习准入。之后须冻结强同信息经典及同chart无学习反馈与可达数值主门/保持门，52/52脚本不代表PPO优势。正式新seed/ID-OOD/统计/端到端成本与方法稿仍缺。

[375清理](results/paper_recovery_20261004/joint_reference_candidate_v1/round375_cleanup.json)实际删除8份可重建helper pyc，80299B；源/原始数据/模型/失败及视频保留。下一方向/清理与整体框架均为380。

第372轮[独立接收](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_zero_input_diagnostic_v1/independent_review.json)通过：模型/初态、实际gyro地址、800子步输入/Nom/guard/命令/481历史/终态保存及错误记录反例核过，0新物理或学习。[路线决定](results/paper_recovery_20261004/joint_reference_candidate_v1/round372_route_decision.json)按预登记正式关闭严格跨轨迹内部值等价准入路线，不重跑368/371、不消费余预算，也不归为未执行Cart的方法失败。

373前瞻合同已按374唯一队列执行；同输入coldzero语义与单轨迹硬物理/reset/history分开，未据已见ULP差选择容差或倒推旧失败通过。真实接口、匹配PPO与科学收益仍须分别接收。

第371轮[新执行器](run_cartesian_zero_input.py)与[105源码准入](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_zero_input_diagnostic_v1/source_admission.json)提交后首次运行，已因[同构造B0自身重复性门失败停止](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_zero_input_diagnostic_v1/runtime/failure.json)。两B0各10world一个zero包，共800图步/800物理证据步、35FD、约7.23秒含构造记录，0学习/retry；未构造或运行Cart，不消费余下400图步。

[实际执行回执](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_zero_input_diagnostic_v1/execution_receipt.json)确认三stage各413个实际模型数组/scalars及初态28字段相同；内部微差再次出现。首次已记录差异是第5子步前world6的qvel[3]相差一个float32 ULP，早于gyro与Nom滤波差；Nom最终float32 ctrl、完整guard、公开输入和历史同。证据支持基准输入链先分歧，不能归因未执行的Cart或锁定具体solver/调度原因。372完整独立接收后执行预登记退出规则，关闭当前严格逐值准入路线，不追加复试或按微差定容差；真实课程/延迟接口和PPO收益仍未通过。

第370轮[方向深审](results/paper_recovery_20261004/joint_reference_candidate_v1/round370_direction_review.json)保留CartLive3/CartReset3几何更新假设，只批准一次[匹配构造零输入诊断](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_zero_input_diagnostic_v1/proposal.json)：B0两次、Cart一次，各10环境一个zero包，1200图步/FD80。B0自身严格复现失败即停，事件未复现记不足，不加包/seed或按368微差选容差；368失败目录保持。371须完成新worker、源码/预算准入并提交后唯一执行，再作分支去留判断，不继续反复审同一前缀。当前只批准方向，未准入运行或PPO。

[整体框架核查](results/paper_recovery_20261004/joint_reference_candidate_v1/round370_framework_review.json)确认后续执行器须补验收helper的冻结覆盖、真实graph/world和已提交源码绑定、首因与二次保存错误分列。未来学习还须三维配对包装和Cart观测/动作/checkpoint身份，不能直接连接当前六维脚本工厂或继承旧D3/V6资格。保原D3/V6、强同信息经典及同chart无学习反馈比较；真实部分reset/换bank/正delay与匹配PPO生命周期、科学收益门各自另行准入。历史[paper.tex](paper.tex)及其readiness是旧奖励/通道审计，不证明当前方法目标完成。

[周期清理](results/paper_recovery_20261004/joint_reference_candidate_v1/round370_cleanup.json)删除3份可由现源码重建的编译缓存，共42528字节，原科学记录、源码、模型与失败保持。下一方向/清理375，整体框架380。

## 近期历史证据

第369轮[数值合同审查](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v1/numerical_contract_review.json)完成：滤波差值绝对值虽小，在该数值处却相隔16066441个double ULP，不能当少量末位舍入。现有组合offset与端点不足以分离中间roll/gyro/算术根因；Cart未直接写memory5，原B0/Cart缓存构造也不同。测量方程欠定反例与ULP自检通过，0新物理/学习，368失败和逐值门保持。

已形成[唯一有限输入链诊断提案](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_zero_input_diagnostic_v1/proposal.json)：同三bank构造的B0两次、CartLive3一次，各10环境一个fresh zero包，总1200图步/FD80，补中间输入/Nom/Cart边界记录。B0自身不复现即停，未复现原差异记不足、不追加；新记录会影响执行上下文，不能倒推旧结果。370方向/清理/框架审查须决定是否执行或停止当前准入路线，接受后另做实现/源码准入，不继续反复审同一失败数据，也不自动PPO。

第368轮的[轻量执行器](run_cartesian_light_interface.py)在[101源码准入](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v1/source_admission.json)提交后首次运行，已因[零前缀逐值门失败停止](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v1/runtime/failure.json)：B0与CartLive3各10环境、一个zero packet，共800图步/800物理证据步、25FD、约5.37秒含构造与记录，0学习/重试。CartReset3和正弦队列未运行，不追加或消费余下预算。

[完整失败复核](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v1/zero_failure_audit.json)确认初态28项全同，零前缀31项中29项全同，包括物理端点、公开38/481、完整guard、命令与历史。差异仅原横滚gyro滤波的一个double值（3.32e-18）及41个参考诊断值（最大5.55e-17）；20行原物理/设计监测均过，Cart零请求及λ核过。仍不改判逐值门，端点相同不能确定中间gyro或数值根因。369须有限核对已存数据与数值合同，任何新执行路径/协议另行准入；partial reset、stage2/3、正delay、PPO及收益仍未qualified，370方向/清理/整体框架保持。

第367轮修复了课程切换的参考更新顺序：父类先执行底层自动复位，再安装新阶段参数并再次复位；仅在底层更新CartReset3参考，会留下旧阶段几何。新增课程收尾刷新，使用返回的公开复位包，只处理done行。[三行回归](results/paper_recovery_20261004/joint_reference_candidate_v1/round367_curriculum_lifecycle.json)运行真实父类逻辑及GPU辅助复位，115→380mm例的旧参考偏差约.265m，修复后1.39e-17m，终态旧回合信息、未结束行和父层mask保持；物理reset/forward采用替身，未验证真实多环境运行。

[第367轮接口协议](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v1/proposal.json)固定两Cart臂、10环境、旧33531银行每阶段前10例；每臂1200次固定正弦调用，另有三个零前缀，共961200图物理步/80FD上限，0学习。原门要求真实部分复位、阶段2/3切换和正延迟执行，并独立核命令时钟/历史；预算内事件不足记inconclusive。368已完成40调用轻量实现与源码准入，但该唯一队列在零门失败停止，不能复用稠密80调用资格或剩余预算继续；PPO仍未准入。

第366轮[独立接收通过](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_pair_delivery_repair_v2/review.json)：20回合、298123物理子步、586244接触、7460行Actor记录、238503个Cartesian子步及340份工件SHA核过。重建公开历史与执行命令、潜变量记忆、reset几何、CPU映射/VMC/共同投影和最终guard组装，8组零动作完整流与B0相同。最大canonical误差2.22e-16、原始电机请求误差1.11e-16Nm；停车撤回和终态/复位记录一致。

[当前接受边界](results/paper_recovery_20261004/joint_reference_candidate_v1/round366_delivery_acceptance.json)仅为四个已见、零延迟、每次单环境的交付实现。脚本仍0/20任务成功；未覆盖多环境课程/部分reset/延迟训练接口，未建立PPO收益或稳定证明。367须明确CartLive3/CartReset3的最小匹配训练前置方案及方法区别，否则拒绝扩展；不重复已通过的稠密交付队列，也不直接长训。370方向/清理/整体框架保持。

第365轮[方向审查](results/paper_recovery_20261004/joint_reference_candidate_v1/round365_direction_review.json)允许一次独立修复验证；[修复版20回合已完成](results/paper_recovery_20261004/joint_reference_candidate_v1/round365_execution_receipt.json)，原v1失败保持不变。实际298400图步、298123首回合子步、7460行动作、105FD，约277.84秒含构建与记录；第366轮已完成上述独立接收。不据工程验收宣称方法收益或准入PPO。

[本轮清理](results/paper_recovery_20261004/joint_reference_candidate_v1/round365_cleanup.json)删除7602字节可重建缓存，以及16份已完成回合的重复前缀，共14681297字节；每份均由保留的完整记录逐字节重建并核SHA。所有主记录和v1失败前缀保留。下一次方向/清理与整体框架均为370轮。

第364轮[唯一20回合队列已失败停止](results/paper_recovery_20261004/joint_reference_candidate_v1/round364_failure_decision.json)：实际写完B0与CartLive3零动作两条，消耗30000图步/29988首回合子步、15FD，余18未运行。[失败复核](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_pair_action_candidate_v1/zero_failure_audit.json)确认命令、q/v、接触及保存的gyro/actor记录相同，但140个子步的投影λ不同，不能改判通过。根因是Cart硬编码模式0，而上游已修正Nom实际使用模式1；旧越界标志被错误解释成残差禁用条件，非零请求也会受影响。

已保留v1，并形成[最小v2修复的GPU回归](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_pair_action_candidate_v1/nominal_mode_repair_review.json)：AST核唯一算法变化为投影模式0→1，16行覆盖live/reset、零/非零及历史标志0/1。随后第365轮已按独立准入完成上述新队列，原失败队列未重跑。

第363轮[GPU残差段核对](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_pair_action_candidate_v1/kernel_review.json)完成221184个冻结状态fixture：两种几何模式、零Nom与95%速度包络Nom，最终float32电机命令对CPU参考误差0；最大边界舍入误差1.83e-8Nm，26996行触发共同缩放。失活/错误输入/cold零/按行复位和观察槽位保护检查通过，**仍未组成或运行完整Native闭环**。

[第363轮集成合同](results/paper_recovery_20261004/joint_reference_candidate_v1/round363_integration_contract.json)明确：原Nom按零残差运行，候选随后生成残差，再进原关节guard；实际canonical状态与停车/Actor用的潜变量shadow分开。构造、显式reset、自动reset及terminal观测都要保持所属回合。[原20回合协议](results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_pair_action_candidate_v1/runtime_proposal.json)包含4个低高/正反seen案例、12个零动作和8个固定正弦条件；其640000图步/210FD上限未耗尽，但第364轮已触发停止门，不能继续消费剩余预算。

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

第360轮确定的候选要求继续保留：先给出完整的公共输入到动作映射、模型/学习定义、同权限无学习强对照和区别性预测，再考虑预注册实验。普通QP、DOB、对称化、安全残差或PPO组合本身不是已证创新，见[文献矩阵](../wheelleg_ppo/LITERATURE_MATRIX.md)。不因当前无合格候选把目标缩为负审计论文，不重复已有失败分支或无限添加源资格回合。365方向/清理已完成，下一次方向/清理与整体框架均为370。

第360轮[同步修复](results/paper_recovery_20261004/joint_reference_candidate_v1/round360_sync_repair.json)已实证：80批有界对象整理与位图生成后，旧快照最后包仅926553字节，已普通快进到远端并清除辅助引用。第365轮后续226批也已完成，最后包683236字节；近期提交及本轮完整数据随后按实际包≤64MiB逐提交普通快进，已核对remote与local HEAD一致。主历史和科学数据保留，最新验证快照见[365执行与同步回执](results/paper_recovery_20261004/joint_reference_candidate_v1/round365_execution_receipt.json)。

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
- [历史非结构化地形训练计划](TERRAIN_PLAN.md)（当前论文执行顺序以本文顶部为准）
- [terrain-v2最终报告](results/terrain_v2_1024_20260922/REPORT.md)

唯一正式目录：`results/formal_native_1024_20260921/`。训练在第6轮因连续3轮无实质改善停止；最佳为第3轮中点，开发集32/32、Jψ=0.939924°。停止后的独立64例终评为60/64、Jψ=0.937169°，未用于重新选模或续训。

该60/64及旧terrain-v1/v2逐场景成绩已因静态地形批量初始化问题撤销证据资格，见[TERRAIN_PLAN审计更正](TERRAIN_PLAN.md#静态地形批量初始化审计更正)；此处仅保留历史记录，不用于当前结论或选模。

训练服务 `wheelleg-native-formal-1024.service` 已正常退出；前端 `wheelleg-dashboard.service` 保持运行。独立渲染服务在训练结束后停止并禁用，已有总览、详情快照、50FPS录像和轨迹仍可查看。不再运行第二组1024世界，也不再尝试2048/4096。

全部1024世界的姿态总览来自训练GPU状态，点击任意编号查看该世界的真实3D训练帧。详情400Hz记录、显示目标50FPS，实际帧率及延迟显示在页面；评估期间暂停，不生成替代运动。

基线是纯仿真地面M3任务。历史CPU/GPU物理全状态等价门仍有失败；工程可用、开发集效果、最终泛化和实机安全不是同一结论。“最佳”是此预算和停止规则下的最佳检查点，不保证全局最优。

非结构化地形扩展已经完成：terrain-v2新64例OOD为55/64，坡道与20～30mm全宽台阶均13/13；横坡、粗糙路各12/13，组合地形5/12。原基线、terrain-v1、terrain-v2动画分别保留在8765、8766、8767。
