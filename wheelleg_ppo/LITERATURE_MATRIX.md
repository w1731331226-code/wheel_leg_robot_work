# PPO论文查新矩阵

最近方向复核：2026-09-27。详见 [三篇全文核对](LITERATURE_FULLTEXT_REVIEW.md)；已关闭其中两篇，第三篇仍缺全文。用于训练前确定对照与研究边界，不是“创新已确认”的证明。正文细节仅填写本次实际可访问的内容；“未核实”不等于不存在。没有用其他平台的成功率与本机横向排名。

| 工作 | 平台/接触与基控制器 | 动作/观测 | 约束与实验 | 阅读/实现状态；本机对应 |
|---|---|---|---|---|
| [Residual Reinforcement Learning for Robot Control](https://arxiv.org/html/1812.03201v2)，2018预印本 | 接触装配；传统反馈加学习补偿 | 控制信号叠加；具体任务观测见原文 | 真实装配任务 | HTML正文可访问；本机不复现装配。残差叠加本身已有先例 |
| [Hybrid LMC](https://arxiv.org/pdf/2204.03159)，2022 | SATYRR双轮人形；WIP四状态LQR，yaw/高度另用PD | 集成SAC补偿轮力矩；15维输入含状态、上步力矩与误差历史 | MuJoCo参数变化；算法中有tanh；并非本机的六状态/VMC三差动通道 | 已核对第II～III节及算法1；未确认公开可运行仓库。直接对应强LQR＋通用残差对照 |
| [Action Space Design in Reinforcement Learning for Robot Motor Skills](https://proceedings.mlr.press/v270/esser25a.html)，CoRL 2024，PMLR论文集2025 | 含轮腿、四足和其他控制任务 | 比较动作表示；强调初始化和策略步间行为的影响 | PPO、100组参数、每组3种子；200Hz物理及25～200Hz策略/控制比较 | 已核对12页全文及附录表3～11；7种表示、3种子、跨表示初始化与子步更新实验。对应B2/B2-V及初始力矩分布检查 |
| [Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter?](https://arxiv.org/abs/1611.01055)，2016预印本/2017论文 | 平面关节角色的步态模仿；动作表示含局部反馈 | 力矩、肌肉激活、目标角度、目标角速度 | 比较学习速度、稳健性、动作质量与查询频率 | 官方摘要核对；不是同构轮腿。不能把低维或有反馈的动作空间统称首次提出 |
| [Residual Policy Optimization With Trust Region Constraints](https://ieeexplore.ieee.org/abstract/document/11202537)，2025 | 轮腿地形切换；无模型RL框架中的残差状态补偿 | 状态误差补偿、双源对比学习；具体维数未核实 | 信赖域及不确定性校准；完整实验设置未核实 | 官方摘要/引言可检索，全文未取得。不能把它当成本机动作残差的已复现实现 |
| [基于TD3-PID-VMC动态混合控制双足轮腿机器人运动优化方法](https://xk.sia.cn/en/article/doi/10.13976/j.cnki.xk.2026.0153)，2026-06-23在线 | 双足轮腿；PID与VMC | TD3输出2项Softmax权重（1自由度）；式23列观测项但总维数未展开 | 摘要报告仿真和硬件；对照含固定权重与LQR-VMC | 已核对13页全文第2～4节、算法1及表2；2e6训练步，10次实验重复；带噪权重归一化等复现缺口见全文核对。区别是融合权重与固定基控制器上的差动力/力矩补偿 |
| [Adaptive Fuzzy-LQR for Stability Control of Bipedal Wheel-Legged Robots on Variable Terrain](https://onlinelibrary.wiley.com/doi/10.1155/joro/7752149)，2026 | 双轮腿；Fuzzy-LQR和髋部PD | 姿态相关增益/重心处理与roll补偿 | 正文第6节有站立、高度变化和低速变化斜面实验 | 已核对正文；第6.3节斜面实验固定中间姿态LQR参数，不能全归因于在线增益调度。支持认真构造B1 |
| [Adaptive multi-mode locomotion for bipedal wheel-legged robots via sparse mixture-of-experts deep reinforcement learning](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1788395/full)，2026-02-25 | Diablo；Isaac Gym；低层PD | 六维位置偏差动作，PPO＋稀疏MoE；正文列出本体状态和10帧历史 | 滚行/抬腿切换；4096环境；仿真结果 | 已核对方法与数据声明；原始数据按作者声明提供，未核实代码仓库。本机暂不做多技能或MoE |
| [Horizon-stability control for wheel-legged robot driving over unknow, rough terrain](https://www.sciencedirect.com/science/article/pii/S0094114X24003148)，2025卷期 | 轮腿平台；自适应阻抗、地形适应、姿态/高度控制 | 跟踪地面反力并调节机体水平 | 仿真及实物；完整参数未核实 | 官方摘要与章节预览核对；全文未取得。水平平台控制本身不是空白 |
| [Robust Quadruped Jumping via Deep Reinforcement Learning](https://arxiv.org/abs/2011.07089)，2020预印本，后续2024期刊版本 | 四足跳跃；轨迹优化结合RL | 增强优化得到的跳跃解；细节以对应版本为准 | 考虑扭矩-转速及总功率；报告实机迁移 | 作者预印本摘要核对；不能将预印本与后续版本细节混用。本机仅借鉴约束边界，跳跃不进入当前学习任务 |
| [Safe reinforcement learning framework for high-obstacle climbing in heavy wheeled-legged robots](https://www.sciencedirect.com/science/article/pii/S0967066126002145)，2026 | 340 kg重型轮腿越障 | NP3O、多成本Critic、非对称Actor-Critic | 约束RL及抑制往返刷奖励的终止设计；摘要报告实机 | 官方摘要/贡献段核对，全文未取得。不把末端力矩裁剪称为同等安全RL或安全证明 |
| [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)，2021 | RL统计评估，非轮腿控制方法 | 少量重复下的聚合统计与不确定性 | 强调区间和评估纪律 | [作者评估工具](https://github.com/google-research/rliable)公开；本机按训练种子统计，不把episode当训练重复 |
| [Adaptive control of a mechatronic system using constrained residual reinforcement learning](https://arxiv.org/abs/2110.02566)，2021作者稿/2022期刊 | Staessens、Lefebvre、Crevecoeur；PI与曲柄滑块 | 绝对残差和相对基控制输出的残差约束 | 特定系统假设下Lyapunov分析；真实机电实验 | 9页公开作者稿已按方法、定理前提、预算与复现条件核对，详见CRRL_FULLTEXT_REVIEW.md；约束残差并非本机首创，不能把其证明移植到轮腿接触切换 |
| [Optimizing Cascaded Control of Mechatronic Systems through Constrained Residual Reinforcement Learning](https://www.mdpi.com/2075-1702/11/3/402)，2023 | 同团队；串级控制、双电机传动系统仿真 | SAC；串级残差与适配Actor结构 | 讨论控制环对残差扰动的衰减及学习效果 | 根特大学19页全文已按方法、线性分析、20次训练/评估及复现条件核对，详见CRRL_FULLTEXT_REVIEW.md；本机应交代残差插入位置，不能只称“在控制器上加RL” |
| [Constrained residual reinforcement learning with adaptive bounds](https://biblio.ugent.be/publication/01KVA83TWHK288BDR6SKJF16C6)，2026 | Staessens团队；经典控制与约束残差 | 摘要提出状态相关边界、双环优化 | 概率模型调节残差界；完整细节未核实 | 官方知识库13页出版稿标注UGent only，未取得全文；本机不得以自适应余量单独宣称创新 |

## 已形成的研究约束

1. Hybrid LMC已将LQR、学习补偿和样本效率放在同一问题中。本机必须具体比较不对称接触、虚拟动作坐标与差动参数化，不能只展示“PPO优于裸LQR”。
2. B2-V与M共享映射及输出约束；B2-V仍为六维，M为低维，因此只检验参数化整体价值。同维数非差模对照未做前，保持有限归因。
3. 起始随机动作在平地、不同腿高、单侧接触三个实际状态检查过分布；不同可达集合保留披露，不声称完全同分布探索。
4. B1的8候选选择集整定已于9月21日完成；正式多种子训练与消融尚未启动。代码准入和短PPO更新不构成论文方法优势。

## 尚未关闭的全文查新项

动作空间设计与中文TD3-PID-VMC的全文核对已完成，细节和复现限制见上述报告。残差信赖域一篇仍缺合法全文，不能用摘要替代；正式创新排他性结论仍待确认。


## 2026-09-21 训练前方向复核：研究准入暂不通过

本轮定向检索两轮，复核官方IEEE摘要/引言、Hybrid LMC方法正文及PMLR动作空间研究摘要，并对照本机现有全文笔记与已见开发结果；不是全领域穷尽查新，不以搜索未命中证明不存在同类工作。新增来源/访问失败记录见[访问记录](tools/results/novelty_2026-09-21/access_record.json)。

| 拟主张 | 当前判断 | 可保留的范围 |
|---|---|---|
| LQR/VMC上叠加PPO，改善跟踪 | 不足以作为新方法贡献：Hybrid LMC已研究LQR与学习补偿 | 仅作控制架构背景；换SAC为PPO不是创新依据 |
| 差动/低维动作更易学习 | 尚不能确认独立创新；动作表示、初始化及子步行为已有系统研究 | 当前M/B2-V只能检验参数化整体，不能单独归因于差模先验 |
| 输出限幅或转速余量λ | 约束残差已有先例；本机无新增稳定性定理 | 作为执行器可行化实现，不称自适应安全边界或安全保证 |
| 航向精度替代成功率作为主指标 | 是问题收敛和评估协议修订，不是方法创新 | 如实披露训练前改题依据，不将15%/0.05°门槛包装成理论贡献 |
| 不对称接触下，腿部差动补偿的航向—速度折中 | 值得检验，但尚未确认新颖性与学习必要性 | 以接触条件相关耦合、强经典对照和动作子空间失效边界为研究对象 |

直接证据：[Hybrid LMC第II节](https://arxiv.org/html/2204.03159)叠加LQR与学习力矩，yaw/高度另用PD；[动作空间论文](https://proceedings.mlr.press/v270/esser25a.html)明确研究轮腿平台，并指出初始化和策略步间行为的重要影响。[IEEE目标论文摘要](https://ieeexplore.ieee.org/abstract/document/11202537)涉及状态误差补偿、非线性接触和信赖域；尚不能排除其完整实现中含相似差动结构。以上差异不能据摘要升级为“本机首次”。

### 推荐方向与证据缺口

建议保留的问题表述为：**不对称接触下，差动虚拟动作子空间对双轮腿机器人航向—速度折中的作用与边界。** PPO是验证工具；当前题目仍为候选，不承诺新算法或可投稿贡献已成立。不增加教师、MoE或额外网络来制造复杂度。

本机支持该假设的有限证据：旧脉冲采用同一0.1归一化幅度、20 ms脉冲和300 ms观察；ΔT通道在平地的航向峰值变化约0.0322°，不等腿高/单侧接触状态中一方向约0.1770°/0.1825°。这些是相对零残差轨迹的变化幅值，不表示改善；不同通道物理尺度也不同，不能拿峰值大小直接排名控制权。说明接触相关作用值得研究，但不证明PPO比反馈更必要。来源：[18组脉冲](tools/results/fixes_2026-09-17/pulses/pulses.json)。

B1在固定32例上已将Jψ从0.762952°降到0.613519°（约19.6%），保持28/32成功。这说明简单手工反馈有贡献，而不是经典控制已达到理论天花板；8候选只是本次预算内比较，不等于最优经典控制。学习是否有额外收益仍未知。

训练前需要关闭的两个判断：

1. **关键全文重合排查。** 对IEEE 10.1109/TASE.2025.3620908补齐补偿位置（状态/参考/动作）、精确动作维数与左右分配、基控制器、观测及特权信息、约束公式、不对称接触实验这六项。当前合法全文仍未取得，此项未通过。可合法分享的作者接受稿或学校文献传递是具体下一入口；现有[邮件草稿和获取记录](LEGAL_ACCESS_SEARCH.md)保留，未代发邮件或提交订单。
2. **贡献识别而非仅分数提升。** 若目标是主张差模结构的作用，建议将轮差矩一维M1与同维数、同VMC映射的固定非差模混合子空间对照前置；后者须实际改变可达子空间，不能仅旋转原差模坐标，也不能事后选择最弱的随机基。幅度、初始化分布、预算及基的选定规则必须在新增训练前注册。M1用于回答腿部通道是否必要，同维对照用于减少普通降维解释。当前第6.4节冻结方案并未包含这两项主比较，因此这里只记录待落实的研究修订要求，不暗改旧协议或宣称已完成新实验。若仍只做原M/B2-V比较，结论必须保持“参数化整体”的探索性工程结果，不能称差模先验创新已验证。

**决定：工程准入保留；研究方向暂保留为可证伪假设，研究准入不通过，长训练暂停。** 不可能在训练前证明学习优势，但可以先证明比较对象、贡献边界和验证设计足够明确；目前仍缺上述证据。若全文发现本质同构，应改为复现/边界研究或缩题，而不是换名继续。门控集和最终集保持未评估。

### 本轮新增线索的边界

检索命中Rough Terrain Path Tracking of an Ackermann Steered Platform Using Hybrid Deep Reinforcement Learning，会议源索引出现LQR转向残差式，但链接当前404；仅作为跟踪残差已有邻域的线索，不列为全文已读。理工大学知识库的Adaptive and Robust Wheel-Legged Biped Robot for Semi-Structured Community Tasks全文约10.8 MB，传输仅得到不完整文件且服务器不支持续传；不据局部页面填写完整方法/实验结论。与本机不直接同构的无人机对接和六足CPG结果未扩充为核心文献，避免扩大任务。


## 2026-09-21 补充SCI/SCIE期刊线索（按用户要求）

以下新增3篇按控制机制选取，不以影响因子代替相关性。Machines的[出版社当前刊物信息](https://www.mdpi.com/journal/machines/imprint)明确列SCIE；Actuators的[出版社收录公告](https://www.mdpi.com/journal/actuators/special_issues/feature_papers_SCIE)明确SCIE覆盖。本轮没有取得Clarivate逐篇入库记录，因此这是期刊层级的出版社收录证据，不冒充逐篇WoS检索证明，不填写未经核实的分区。

| 论文 | 期刊、年份与DOI | 实际核对范围与对本机的作用 |
|---|---|---|
| Real-Time HILS Comparison of Full-State Feedback and LQ-Servo Tracking Control for a Wheeled Bipedal Robot | Actuators 2026,15(3):170；[10.3390/act15030170](https://www.mdpi.com/2076-0825/15/3/170) | 出版社索引可读摘要及方法/限制段：比较普通全状态反馈和含积分LQ-Servo；矢状面HILS，不覆盖yaw/roll和滑移。提醒不能将持续偏置改善自动归功于RL；不能直接当本机三维接触控制的已复现基线 |
| Extended Kalman Filter-Enhanced LQR for Balance Control of Wheeled Bipedal Robots | Machines 2026,14(1):77；[10.3390/machines14010077](https://www.mdpi.com/2075-1702/14/1/77) | 出版社摘要及第3节说明EKF平滑的是LQR输出力矩，不是状态估计；报告硬件平衡/行驶实验。提醒核实补偿插入位置，并避免将平滑效果包装成学习优势；本机不因此新增滤波器 |
| Multi-Agent Reinforcement Learning Tracking Control of a Bionic Wheel-Legged Quadruped | Machines 2024,12(12):902；[10.3390/machines12120902](https://www.mdpi.com/2075-1702/12/12/902) | 出版社方法/结论段：Pegasus四轮腿、逐腿多智能体、运动引导，含硬件部署；与本机固定LQR上的低维差动残差不同。用于区分引导信号和控制器结构，不照搬多智能体或跨平台比较航向误差 |

上述页面的关键段落已核对，但未逐页审计全文所有表格/公式、未复现，不将它们计入关键IEEE全文任务完成数。直接最接近的约束残差仍优先使用已读的IEEE TIE 2022作者稿、Machines 2023串级CRRL与缺失的TASE 2025目标全文，而非用新增数量替代关键重合比较。

对照设计已具体化为M1、M3、N3±，矩阵及解释规则见PAPER_PLAN.md第12节；代数检查通过，但尚未加入训练接口或做物理分布验收。关键IEEE六项已知/未知比较见LITERATURE_FULLTEXT_REVIEW.md第4节。研究暂停状态不变。

## 最新执行口径：不再追加全文门槛

用户明确无需继续追索IEEE全文。现有文献用于支持立题与贡献验证设计，保留该篇实现细节未核实的限制；此前“必须取得全文才能训练”的决定不再适用，也不因文献数量不足而继续扩充EI/SCI。创新是否成立仍需强经典对照、一维轮差矩及同维非差模比较，不能预先宣布。

M1/N3±现已接入环境并完成三类已见状态的物理分布及局部集成检查，见PAPER_PLAN.md第12.4节。整体力矩RMS接近、未见该批样本的饱和偏置，但各电机协方差仍不同，不能说已完全排除探索差异。下一项是v2训练配置/短更新验收；门控与最终集继续封存。

上述v2下一项已完成，独立协议和新增三方法各4000步验收见PAPER_PLAN.md第12.5节。当前可以进入受限贡献的预实验；“研究协议就绪”仅说明比较设计和执行规则已落实，不改变本矩阵对已有工作、新颖性缺口及未验证方法优势的判断。无需再以IEEE全文获取或追加EI/SCI数量阻止该阶段。

## 2026-09-27 GitHub 与 SCI/EI 近邻复核：创新点仍待实证

本轮按用户要求重新定向查找“双轮腿＋经典控制/残差学习”“不对称地形与越障”“动作空间/约束”近邻，逐项核对论文官方页、作者公开稿或 GitHub 仓库；没有运行外部代码，也不是全领域穷尽查新。已有 Hybrid LMC 的 LQR＋RL、CoRL 动作空间研究和 TASE 2025 轮腿残差策略见上表，以下只列本轮对结论有增量的来源。出版社[《Sensors》索引页](https://www.mdpi.com/journal/sensors/indexing)与[《Electronics》索引页](https://www.mdpi.com/journal/electronics/indexing)列出期刊层面的 SCIE 和 Ei Compendex 覆盖；未取得逐篇 WoS/EI 入库记录。其他会议、预印本与期刊的 SCI/EI 状态均不据题名推断；正式选刊仍须查 [Clarivate Master Journal List](https://mjl.clarivate.com/) 和 Engineering Village 及学校目录。

| 新核对工作 | 与本题最接近之处 | 已知差异及证据边界 |
|---|---|---|
| [Hybrid Balance Control for Wheeled Bipedal Robot via Residual Reinforcement Learning Optimization](https://doi.org/10.1109/ACIRS66343.2025.11360885)，ACIRS 2025 | 同为轮腿、模型平衡控制上叠加残差 RL | IEEE/Crossref 出版元数据可核；IEEE 正文当前不可读，动作维数、是否差模、是否 PPO 与实验细节**未核实**。因此不能声称已排除方法重合，也不能说本文已经同构 |
| [Friction Feedforward LQR](https://www.mdpi.com/1424-8220/25/4/1056)，Sensors 2025，DOI 10.3390/s25041056 | 双轮腿、LQR、单侧桥面与扰动实机测试 | 主要补偿轮电机摩擦，说明裸 LQR 不是唯一强经典对照；本机未验证同等电机摩擦模型，不能照搬数值或称复现 |
| [Whole-Body Control with Uneven Terrain Adaptability Strategy](https://www.mdpi.com/2079-9292/14/1/198)，Electronics 2025，DOI 10.3390/electronics14010198 | MPC 力分配、姿态规划、左右轮高差、单侧桥面/坡/阶梯 | 非学习控制已有不对称地形能力；平台、动力学和指标与本机不同，不横比成功率 |
| [A novel adaptive dynamic optimal balance control method for wheel-legged robot](https://www.sciencedirect.com/science/article/abs/pii/S0307904X24004906)，Applied Mathematical Modelling 2025，DOI 10.1016/j.apm.2024.115737 | 在线增益迭代与 VMC 映射，在不平地仿真验证 | 已核官方摘要与方法预览；提示固定 B1 增益扫描不代表经典控制上限，全文实验细节未核 |
| [Reinforcement Learning for Blind Stair Climbing with Legged and Wheeled-Legged Robots](https://arxiv.org/abs/2402.06143)，ICRA 2024 作者稿 | 轮腿/双足盲爬台阶，含 Ascento 实机 | 越障 RL 已有先例；其 15 cm 台阶与本机 24～27 mm 不是可直接比较的任务 |
| [CTBC: Contact-Triggered Blind Climbing](https://arxiv.org/html/2509.02986v3)，作者预印本v3，2026-02-07 | 作者全文已核：轮足接触力三帧滑窗（示例阈值30 N）触发单侧优先抬腿；髋/膝位置动作与约0.6 s余弦前馈轨迹、退火示教和PPO协同，已有实机盲爬 | 本机目前只测试100 ms的M3差动支撑残差，**不能**据其失败反驳CTBC；本机0.16 m名义下限无额外收腿余量，直接移植收腿抬轮不能覆盖用户的全高度目标。未核实正式IEEE发表与逐篇SCI/EI收录；接触/预瞄若作为新分支须与其同信息强对照 |

GitHub 代码近邻（仓库均可访问；只核 README 与相关实现，不复现实验）：[Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) 与 [wheel_legged_gym](https://github.com/nfhe/wheel_legged_gym) 均已有 PPO＋VMC 六维左右参考；[Wheel-Legged-Lab](https://github.com/zyicome/Wheel-Legged-Lab) 还提供六维 VMC 动作、轮速 PI、障碍课程及检查点；[mjlab_upkie](https://github.com/MarcDcls/mjlab_upkie) 已在 MuJoCo Warp/MjLab 中做并行双轮机器人 RL。仓库平台和任务不同，不能把 README 的收敛步数或视频与本机横比；但 PPO、VMC、课程、轮速积分及 GPU 并行均不能单独作为本论文创新点。

**判断。** 研究问题“未知不对称接触下，差动虚拟动作子空间是否改善航向—速度折中及其失效边界”仍合理、可证伪；**现阶段不满足已证实方法创新的证据要求**。已有工作占据“经典控制＋RL”“轮腿 PPO/VMC”“地形越障”和“残差约束”的宽泛表述。受限检索尚未确认完全相同的三维差动映射，但 ACIRS/TASE 正文缺口使“首次提出”没有依据。当前 CPU 航向协议要求 M3 对最强对照同时降低至少 15% 和 0.05°、保持成功/速度/姿态非退化并经多种子配对；原计划只记录 M3/1609 的 20 万步阶段最优 Jψ 0.803941°，仍差于 B1 的 0.613519°。本机当前找不到该阶段引用的 `pilot_v2_M3_seed1609_resume1` 原始目录，故此轮无法独立复核该历史分数。独立 Warp 地形契约下 terrain-v3 困难台阶三次 32/32/33，对原 GPU 35/35/35；最近 24/27 mm 接触前四种单轴动作偏移在四例均失败。两套指标/场景不同，不能合并成一项已获优势的论文结果。

**最值得继续的方向：接触—电机包络约束下的合法动作与差动分配机制。** 先固定任务、原速度命令和电机/姿态边界，找出可重复通过的动作及其所需观测；再单独检验轮速限幅时统一 λ 是否不必要地抑制可行腿部/轮部动作。已有 2 kHz 轨迹提供这一假设的线索，尚未证明它是唯一根因。若合法动作不可达，应转为物理/控制限制分析，停止堆 PPO；若可达但 32 维观测不可区分接触，应明确增加可实现的接触/位置感知并与 CTBC 类工作比较；只有动作可达、信息足够而标准 PPO 在配对预算下仍学不会时，才有理由单独诊断 KL、价值误差或奖励并试一项最小 PPO 修改。现阶段不建议为了“创新”改 PPO 损失、加大网络或加教师。

若继续现有差模论文，必须让已注册的 B1、M1、同维 N3±、B2-V 与 M3 在同任务、同后端、同预算下比较并逐例重复，主张限定为本任务中的动作子空间作用；高台阶成功率与 CPU 航向 Jψ 各有独立冻结门，不因某一侧的好看数值改写另一侧门槛。若到冻结比较时仍无稳定优势，应将结果诚实写为失效边界/比较研究或缩题，不把工程实现包装成 PPO 算法创新。具体 SCI/EI 期刊是否适配、录用概率和学校认可资格需按目标刊物单独核对，本轮不能保证。

### 主流机器人论文与官方开源项目补查

下表优先选择与本题方法或台阶能力直接相关的 Science Robotics、IEEE T-RO/RA-L、ICRA/IROS、CoRL 来源。“影响力”指研究载体、公开实机/方法与后续可复核资源；不以期刊名或 GitHub star 数替代本题的重合度判断。正式发表版、作者稿和只读摘要分开标明。

| 工作 | 已核对的核心内容 | 对本题创新性的约束 |
|---|---|---|
| [Klemm 等 LQR-assisted WBC](https://doi.org/10.1109/LRA.2020.2979625)，IEEE RA-L 2020，[作者稿](https://arxiv.org/abs/2005.11431) | Ascento 闭环运动学、全身控制和 LQR 平衡任务，实验有粗糙地形与弯道 | 双轮腿的 LQR＋多任务接触控制已成熟，不能称本机基控制架构首次 |
| [Klemm 等 Non-Smooth Trajectory Optimization](https://marco-tognon-robotics.com/content/non-smooth-trajectory-optimization-wheeled-balancing-robots-contact-switches-and-impacts)，IEEE T-RO，DOI 10.1109/TRO.2023.3326334 | 作者公开稿/项目页：接触切换、冲击、附着及执行器边界进入 Ascento 轨迹优化，LQR 跟踪并实机越台阶/楼梯；2023 在线、2025 卷期 | 受约束越障无需 PPO 也可能实现；本机需说明未知接触与在线差动补偿相比已知接触相位规划的确切不同 |
| [Lee 等 Wheeled-Legged Navigation](https://arxiv.org/abs/2405.01792)，Science Robotics 2024，DOI 10.1126/scirobotics.adi9641 | 作者稿：四轮腿的学习式地形运动/导航，真实城市长距离任务 | 多地形 RL 与真实部署已有高水平先例；虽与双轮自平衡不同，不能把“复杂地形泛化”本身当创新 |
| [Chamorro 等盲爬楼梯](https://doi.org/10.1109/ICRA57147.2024.10610069)，ICRA 2024，[作者稿](https://arxiv.org/abs/2402.06143) | 轮腿与双足的 RL 盲爬台阶，Ascento 实机示例 | 台阶 PPO/RL 任务并非空白；机器人尺度与评价协议不同，成功高度不可横比 |
| [Yang 等 Integrated Modeling and Control](https://doi.org/10.1109/LRA.2024.3522773)，IEEE RA-L 2025 | 可核元数据/摘要：VMC 快速映射、自调 LQR、转向及重力补偿，仿真和硬件 | 与本机 VMC＋LQR＋航向控制高度相近；正文未核，不能照摘要称完全同构，但经典对照不能只保留裸 LQR |
| [Multi-Loco](https://proceedings.mlr.press/v305/yang25a.html)，CoRL 2025 | 官方论文页：含轮式双足，以扩散先验＋RL 残差比较标准 PPO，仿真及真实平台 | 轮腿残差与 PPO 比较已有先例；其跨形态数据/生成先验与本机低维差模不同，不可直接移植增益数值 |
| [Integrated Motion Control on Rugged Terrain](https://doi.org/10.1109/TMECH.2025.3627438)，IEEE/ASME T-Mech 2025 | 官方元数据与作者机构记录：双轮腿浮动基动力学、分阶段平衡与复合状态估计，实机越障 | 不平地强非学习控制已有实机成果；全文机制未核，不能把本文机器人表现直接作为本机数值基线 |
| [Robust Whole-Body Balance for Perception-Less Terrain Adaptation](https://doi.org/10.1109/ACCESS.2026.3679744)，IEEE Access 2026 | 出版元数据与作者稿：姿态/力规划、LQR 轮矩、QP 分配，Gazebo 坡道/楼梯/不平地仿真 | 纯仿真、无视觉和轮腿地形适应本身已非独特点；作者全文与本机接触/约束协议仍不等价 |
| [Whole-Body MPC and Online Gait Sequence Generation](https://www.markobjelonic.com/publications/files/2021_iros_bjelonic.pdf)，IROS 2021，DOI 10.1109/IROS51168.2021.9636371 | 作者全文：四轮腿 MPC 优化车轮/机身运动与在线接触时序，实机验证 | 接触规划是有力的替代机制，但高维四轮 MPC 不能未经适配就称本机公平基线 |

具有明确研究或工程传承的 GitHub 仓库另核：[ETH Legged Gym](https://github.com/leggedrobotics/legged_gym)（粗糙地形并行 PPO、课程、随机化）、[ETH RSL-RL](https://github.com/leggedrobotics/rsl_rl)（Isaac Lab/Legged Gym/mjlab 使用的 PPO 库）、[DRCL-USC RL-augmented MPC](https://github.com/DRCL-USC/RL_augmented_MPC)（四足 MPC＋学习残差，作者论文报实机）、[mujocolab/mjlab](https://github.com/mujocolab/mjlab) 与 [Google DeepMind MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)（训练框架/物理后端）。这些分别说明 PPO、模型控制＋残差、GPU 并行物理均已有公开基础；四足/框架并不直接解决本机双轮腿差模问题。未运行外部项目，不能用 star 数、README 曲线或异平台视频给本机方法排名。

**本轮方向排序：** 第一，接触及电机包络约束下的动作可达性、可观测性和残差分配，直接对应现有 2 kHz 失败链且可通过单变量干预证伪；若形成方法，须对 T-RO 接触规划、RA-L 强经典与 TASE 受约束残差说明差异。第二，保留差模 PPO 作为已注册的比较手段，在上述机制允许有效控制后再检验其相对 B1/M1/N3±/B2-V 的额外收益。第三，当前无证据支持优先改 PPO 损失、网络或另建教师/世界模型：物理可达性与观测不足尚未排除，修改优化器只会叠加解释变量。上述排序是根据本机与论文证据作出的研究决策，**不是已确立的新方法或可发表性保证**。
