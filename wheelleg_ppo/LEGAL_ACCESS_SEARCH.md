# 第三篇论文合法全文获取续查

第289轮（2026-10-09）新入口跟进：[arXiv2609.13779v1 HTML](https://arxiv.org/html/2609.13779v1)成功取得Method II-B/C相关文字和公式，较228仅摘要有新证据；未下载图片、视频或再取PDF。GMO残差包含支撑/交互/模型误差，contactprojection含unilateral/friction约束，不能直接挪成本机positive wheel-load lowerbound证明。[IEEE9995759 referencegovernor](https://ieeexplore.ieee.org/document/9995759/)搜索publisher摘要可读，direct返回3行无方法正文；不重复尝试，全文/数值未复现。既有11202537/MDPI429等阻塞入口不重访。本轮普通观测器/参考调节组合不认定新颖，下一步必须清楚区分本机jointworkspace与可得观测/authority条件。

第255轮模型覆盖/gain方向近邻（2026-10-07）：[PMLR2023 Lyapunov Design for Robust and Efficient Robotic Reinforcement Learning官方摘要](https://proceedings.mlr.press/v205/westenbroek23a.html)核CLF costshaping、stabilizing策略及cartpole/A1实机；未复现全文/数值。[2024 affineLPV RL-LQR出版社条目](https://www.tandfonline.com/doi/abs/10.1080/00207721.2024.2321370)仅出版社搜索摘要涉及commonLyapunov，直接403未取全文，不重试。增益调度/CLF+PPO已有，本机有限下一步仅fullcommon/differential模型覆盖准入，不声称新算法/全非线性安全。

第228轮执行输入/history/DOB近邻（2026-10-07，1定向搜索+1primary跟进）：[Force-Aware Reinforcement Learning with Hybrid Sensorless Force Estimation for Wheeled-Legged Loco-Manipulation](https://arxiv.org/abs/2609.13779)，作者摘要标2026-09-12，momentum observer+contact-constrained wrench projection+temporal residual learning给wheelleg force-awareRL输入，有实机loco-manipulation。只核摘要，不复现公式/数值。[2021 UniNA momentum disturbance observer机构摘要](https://www.iris.unina.it/handle/11588/854571)描述quadruped扰动估计/wholebody控制与仿真对照，普通DOB已有。[MDPI14/5/568 history-aware wheelleg](https://www.mdpi.com/2075-1702/14/5/568)只搜索片段，direct429未取正文，不重试。execution-input-history候选仅以本机总command缺口和同信息generichistory控制为可证伪实证问题，普通history/DOB+PPO不称首创/已证论文贡献；不得将ctrl-before-gain惯量差异误称真实contact force。

第215轮支撑/轮速方向近邻（2026-10-07，1定向搜索+1作者入口）：[UCSB Bellegarda/Byl作者PDF](https://web.ece.ucsb.edu/~katiebyl/papers/cdc19_SkateTrajOptWithSlip.pdf)核摘要/建模，passive轮摩擦与允许slip/skid优化，简化不含pitch/roll；不是本机active双轮自平衡数值对照。[2025 contact-aware whole-body作者条目](https://arxiv.org/abs/2509.14010)仅摘要范围，不全面复现。现有滚动/支撑MPC及滑移建模限制本机新颖性：基本anti-spin/轮速阈值或分配不是创新，需在可得观测、闭链变高度和速度-yaw/电机包络上证明区别。没有重复访问此前418/403阻塞入口。

第209轮动态关节/残差停车近邻（2026-10-07，1轮定向搜索+1轮入口核对）：[Sony Tachyon3机构页](https://www.sony.com/en/SonyInfo/technology/publications/real-time-perceptive-motion-control-using-control-barrier-functions-with-analytical-smoothing-for-six-wheeled-telescopic-legged-robot-tachyon-3/)可读摘要，机构标IROS2024，CBF处理关节、碰撞及支撑约束；[Choi等2020作者摘要](https://arxiv.org/abs/2004.07584)可读，RL学习CBF/CLF约束中的模型不确定性。仅核这些摘要，不声称全面复现或相同机器人。新增[轮腿Residual Policy Optimization With Trust Region Constraints出版社线索](https://ieeexplore.ieee.org/abstract/document/11202537)直接418；[重型轮腿Safe reinforcement learning framework for high-obstacle climbing出版社线索](https://www.sciencedirect.com/science/article/pii/S0967066126002145)直接403。检索摘要提示残差trustregion、约束成本/多头Critic等类别，未取全文/不据片段套其公式或数值、不绕过。普通CBF、安全RL和停车退出本身不能作为本机独創；下一步须在可得信息、闭链动态设计边界、接触-停车耦合与电机包络上给出区别及强对照。

第199轮定向访问（2026-10-07）：一轮“residual reinforcement learning/equilibrium/zero at origin”查询后跟进两个机构入口。TU Darmstadt Jascha Hellwig 2023作者PDF可读，83页，仅核封面/摘要/章节结构；不表示已精读全部方法。UC eScholarship条目9s07n7vn搜索片段给出controller difference Eq5.31，直接页面要求JS/robot验证，正文未取得；已记录限制，不换浏览器/重复抓取或据其确认完整理论。来源见LITERATURE_MATRIX最新条目，本轮不访问旧IEEE阻塞全文。

2026-09-17。结论：仍未取得全文；新增已核实作者联系渠道和公开代码线索。下列途径不是全文已经可得的保证，查新状态保持2/3。

## 目标文献

Naifeng He; Zhong Yang; Xiaoliang Fan; Wenqiang Que; Siyang Liu; Hongyu Xu; Chunguang Bu; Bi Zhang.
Residual Policy Optimization With Trust Region Constraints: A Learning Framework for Stable and Agile Wheel-Legged Locomotion.
IEEE Transactions on Automation Science and Engineering, 2025, 22:23352–23365.
DOI: 10.1109/TASE.2025.3620908。IEEE文献号11202537，出版日期2025-10-14。
[出版社入口](https://ieeexplore.ieee.org/abstract/document/11202537)

## 本轮核查结果

| 路径 | 核查及结果 |
|---|---|
| 完整标题/DOI/作者与预印本、accepted manuscript等检索 | 没有检索到可核实的作者稿；不能据此证明公开稿绝不存在 |
| 南航/中科院沈阳自动化所机构域检索 | 未命中本文全文；知识库直接访问失败，不能声称完成库内穷尽检索 |
| 第一作者联系渠道 | 同团队公开论文的作者信息明确给出Naifeng He的邮箱nfhe@nuaa.edu.cn；Zhong Yang邮箱yangzhong@nuaa.edu.cn。来自另一篇公开论文，未假定其为目标论文的当前通讯作者 |
| 作者GitHub | 由同团队公开论文的代码链接确认nfhe账号。wheel_legged_gym为轮腿Isaac Gym/PPO项目，README未声明目标DOI关系，不能认定是本文实现 |
| 轮腿仓库完整目录 | GitHub API返回未截断目录，提交d8948898423ae447be3daa9d9dc8ed2e67fb7ee8；未见PDF/TeX/Bib论文稿。目录已留档 |
| 作者show仓库 | 网页抓取失败、API限流；未完成内容核对 |
| OpenAlex | API返回额度耗尽错误，不是论文OA状态；不得将该错误解释为不存在开放稿 |
| Semantic Scholar API | 网页工具无法打开端点，未拿到开放稿定位结果 |
| ResearchGate | 仍为Request full-text入口，没有取得公开PDF |

邮箱及账号的原始来源：[同团队2024年公开论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435623/)、[出版社代码声明](https://www.mdpi.com/1424-8220/24/18/5925)、[作者轮腿仓库](https://github.com/nfhe/wheel_legged_gym)。

## 可以执行的下一步

1. 向第一作者nfhe@nuaa.edu.cn索取允许分享的作者接受稿/预印本；若无回应，可向共同作者yangzhong@nuaa.edu.cn咨询。同团队公开论文核实了这些学术联系方式，但没有保证邮箱当前可投递。
2. 使用自己所属学校/机构的图书馆文献传递，提交上面的完整书目信息及DOI；是否包含IEEE 2025年卷需图书馆核实。
3. [NSTL全文服务](https://dx.nstl.gov.cn/Portal/sy_qwfw.html)面向中国大陆注册用户提供原文传递；无法直接获得时可申请代查代借。官网说明一般原文传递24小时、成员馆范围内代查代借原则上2个工作日，但本篇是否馆藏、费用及交付时间尚未确认。热线4008-161-200；[用户权益说明](https://www.nstl.gov.cn/user_rights.html)。

尚未发送邮件、提交传递订单、购买论文或代用户注册账号。

## 作者稿索取邮件草稿（未发送）

收件人：nfhe@nuaa.edu.cn
主题：学术阅读申请：TASE 2025轮腿机器人残差策略论文作者稿（DOI 10.1109/TASE.2025.3620908）

He老师您好：

我正在开展轮腿机器人经典控制与残差强化学习方向的研究，希望阅读您与合作者发表的论文：
Residual Policy Optimization With Trust Region Constraints: A Learning Framework for Stable and Agile Wheel-Legged Locomotion，IEEE TASE，2025，DOI 10.1109/TASE.2025.3620908。

目前尚无法通过现有渠道获取全文。请问是否方便提供一份允许用于个人学术研究的作者接受稿或预印本，或告知合法公开访问链接？希望重点学习文中的状态误差补偿、信赖域约束以及实验设置。若有与该论文对应的公开代码，也烦请指引。

感谢您的时间与帮助！

[姓名]
[单位/研究身份]

## 其他合著者及其他团队续查

按Zhong Yang、Xiaoliang Fan、Wenqiang Que、Siyang Liu、Hongyu Xu、Chunguang Bu、Bi Zhang组合目标标题继续检索，仍未找到可核实的公开稿。共同作者公开联系方式还包括fanxiaoliang@sia.cn与cgbu@sia.cn，来源为上述2024年同团队论文，未发送联系请求。

其他团队的可读文献优先级：

1. **Hybrid LMC**，Donghoon Baek、Amartya Purushottam、Joao Ramos，2022：[arXiv全文](https://arxiv.org/pdf/2204.03159)。最接近本机“轮式平衡机器人+LQR+学习补偿”；集成SAC，含历史与上一时刻力矩消融。该版本主要是仿真，不能当作实机验证。原矩阵已收录，本次再次核对。
2. **Adaptive control of a mechatronic system using constrained residual reinforcement learning**，Tom Staessens等，2021作者稿/2022期刊：[公开作者稿](https://arxiv.org/pdf/2110.02566)。第II节区别绝对与相对残差约束，式3为u_total=u_base(1+βπ)。相对约束在基控制输出为零时也令残差为零；因此不宜不加判断地移植到本机需要主动差矩补偿的场景。其稳定性分析有动力学和增益前提，不能把力矩裁剪等同该证明。
3. **Optimizing Cascaded Control of Mechatronic Systems through Constrained Residual Reinforcement Learning**，同团队，2023：[期刊全文](https://www.mdpi.com/2075-1702/11/3/402)、[根特大学PDF](https://backoffice.biblio.ugent.be/download/01GX6M3GSZ40NPSH8AGNWYAJYK/01H39YXMRWMAMJ4QFPNGQGG6VR)。19页全文已下载，SAC串级残差；第2节指出低层闭环可能衰减学习补偿，对本机残差必须在六状态覆盖后、最终限制前注入很有参考价值。

还发现2026年的**Constrained residual reinforcement learning with adaptive bounds to optimize control of a mechatronic system under uncertain conditions**（DOI 10.1016/j.engappai.2026.115343）。出版社预览涉及状态相关自适应残差边界，是应持续核对的新颖性近邻；本次未取得该篇全文，不把它列入已读全文。

这些材料可以补强研究对照，但原IEEE文献的全文缺口仍保留。新增两篇只完成关键方法核对，尚未完成所有实验/复现条件的逐项审计。

新增公开PDF已完整下载并以pdfinfo检查9/19页，关键公式页已渲染核对；来源和SHA256见tools/results/legal_access_2026-09-17/download_manifest.json。

后续核对已完成新增两篇的方法/预算/理论前提及复现条件检查，见[CRRL_FULLTEXT_REVIEW.md](CRRL_FULLTEXT_REVIEW.md)。2026自适应边界续篇已定位根特大学记录，附件为UGent only，仍缺全文。


## 2026-09-21 有界续核

针对差动/航向残差及关键IEEE标题补查，并按新出现的跟踪控制线索跟进，共两轮。IEEE搜索可见的官方摘要和引言仍不包含可完整核对的动作定义，直接正文获取失败；未取得新合法全文。既有作者联系/文献传递渠道不重复检索，邮件未发送。详见[本轮来源及失败记录](tools/results/novelty_2026-09-21/access_record.json)。

新增会议PDF地址返回404；理工大学知识库PDF受网页提取大小限制，本地45 s传输仅取得2287447/10839206字节，续传返回不支持Range。已将文件明确标为.pdf.part，局部文本为.partial.txt，不计入全文阅读完成数。下一步优先取得目标IEEE的合法作者稿或机构全文，而不是继续重复同义词检索。


### 同日用户要求补齐目标全文后的续查

再次定向核对DOI/作者稿线索，结果仍为元数据、Request Full-text或引用本文的其他论文，未获得目标正文。已向用户请求合法PDF本机路径或访问链接。新增SCI/SCIE期刊检索用于完善非学习与RL对照边界，不替代目标IEEE全文，不把题名相近的PDF算作目标稿。未发送邮件或外部求助。

第157轮（2026-10-06）一次定向检索轮式饱和/约束分配，随后直接核三个来源。ScienceDirect轮式论文S2405896320327683与综述S0005109813000368均403，无新合法全文入口，不重复重试。作者机构DiVA [报告2594 PDF](https://www.diva-portal.org/smash/get/diva2:316757/FULLTEXT01.pdf)可读，封面日期2004-02-16；仅核摘要/绪论，不声称完整精读或輪腿直接同平台对比。文献不要求联系作者/订单，本轮未发送任何消息。
