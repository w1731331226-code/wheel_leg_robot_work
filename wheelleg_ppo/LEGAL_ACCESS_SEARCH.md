# 第三篇论文合法全文获取续查

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
