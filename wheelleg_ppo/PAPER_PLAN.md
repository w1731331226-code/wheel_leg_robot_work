# 双轮腿机器人PPO论文方案

第182轮参考角色160新回合采集完成（2026-10-07，目标继续）：

run_reference_role_probe复用full/gyro checker及partial保存，freeze原source/隔离副本/noise/API/runner和原80参考绑定后唯一8job×20首回合完成，new160+reused80=240比较/20发展case、0training。新增physics2380549、全contact4700984；160full1346684703B+160gyro59017402B+160role52644665B共480trace，单份max8925810B。全hash/schema/finite/dense/contactcounts/qvexactjoin/gyro输入递推/noise及role mean/target/保护界限/原header desiredoffset与left-right对齐核过；实际新controller input单列日志，原headerbaseheightref不误当governedmean。

结果计数按cleanB0/cleanB1/noisyB0/noisyB1：original8/8/8/7，floor_only12/12/12/11，consistent_pair12/12/12/12；全部新job physical/design20/20。只是当前注册发展样本计数，不据此证明每case无lost、多轴不退化、普遍噪声鲁棒/纯参考因果或新算法；183必须配对原fullgate并查不受投影区域的数值同一/差异后再归因，不把1个额外noise成功自动当方法优势。

combined full/gyro/role prefix/latestbuffer/frozen-case合成保存通过，实际无interruption/追加回合。worker实见PID8845，现MainPID0/SubStateexited/Resultsuccess/exit0，00:43:52→00:48:25整采集273秒（含锁可能/构图/记录压缩/check，非PPO或purephysics）。round182_data_review绑定数据/源/单位日志。默认controller/env/model/CPU-GPU/model/RMS/原数据/final封存未替换；新增常规角色工程对照不当已达论文贡献。

183多轴/height/任务原gate与paired role去留，184有据剩余方法/学习需要，185按约深审清理。仍未有完整可辩护新方法及稳定学习优势、资格3→正式5seed/消融/新独立ID/组合/geometry/参数-delay OOD/易任务、统计/真实PPO端到端与新稿，不以本轮正计数结束整体目标。

第181轮隔离参考角色源码资格完成（2026-10-07，目标继续）：

reference_role_control保存原control_step/wrapper的可核对实验副本，仅增加optional reference15径向anchor覆盖；原controller/env/model源码不改。reference_role_probe包裹已有full/gyro/noise采集，将实际控制输入换为自有16列ref，并核原/重捕获80个实际controller input signature两段40完全一致，原245corecalls/40steps/4sharedref保留。padding10–14全0保证不意外启用arrival/parking；原112header height_reference明确仍是basecmd，新11列role日志记录actual mean/lower/左右目标/保护floor/虚拟requested-applied，不混淆实际接触力。

三arm×两noise各20world构造/capture，3合成roll姿态静态原control查询：original branch input/state/diag/control exactnoop，支持原80source资格复用；实验函数与原函数差异逐源码核只有可选anchor，未引入新gains/gyroalpha。consistentpair目标界限/mean shift≤17.5mm及保护floor≤目标minimum、最终motorbounds、inactive/reset/owner通过。原冻结噪声/39packet时间层保持，alpha.025；新增nested full/gyro/role合成首终态/重复done不记录/reset清chunks通过。静态.115/+5deg，consistentpair mean.128089969/floor.115，guard0且maxhip4.5924Nm，避免179meanonly的848N虚拟component/40Nm；不是实际动力性能或安全证明。

source/unit/supplemental/proposal/noise/installed forward/support绑定admission_review。本轮无physics积分/评价/学习，original80复用属source/noop而非同次轨迹校对；floor_only仍保原低位tiny target-floor差，consistentpair用operational lower.115，普通角色分离/投影不当新算法。182须freeze唯一160runner与合并partial保存后才执行；183原完整多轴/height/task/phys/design门，184有据剩余方法/学习需要，185深审清理。完整可辩护贡献/3资格→5正式seed/独立泛化/统计PPO耗时/新稿仍未齐。

第180轮方向深审与冗余清理（2026-10-06，下一185，目标继续）：

**去留与继续价值。** 176–179 source/原control-noop/noise/120实际回合和全240新trace哈希再次核过。alpha1降低yaw却新增12配对roll违规，fixedalpha1整体改进保持closed；mean-only投影抬高同字段guardanchor、静态触及40Nm，仍不准部署。真实lowheight guard max与reference关系只支持有限控制角色假设，不指认全部失败/物理模型错误。正确方向是先核强Nom契约，而非盲加PPO/极端场景或普通坐标改名创新；整体方法/训练/泛化目标不能缩为工程负材料。

**仅准入分步角色对照范围。** reference_role_probe_v1/proposal登记3arms×2noise×2laws×20原发展case=240比较记录：original alpha.025的80记录只在source/no-op/noise一致后复用172clean40与177originalnoisy40；新增floor_only80与consistent_pair80共160首回合，训练0。floor_only保原trackingmean/room/下限，只将径向保护参考独立固定为已验证工作下限.115m，k200/质量与转子模型/headroom不变。consistent_pair在相同fixedfloor上，将原hcmd±clippedrolloffset分别投影到[.115,.38]m，再用新mean/原room逻辑及operatinglower.115作为双腿tracking；其他参考angle/gains/gyroalpha.025保持。known危险mean-only+trackingguard不跑，所以不称完整factorial或纯投影单因素效应。

此改变明确放开中低位“每腿不低于平均命令”的保护意图，**不放松原physical/design/torque/task门**；fixed.115高于原geometric.114704m并保最低工作支撑，但动态保护/actual height/joint invariance仍未证明。trackingmean shift<=17.5mm不等于actual RMS/end保证。所有原5deg分轴、每case强classic成功、contact/exit/speed/arrival/stop/tail/height/phys/design仍必须核，不能用yaw改善掩盖新增roll或用限幅当安全。普通fixedfloor/boxprojection不当新算法或正式学习贡献。

181隔离optional源码（冻结原controller/env/model不改）、originalbranch精确no-op、扩展ref shape不得启用arrival/parking、tracking/保护/basecmd及实际input日志、packet/noise/reset/owner/静态资格；182freeze source/runner/原PCG64噪声表与已耗预算后唯一160队列，复用不合格即停、不暗增original回放；183逐law/noise原完整门/角色交互去留；184据结果选择剩余机制/方法/学习必要性；185深审清理。Fn/COM/geometry只eval，schema/cadence/所有gains/physics/model/limits/原data/RMS/final不改，不扫floor/alpha/gain、不默认部署或新PPO。目前仅scope注册、source未准入/未执行。

**清理。** 删除run_nom_yaw_filter_probe/test_nom_yaw_filter_probe两份ignored unused源码等价pyc，共13993B，source/cacheSHA和fuser/codeobject核对留round180_cleanup；再生缓存非永久节约。全部源码/120full+gyro/原40/205/model/RMS/noisetable/唯一失败/CPU-GPU/final保留。round180_direction_review绑定周期和newproposal，完整可辩护方法、资格3→正式5seed/消融、新独立ID/组合/geometry/参数-delay OOD/易任务、统计/真实PPO耗时/完整新稿仍未齐。

第179轮腿参考与径向保护角色核对（2026-10-06，目标继续）：

audit_leg_reference_pair沿原控制链验证：最终fleft/fright用各自实际腿长与左右目标，不是共同高度误差在LQR中完全抵消。固定平均参考的symmetric room在.115仅.2955mm/.38为0；普通pair box projection可以经改变mean再走原room逻辑得到相同左右目标，289网格+边界/invalid检查过，mean变化<=|desiredoffset|/2<=17.5mm。它是常规投影，不是新算法，reference界限不保证actual高度RMS/终态或动态安全。

5高度×3合成roll姿态×2参考方式=30静态输出（另15原control预备缓存计算），不mj_step积分。原kernel输出验证reference实际可达并符合最终电机限幅，但发现关键角色耦合：径向保护anchor=min(reference_mean,.160m)，与trackingmean同一字段。例.115/roll5deg，pair mean=.127942m，shorttarget=.114704m低于guardanchor；保护requested虚拟force6547.565N、limited applied848.151N，hip命令达40Nm。普通mean投影因此不准部署，力矩限幅通过不当整体安全；这不是实际接触力/真实运动结果。原.16短腿目标.133820<guardanchor.16，说明保护意图与差动跟踪可能相互对抗，不能从源码关系宣称物理模型错误或全部失败由此导致。

绑定178全部8结果，另列160记录的真实episode guard max/limited/positive counts作背景；positive可包含极小值，不当显著载荷/duty或因果效应，真实max也不隔离保护选择影响。alpha1仍按178关闭，不救参数。180深审候选仅为先分开trackingmean/left-right targets与有物理依据的保护下界，再审是否有限dynamic对照；fixed .115 operating support floor是待核的常规候选而非已安全的新方法。必须保原physical/design/actuator/低位已验证支持，不能直接换Mean或全局L_SQUAT_MIN、削弱门/添加真值。

本轮0newphysics/eval/learning，无原controller/environment/model变更，只有隔离静态审计；原base/raw/model/RMS/final保持。180按约深审与冗余清理；具体可辩护方法、资格3→正式5seed/消融/新独立泛化/统计/PPO真实成本/完整新稿仍未齐。

第178轮固定滤波变体配对去留完成（2026-10-06，目标继续）：

review_nom_yaw_filter对4组同noise/同law配对、160比较记录/20发展case重建原全部success及flags，并显式列出原5deg roll/pitch轴，避免原已yaw失败的case掩盖新增roll失败。3个added-flag断言通过；8结果/source/177completion/datareview哈希过。四组原→latest成功分别cleanB0 8→8、cleanB1 8→8、noisyB0 8→8、noisyB1 7→8，无原成功丢失、各physical/design全通过。

但四组平均yaw peak降低.9128/.8675/.8551/.8952deg的同时，新增roll>5deg配对记录2/4/2/4=12条，只涉及0.115m的4独立case6301001/3/5/7。例cleanB0/6301001：yaw6.9315→5.7159deg，roll4.9161→5.3156deg，两轴仍不满足task。Jpsi仅降低.0538/.0484/.0391/.0604deg（约1.97–3.79%），不能把peak改善替代原指标/正式学习优势。noisyB1恢复6301031的成功是当前固定noise实现，不抵消低位roll退化或称普遍鲁棒。

**固定alpha1全任务改进分支关闭。** 依175原“其他门不退化”要求，四组资格均False；不采用为新基线、不扩alpha/gain/训练预算或晋升旧policy。保留代码/完整120noise与gyro/原40/唯一结果为有价值的耦合诊断。滤波选择确实改变结果，但不能归因为纯延迟、不能说所有滤波或后续协调无效。物理/design通过不等于多轴任务通过，失败case新增另一轴违规也不能忽略。

plot_filter_tradeoff的PNG/SVG覆盖每panel全部20case，0.115全部case红色强调、原5deg双轴界限/成对箭头显示；source/review/outputs绑定manifest，已视觉核过，不按胜例选点。0新physics/eval/learning，原score/model/RMS/CPU-GPU/final封存不改。179按实际coupled roll-yaw变化核控制链/腿参考权限，不能直接把边界clipping当全部根因或扫增益救alpha1；有据候选仍须方法区别与公平准入。180深审清理，完整贡献/资格3→正式5seed/消融/新独立泛化/统计/PPO真实成本/新稿继续未齐。

第177轮陀螺滤波120新回合执行完成（2026-10-06，目标继续）：

run_nom_yaw_filter_probe复用完整记录的checker/partial保存，冻结source/noise/installed API/runner及原40参考hash后，唯一6job×20首回合=120新评价完成，另40 original_clean源码资格复用，总160比较记录、20发展case。新首physics1785937，全contact3525318，120 fullNPZ1021385933B+120gyroNPZ37427010B，最大单份8974406B；全源/噪声/结果/240trace hash、原success重建、dense/count/finite/前后qv连续/gyro承接、float32noise与alpha递推/首终态核过。无interruption/隐式重跑/追加case/学习更新。

结果计数：latest_clean B0/B1-route均8/20；original_noisy B0=8/20、B1-route=7/20；latest_noisy两者8/20。六job physical/design各20/20。无噪声任务成功数未增加，噪声B1差1只属当前固定人工noise样本，不能就此称普遍鲁棒/新方法优势、不能弱化强classic或自动换基线。所有vs_original_clean标签变化保留，不当同条件复现漂移；178按同noise同law配对原完整门和机制细查。

worker实见PID66170，保留RemainAfterExit便于核终态：MainPID0/SubStateexited/Resultsuccess/退出0，20:28:35→20:31:52整采集单元197秒（含锁等待可能/构图/传输压缩和checks，非PPO或purephysics）。合成全记录+gyro prefix/latestbuf/error保存通过，实际无中断。原源/model/RMS/CPU-GPU/旧40与205/final封存保持，0训练。

160是20原发展case×2固定law×4conditions（40复用+120新），不是独立160case/训练seed或新独立泛化。178完整gate与配对去留，179仅据实际结果决定剩余方法/学习必要性，180深审清理保持；可辩护新贡献、强经典学习优势、资格3→正式5seed/消融、新独立泛化/统计/真实PPO端到端与完整新稿仍未齐。

第176轮隔离滤波/噪声源码资格完成（2026-10-06，目标继续）：

nom_yaw_filter_probe.py包裹冻结控制/测量调用：alpha1仅在原Nom更新前令state8等于已有最新gyro，然后仍调用原controlkernel；alpha.025/无noise不写任何原控制/传感状态。四条件各20world构造/capture核245原corecalls、40steps/4sharedref、原inputs/order及前后两次capture插入位置相同，未修改冻结controller/environment/完整录制器。

每case PCG64 seed17500000+case、float32噪声表及NumPy版本/hash冻结；排序无关/重复一致、容量越界显式拒绝。初始sample0同时初始化已有gyro和obs/history，随后只对原physics forward产生的新yawgyro样本加噪一次，供后续Nom和原after的delay0/39packet；不提前forward或用更新qvel替代，其他通道不改。Nom使用前一物理步最新可得样本的原分期保留。8列记录valid/step/Nom measurementindex/input/filtered/freshclean/noise/freshnoisy，逐步核输入承接、滤波递推/float32加噪和记录计数。

原始clean输入与原kernel control/state/diag输出exact-noop；两alpha真实原control计算符合预期、公开packet与噪声测量一致、无重复加噪/其他通道变化、inactive/overflow/owners/reset检查过。补充合成terminal/automaticreset令新sample0同时供Nom/packet、重复done不再存、显式reset清chunks过。仅构造/捕获/原控制计算与合成测量/after，无mj_step实际积分或评价回合；原force/geometry资格继承且原sourcehash不变。本批20case delay0，不能泛化为任意传感延迟噪声认证。

noise_table.npz/source/unit/supplemental/installed forward/support/proposal绑定在admission_review，original_clean40具备source/no-op复用资格，不称同次/bitwise trajectory证明。当前0新eval/learning，177须freeze唯一120runner/消费和中断保存后执行；178原完整门及配对clean/noisy检验，180深审清理。新方法、正式5seed和独立泛化/完整新稿仍缺，源码准入不当性能或论文证据已通过。

第175轮方向深审与冗余清理（2026-10-06，下一180，目标继续）：

**继续价值与方法。** 171–174完整源码/CPU力矩资格、40实际全步数据、力矩事件分解与权限/时序核对有价值，相关source/契约/分析和全部40raw哈希再次核过。已从缺记录推进到可检验控制时序，不再做通用同数据诊断。旧分配/冻结V6晋升、坐标改名创新、接触触发可预防此前冲量的表述保持关闭；后续恢复或预测策略并未被证明全部无效。当前39packet与2kHz Nom权限不同必须披露，不利用故意慢classic制造PPO优势。40=20发展case×2law，不当独立40case/训练seed；工程检查、观察相关及普通滤波改进不替代论文贡献。

**唯一有限实验准入范围。** nom_yaw_filter_probe_v1/proposal登记alpha=.025/1×yawgyro噪声std0/.02rad/s×20case×B0/B1-route的160条比较记录，其中original_clean40仅在原模型/控制/信息精确同一及no-op资格成立后复用172数据，新增预算120，训练0。同五高度/左右20mm障碍/双向.7m/s；不按成功挑样本、不扫alpha或gain。单轴人工Gaussian/2kHz/PCG64/case seed17500000+case，alpha和两law使用同一measurement-time-index序列；std为预设敏感性试验，未经硬件标定、不当全面噪声鲁棒证明。必须同一现有gyro测量同时供Nom和延迟39packet，一次加噪不累加、不提前sensor forward、不换用新qvel推导gyro，其余通道和schema时序保持；clean/noisy输入及filtered gyro逐步记录。

干预只改Nom yaw gyro处理，所有gains/VMC/LQR/模型/任务/奖励/commands/原physical/design/执行器限制保持。噪声条件有意改变gyro数值但alpha比较信息一致，Fn/COM/geometry继续仅eval。假设部分早期yaw overshoot与原19.75ms低通有关；alpha1也改变带宽/噪声抑制，成功不能归因纯delay或宣称新算法。主验收仍是完整task与逐case强classic不退化，另列roll/pitch/yaw/height/speed/arrival/stop/tail/contact/exit/physical/design；只yaw好看但原success丢失或其他门恶化不晋升。复用不合格即停，不暗增原始回放；失败/中断保存并计已耗预算，不自动重跑/调alpha/PPO/换基线/旧模型晋升。

176隔离最小gyro/noise/observer源码资格及0干预identity、共享测量时标/packet、owners/reset/force/原图契约；177冻结runner/噪声表/API与预算后唯一120队列；178按原完整门检验固定变体及配对clean/noisy效果；179仅据结果决定剩余控制/信息/方法和学习必要性，不盲续失败候选；180深审清理。目前仅范围注册、source未准入/未执行，不称120结果已完成或新独立泛化。

**清理与全目标。** 删除analyze_complete_contact与complete_contact_witness两份Git忽略且未使用的源码等价pyc，共20237B，cache/sourceSHA保留round175_cleanup；再生缓存不称永久节约。全部源码/完整40与205轨迹/model/RMS/unique失败/CPU-GPUbaseline/final封存保留。round175_direction_review绑定本周期与newproposal。具体可辩护新方法、稳定同信息强classic学习优势、资格3→正式5独立训练seed/消融、新独立ID/组合/几何/参数-delay OOD/易任务、统计/真实PPO端到端及完整新稿仍未齐，不能以本实验或负工程材料结束整体目标。

第174轮入口候选权限/时序与近邻核对（2026-10-06，目标继续）：

audit_impact_authority复用40完整记录和173力矩函数，核实际yaw config=.4/2/.24/.3、每physics请求slew=.01、50Hz策略与2kHz物理步，next-boundary/渐增边界断言通过。乐观假设接触一出现就能在下一策略边界完美识别（实际39packet并无Fn真值），.115/.16的16条记录中，固定接触后100ms abs normal yaw impulse有39.27–80.56%在首可用contact-triggered action之前发生，中位74.47%；所有场景等待4–20ms。共同轮残差每轮最多.3Nm、从0到满值要100个substeps（首命令到第100命令49.5ms），原lambda/共同差动竞争可进一步压缩。命令积分不等于实际接触力/减速/航向改善，不能据此宣称所有policy无效；但接触触发动作不能预防此前已发生的冲量。固定100ms为探索诊断，不换原primary或用它事后救分。

共同/差动坐标满足tauL=c+d,tauR=c-d，允许域|c|+|d|<=.3Nm；441点变换逆/diamond恒等过。该能力已在virtual6内，重新命名/删腿减少维度不构成独立新算法。官方PMLR动作空间摘要复核轮腿、初始化和步间执行已被系统研究；作者T-RO页面本轮只读元数据确认已有impact/contact-switch主题，不用该页宣称方法完全同构，也不重试用户已要求停止追索的IEEE全文。近邻核对非穷尽排他性证明。

**去留。** 关闭“共同/差动坐标变化就是新贡献”与“接触触发策略能预防首个尚未观察冲击”的表述；后续共同动作可能改善恢复的假设未被反证，但没有具体方法区别和受控效果，不准入新PPO/五seed。175深审建议先评估一次共同Nom时序干预：原Nom在2kHz可得最新可得gyro，却对yaw gyro用alpha=.025（e-fold19.75ms）；仅与alpha=1最新可得gyro比较，增益/VMC/LQR/任务/模型/所有物理设计与输出门/信息保持，同一强经典参考。其带宽及噪声抑制同时改变，不是纯delay-only因果，也不是新算法；须查高位roll、速度/到达/停车/高度及噪声敏感性，成功不自动替换原基线或晋升旧模型。

本轮仅形成175待评候选，0新physics/learning预算，无控制源码改变/新回放；禁止以慢classic构造PPO优势。175方向深审和冗余清理如期，具体新方法、资格3→正式5训练seed/消融/新独立泛化/统计/真实PPO耗时/新稿仍待完成。

第173轮完整接触航向力矩解释与候选收束（2026-10-06，目标继续）：

analyze_complete_contact按solverpre同时间COM/frame/local6wrench计算每条wheel-static接触的tau_z=((p-COM)×signed(force@frame)+signed(torque@frame))_z，再逐physicsstep累加；normal-only只用Fn×normal，remainder包括切向力及全部contacttorque。5项符号/旋转/无事件断言通过，40原任务记录/trace及所有源码hash核过，full=normal+remainder数值恒等核过，0新回放/学习。原24失败为12独立case配对重复，其中20条（10case）yaw>5度，全部在firstpositive targetcontact后82–107.5ms发生，8条无referenceclipping。

20条从firstcontact到firstyaw5的signed normal/remainder impulse方向均相反；.115/.16的16条记录abs signed normal impulse约.318–.352Nms，remainder约.173–.214Nms。这是记录力矩的分解与事件先后，不能将remainder当全部外生摩擦扰动，更不能推论去掉它会改善。它包含当前控制器牵引/恢复作用，不能靠normal-only或整段signed cancellation推因果/yaw加速度。该终点依赖失败的window为探索诊断，后续干预指标必须事前独立固定。

plot_complete_contact导出PNG/SVG并视觉核过：确定选0.16m最小注册seed6301009（169已用于无截断反例），B0/B1-route的法向入口负脉冲、反向牵引部分及随后yaw跨−5度相似；仅作说明，不当独立优势。16条原成功中8条central存在零upward targetnormal样本，不能把保守零卸载规则升级为primary，也不能由零targetnormal推腾空。完整centre/wholetyre/topload/airborne主张仍需相应条件，原success不改。

**唯一待准入候选。** 转向入口冲击阶段的common/differential wheel协调：在同Nom、同轮差模允许范围、原总执行器限制及公开延迟39packet下，增加共同轮权限是否能降低early normal yaw impulse/peak，同时保持原速度/到达/停车/高度/roll/physical/design及全部强classic成功。真值contact/COM/geometry不进policy/controller，不作侧缘路径等同完整passage。需同信息非学习共同制动参考和matched训练/动作权限/初始化对照；冻结V6删channel不能识别matched learning因果。共同/差模表示与简单制动是常规工程，不宣称新算法；174先核近邻区别与权限/可行性，有据且有可辩护贡献才登记学习，否则关闭此候选换问题，不扩预算救分。当前该候选新增physics/training预算均0，175深审清理保持；完整贡献/正式5seed/消融/新独立泛化/统计成本/新稿仍待完成。

第172轮完整40回合测量完成（2026-10-06，目标继续）：

complete_contact_witness.py在源码/installed API/runner/旧参考结果/预算hash核对后执行唯一40首回合队列，B0与B1-route各20原受控开发场景。分别297653/297661首回合物理步、587304/586733全接触记录，总595314步与1174037条接触；40份完整NPZ压缩331907245字节，最大单份8518895字节。每步完整pre/post状态、全contact/count/COM和原112header落盘，校验全部SHA/schema/finite/dense/nooverflow/逐step接触计数/首终态/相邻pre-post状态完全连接。两classic各8/20原任务成功、20/20physical及design通过，与原标签变化0；不称轨迹bitwise等同。

执行前显式修正跨job构造失败时的旧raw句柄误标风险，0预算消耗，initial runner contract/日志和可反向重建源hash的amendment保留；canonical JSON避免原tuple/list账务故障。合成中断保存检查通过，prefix与最新buffer分别保留、completed world不误标、已耗物理步保留；实际队列无interruption/追加回合/学习更新。unit实见PID56863运行后已GC/无live handle，completion与progress均complete；journal记录19:41:46→19:42:50约64秒整采集单元耗时（含锁等待/构图/传输压缩与校验，不是PPO或纯物理吞吐）。GC后exit status不保留，不将缺省状态当直接退出码证据。

这40条是20原开发case×2固定law，不是40独立场景或训练seed。完整数据可用不等于完整通过/安全/新方法优势已证明；原模型/CPU-GPU基线、205旧证据、失败、原判据及最终封存保持。173用本数据解释真实路径/完整solver接触wrench和机制/任务可证伪候选，作去留而不再增加同类回放；174只对有据候选核预测、公平权限/信息和近邻区别；175深审清理。全部新贡献、3资格→5正式训练seed/消融/新独立泛化/统计/PPO真实成本/完整稿件仍待完成。

第171轮完整状态/接触录制器源码资格（2026-10-06，目标继续）：

complete_contact_recorder.py复用冻结v1的pre/post/header/contact helpers，使用无范围裁剪窗口，新增仅写自有数组的完整pre/post qpos/qvel、六路command/actual force、全局接触表和每world计数/COM。接触表包含geom/world/dist/dim/position/完整frame/5friction/6local force与torque，保留零力接触和无接触状态，每physics步记录；capacity/overflow、计数、有限/步序/首终态检查不允许截断当完整。旧源码与205结果不修改；真值不进入policy、critic或controller。

diff3/virtual6各20world构造/capture/reset资格通过：245原core calls、40mjw.step、4shared-reference更新，原数组输入指针/标量、调用顺序、受保护状态不变，所有记录buffer由env持有。每world轮心CPU mj_forward一致；pre/post全状态复制、远离场景仍保留每步、合成首终态冻结/再次done不重复/显式reset通过。两cone×condim3/4/6×physical/synthetic的12配置、各两geom顺序，全部6力/力矩与CPU mj_contactForce一致，切向和扭转/滚动分量及world方向有非零夹具；零载荷接触保留、inactive不记录、漏contact/overflow/stepgap拒绝。额外检查3contact在2world分配、容量超限与overflowbit、inactive、20world静态COM/CPU一致；不将纯forward/合成检查当物理回放。

当前installed implicitfast流程先forward再integrate，接触/frame/force/subtree COM属于solver pre层，q/v分别存积分前后，local force/torque按local@frame转world，作用geom1/对geom0反号。installed forward/support与全部源码/契约hash留admission_review。首次通过版本在first_admission保留；随后仅新增未完成chunks/frozen可读入口，供执行中断保存已耗回合，重新完整资格通过，额外测试kernel AST不变。

本轮0实际评价回合、0学习更新。source admitted不能证明40回合数据已完整或新方法有效；172须先freeze唯一runner、预算及partial/failure保存，然后才执行40首回合并核对逐步完整性、原gate差异。主方法、3→5训练seed/独立泛化与完整新稿仍缺，175深审清理保持。

第170轮方向深审与冗余清理（2026-10-06，下一175，目标继续）：

**方向去留。** 166–169复核真实车轮椭球几何、原contact与严格通过的差别、缺证据三态及参考截断反例有价值，但同205数据的通用离线诊断在此收束。旧轮分配、冻结V6整体晋升及“截断解释全部经典失败”分支保持关闭；不追加旧学习预算或用更极端场景掩盖无合格贡献。边界参考截断的局部作用尚未测，不能由反例宣称完全无效。

**方法评估。** 保留原基线/失败/指标、配对开发场景、信息与时序核对、事前有限预算和未知不算通过的方法正确。零允许卸载是保守二级诊断，不是已采纳通用地形任务；不能据其125失败/80未知改写原132成功，也不能把几何投影和最大法向接触当完整载荷或腾空证明。统计重复和工程验证不能替代新方法贡献。

**必要补充仅一次完整测量。** complete_contact_witness_v2/proposal.json预登记40个首回合：5高度×左右20mm障碍×正反0.7m/s的20个原开发场景×B0/B1-route两固定控制器，不含学习。新录制器需每物理步完整pre/post qpos/qvel、完整未截断接触表（frame/全部局部force与torque/geom/world/dist/friction）、对应COM/geometry及实际执行量。原112字段缺完整solver姿态、较小并发contact和切向力，无法用再分析补齐。新真值只供评价；原模型、控制调用顺序、共享参考、奖励、原task/design/physical门不改。先做CPU前向几何/接触力、完整多接触、时间层、owners/终态与graph输入资格，再冻结source/runner；当前未准入源码、未执行，不称40条已完成或独立泛化。

具体机械测量假设：完整轮地接触wrench及其航向力矩可能区分外部扰动与参考截断，比较全力和normal-only力矩及时间顺序；观察相关性不构成因果或补偿性能证明，不凭本假设直接PPO。171隔离最小记录器资格；172仅准入后执行唯一40预算（失败保存和显式计已耗预算，不暗重跑）；173作实际任务/接触机制去留；174仅对有据候选给区别性预测、公平信息/权限和近邻文献核对，未支持则换候选不续旧学习；175深审清理。所有高度/镜像/方向完整覆盖，样本不按已知成功挑选。

**清理与完整目标。** 删除2份确认源码codeobject等价、Git忽略且无打开句柄的pyc，共9824字节，可再生缓存不称永久节约。全部源码、205 raw、模型/RMS、独有失败、CPU/GPU与封存测试保留。round170_direction_review/cleanup绑定证据和新proposal。具体贡献、稳定强经典学习优势、3资格→正式5训练seed、消融、新独立ID/组合/几何/参数-delay OOD/易任务、统计与真实PPO耗时、完整新稿仍待完成；不以本轮深审结束整体目标。

第169轮共同参考候选反证（2026-10-06，目标继续）：

实际controller先将roll offset限制±35mm，再限制到对称reference room。0.16m room=45.2955mm、0.24m=125.2955mm、0.30m=80mm，因此这些高度不会被后一项再次截断。audit_reference_clamp_hypothesis核对163原分析及全部输入哈希、159依赖源码并复用80条B0/B1-route受控记录，两个合成断言通过：0.16m的4个独立场景在两经典控制器下均失败且无该参考截断（8条配对记录）。因此“所有受控经典失败均必须经过roll-reference截断”的统一候选被反证，不准据此直接commonheight修复或开始新PPO。

0.115m及0.38m各经典控制器同时有4成功、4失败且都有采样截断；截断也不是充分失败判据。这不证明边界局部干预无效，不排除其他限制，不从观察相关性推断因果效应，也不将混合密度sample fraction解释为时间占比。原成功与完整通过证据分开，0新增物理回放/训练，原模型、任务、指标及最终保留集不变。

170按约深审与冗余清理：需要明确接下来的工作如何打破“只有诊断、尚无合格贡献”的瓶颈；比较补齐通过测量和一个可证伪的任务/控制候选的价值，先冻结有效任务、可得信息及有限预算。不存在合格候选时不盲目学习，但继续推进核心论文所需完整实验链，不把本反证报告当作完成目标。

第168轮既有数据三态证据应用（2026-10-06，目标继续）：

复用第167轮规则，apply_passage_evidence.py 对第164轮205条记录逐条检查，绑定源代码、路径草案与并列证据契约的SHA256。5项可运行断言通过，原任务132/205成功标签保持不变，0新增物理回放、0训练更新。ground_rollover 与 airborne_passage 各有125条已观察必要条件不满足、80条证据不足、0条完整证据通过。两项均为独立保守诊断，不是原控制器的新成绩；缺少完整几何、完整接触或起落证据必须保持未知。

这205条仅覆盖41个开发场景（2个固定经典控制器与3个已训练seed），不是205个独立场景，也不能替代新独立泛化测试。当前记录不足以支持完整轮道通过、持续顶面支撑或跳跃越障的论文主张。下一169只选择有证据且可证伪的具体候选，先冻结公平信息、任务和预算；不自动启动PPO、不事后改阈值救分。170按约深审与冗余清理。主方法贡献、强经典对照优势、3→5独立训练seed、消融、新独立泛化、统计及真实端到端耗时、完整新稿仍未完成。

第167轮并列操作定义与三态证据规则（2026-10-06，goal active）：

passage_evidence_contract.py/parallel_passage_evidence_contract将original完整contact task保留为原权威分数，并定义strictsecondary ground_rollover/airborne_passage，不替换primary或用silence选题。纯三态aggregator8项positive/missing/false/invalid fixture过：已测False则该独立定义失败；无False但缺证据None则insufficient；全部True仅supported_at_recorded_resolution，不是continuous safety定理。未经测量的数值true不接受，不以None算pass。

两候选共同必要项：original完整task/physical/design、指定路径、按事件顺序经过、同时间层完整车轮geometry/clearance及充分sampling。ground例要求完整每目标contact记录、每个已记录central样本positive upward top normal support且无airborne central interval；这是零允许记录卸载的保守continuous-observed诊断，并非已采纳general terrain任务，自然bounce可能fail。若容许support duty/gap，必须另预登记，无policy score救分。air例则documenttakeoff/wholeobstacleclearance/landing/完整airinterval pose，不能当ground support。1e−6N只沿用已有噪声区分，不新增或调load fraction/duration/performance阈值，时空tolerance以后连evaluator冻结。

当前112含maxwitness且缺pre完整orientation，mixed event映射未实现，不能使全部ground/air必要项True。此轮不往205已有结果写newsuccess，原failed methods/data/OldCPU-GPUbaseline/final不变，0newrollout/learning。任务偏好仍pending，当前originalcontact研究和strictsecondary定义并行；不称这已构成新训练协议或MethodAdvantage。

168用边界fixture及现205按三态apply只出资格/缺项（旧分数另列），169据资格挑一个可检验候选并freeze fair信息/预算，不开始不合格定义dependentPPO或Nomrepair；170深审清理。整套具体新method贡献/strongclassical学习优势/3→5seed/独立泛化/统计/cost/完整新稿继续active且缺项。

第166轮整轮碰撞几何与证据完整性（2026-10-06，goal active）：

audit_wheel_geometry_evidence.py在原frozen source/proposal对应41case模型上只CPU编译，不mj_step/新rollout/训练。82个wheel_collide_L/R均为ellipsoid（不是sphere/cylinder），halfaxes=[.05,.0275,.05]m，全宽55mm；geom local pos0/quat identity、body链静态quat identity且铰链轴localY、root free body核过。场景质量/drive等不改该geometry。源/contract/hash与compiled参数保留，support函数6项basic/boundary/invalid检查通过。

ellipsoid方向support rho(n)=norm(diag(radii)*R.T*n)。x/z等半轴使localY所有链/轮spin角不改外形，post body Euler可据此推导采样post orientation的支撑投影；不能把post orientation挪到pre-solver force时刻。在中性方向，legacy半道35mm+车轮半宽27.5mm=62.5mm，wheel centre偏55–60mm仍可能有侧缘overlap，与先前positive target接触并不矛盾；projection overlap只是可能性，不是实际接触或完整通过证明。1e−7几何误差/float32/离散时间范围须明确，未给continuous certificate。

现有112列证据checklist：post整轮投影可带rounding假设推导；solver-time exact轮orientation未完整记录（只有pre centre，无pre完整bodyquat/jointstate）；positive target normal/vertical witness有但非friction-inclusive totalforce/非top完整cert；perface/pergeom support duty不足（最大witness省较小并发contact，vertical witness没有单独dist）；mixed event/ground-airborne完整规则未冻结。原task/physical/design/success labels原样保留，当前complete_loaded资格仍unknown。

167写并列操作定义及缺项处理，不先再跑205/更换callback/model、改变旧score或用support断言成功，也不开始PPO/Nomrepair；原primarycontact研究和strictsecondary资格并存、用户taskchoice仍pending。全部具体方法/5seed/消融/独立泛化/cost/新稿goal active，170深审清理。

第165轮方向深审与清理（2026-10-06，下一170，goal active）：

**继续价值。** 161–164的只读接口资格、205实际轨迹与解释、独立necessarygeometry暴露了原contact判据与完整轮道通过主张的差别，推进数据解释与公平协议有价值。旧coupling/component分配微收益及frozenV6晋升保持closed，不追加相同回放、gain/weight/reward/network或PPO预算。13低高度V6成功证据不当新centre/loaded可达性、channel必要或整体优势；oldsuccess仍按原task有效。新完整主张/learning superiority/独立泛化仍未完成，不能缩成负材料或工程就绪结束全goal。

**方法评价。** 245原调用和pointer/scalar保持、owners/force API/firstterminal检查和冻结source/RMS/完整失败保留有效；41+164因tuple-JSONpostwrite比较故障恢复已明确、total205不增加、B0未重跑，原failed unit和rawerror保留。现有validator不等外部完整physics重建；112列max force/vertical witness可能省略小的并发接触，geometrycentre不是完整轮胎，pre/post时间层不能混用。legacynecessarygeometry实测B0/B1route原28各28满足、V6原28/28/20中13/4/7满足，不作为final loaded task成功率。

**下一必要工作，而非新训练。** 166从不变模型提取actual wheel collision shape/size/tilt信息，并将已有112字段对应到完整通过证据checklist，不新rollout/改controller。167并列保留originalcontact与candidateground/airborne passage操作定义，假设和未定量明确，不能按policy分数选support阈值。168边界fixture/现205做完整资格逻辑校对，原成绩另列且mixed/缺证据显式unknown。169只能在资格成立后选一个具体有预测的共同reference/authority/taskdesign假设，给classic/RL同信息同task、source/预算事前冻结；没有资格则不新learn。170深审清理。没有本轮新physics/training预算，用户严格lane或edge可接受选择仍pending，不把未回答当primary任务更改或开始依赖训练。

**清理。** 2份ignored/fuser无人使用、完整compiledcodeobject及instruction bytes与源码一致的pyc删除28004B，round165_cleanup保存源/cacheSHA，缓存可再生。所有源码、models/RMS、205rawtraces/geometry、unique失败与recovery/基准/final保留；不会将当前学术失败资料当冗余删除。

round165_direction_review绑定161–164 source/admission/实际completion/recovery/分析与draft证据（含所有trace hashes复核）。整套具体方法贡献、stable强classic学习优势、条件3→正式5seed/关键消融、新独立ID/组合/geometry/参数-delay OOD与易任务、真实端到端cost/新稿/复现仍缺，goal active。

第164轮独立路径资格草案（2026-10-06，goal active；用户任务选择pending）：

按论文可能主张“指定轮道通过”的假设，新增path_qualification_draft.py与lane_centred_passage_contract_draft.json，未改原contact success/任何原failed gate或控制器。scope仅legacy单箱体/指定wheel centre；实际geometry局部坐标与方向、entry-before/exit-after、inside pre/post lateral界、interior连续physics采样共同作为必要几何条件，geometry epsilon1e−7m是浮点标度不是性能调参。5项synthetic检查覆盖正向/反向、边缘、缺密度、loaded资格保持None；数据/模型未新增rollout或更新。

复核现有205trace/result/geometry全部hash并另报必要条件：B0/B1-route原success各28且28均geometry prerequisite过；V6-1761原28其中13过、1762原28其中4过、1763原20其中7过。不要把这些写成新完整任务成功率或新策略比较；不含顶面loaded/整轮外形/continuoustime保证，mixed per-event映射未实现，complete_loaded_rollover_qualified始终None。当前草案不是“更严格指标已正式通过”或真实rollover完整定义。

完整新资格仍需明确centred ground rollover/edge contact/airborne scope、轮外形/倾角/间隙与匹配solver time顶面载荷证据、support coverage/最长卸载间隙及混合event progression；不根据哪个policy能过选择load/duration阈值。以后对classical/RL使用同声明任务，分别保存旧原判据，fresh发展与独立场景、3seed资格后5seed/ablation/OOD/成本/新稿仍是全部目标。

已询问用户任务选择但无回答；当前仅起草与必要几何校对，不把推荐项视为已提交答案，不开始依赖定义的新训练/默认Nom修复/切换任务。165按约深审清理时明确继续价值、完整证据及范围决策，原baseline/final保护、goal active。

第163轮实际路径接触解释与主张资格（2026-10-06，goal active）：

analyze_task_mode_witness.py核205trace/result/geometry SHA与原load_rows，4点synthetic centre/边缘/部分路径分类unit过；无新eval或learning。Descriptor仅描述真实wheel centre相对compiled box投影，post-pose与pre-solver loaded witness分开，不把中心离开投影直接当整个轮胎绕过。multiple-geometry mixed case不强套legacy单box类别；normal/vertical-normal分量不等friction-inclusive总反力。

原低高度V6-only8案例，在3seed共13成功轨迹中：9条目标侧轮centre在整个已采样box纵向区间都位于70mm宽投影之外，4条仅部分在内，0条全在内；所有13有正目标vertical-normal接触，5条在障碍中央半段也有positive max-vertical target witness，另8条没有。故不能一概称纯零载荷擦碰，也不能称完整沿指定轮道加载越障或凭centre断言整个轮胎绕过/作弊。经典B0/B1route示例case6301001轮中心较多落在box投影且有持续约35N目标normal支撑，而V6中心走侧缘并主要在入口有脉冲；这是路径与接触方式区别，不证明哪个channel或Nom高度room唯一根因。

plot_task_mode_witness.py已生成该.115/20mm case全部5controllers真实路径+目标vertical-normal图PNG/SVG，已视检布局；caption明确单例说明、pre/post时间层和旧success不改。physical_mode_analysis覆盖全部205（同41dev cases），中段证据及每case统计完整保留，figure_manifest/rawtrace/source/closure SHA绑定。采样频率混合的Nom/reference统计只作sampled描述，不当统一时间权重/causal。

**必须作出的取舍。** 8低高度见证保留为原非对称接触判据下的实际成功，但撤回把它们当作全部真实居中完整traversal可达性/低维authority不足必要证据的资格；更不晋升V6整体方法（旧优势门仍failed）。不依据这些winner构造oracle、直接Nom/commonheight修复/扫动作channel或新PPO。若论文要主张真实沿轮道越障，必须先单独定义路径/geometry/loaded-or-airborne passage资格，并给经典与RL同任务；若允许侧缘接触，则明确论文只主张对应contact task并另找稳定整体收益，不能改标题挽救已失败门。

已通过async问题询问用户的预期障碍任务：严格指定轮道或允许侧缘不对称接触；尚未收到回答。164可先拟独立严格passage qualification（不更改旧原判据或启动依赖的新训练），预期选择未确定须标注假设。该项比继续微小wheel allocation、更大network/budget或极端terrain优先。165按约深审清理，整套主方法/正式5seed/新独立泛化/成本/完整新稿goal继续未完成。

第162轮实际方式数据闭合（2026-10-06，goal active）：

task_mode_witness.py先冻结runner_contract并核recorder/source/unit/supplemental/API与3models/RMS，初unit PID35277完成B0全部41回合并落盘后，在in-memory row与JSON row直接比较处失败（Native yaw_config tuple变JSON list）。原load_rows全任务重建在该assert前已经通过；故障/interruption/原run.log保持。独立调用原checker审完整B0原task、41traces/geometry/hash/时刻/计数确认全完整、消耗预算已知41；不是物理/recording接口失败，也没有把未保存状态当可精确恢复。

公开recovery_contract登记执行分段41+164；task_mode_witness_remaining只运行B1-route及全部3V6，JSON规范化后复用原检查，未改原runner/recorder/控制、模型、RMS、case/阈值/任务。原failed unit保留exit1，newremaining unit初PID36194正常exit0/inactive/MainPID0。单进程不间断计划未实现，明确记录分段偏离，不重跑B0、不增205预算、无silent retry或新PPO。

全部5jobs205首回合、41独有发展case，共记录3043633首回合physics steps（非autoreset全部仿真工作量）、615184采样rows，205压缩trace约153MB；geometry/runtime/result/trace字段和SHA保留。任务/physical/design按原load_rows核，trace列112、所有valid/finite、时刻和增长step/count、真实geometry窗口50Hz/2kHz取样条件再核过。3model权重/更新计数在eval前后同、savedRMS mean/var/count保持。旧存档→重放success变化0；这不等于全trajectory bitwise身份或完整外部物理重建。

round162_data_review绑定completion/原runner/recovery/recorder与真实两个unit终态，原tuple/list比较失败及确认恢复数据保留。163基于已落盘205原判据数据解释loaded/边缘路径/腾空或证据不足及Nom/reference/authority关联，必须决定是否需具体下一假设；不从同success标签先宣称真实climb/学习优势。原failed gates/final/CPU-GPU基准保护，完整新贡献/5seed/消融/新独立泛化/成本/新稿goal仍active，165深审清理。

第161轮真实路径记录器准入（2026-10-06，goal active，205队列尚未run）：

task_mode_recorder.py保留当前Native的diag与全部原方法调用，通过重新capture加入只读pre/post/contact节点；原CUDA图句柄会更换，原核心调用序列和参数必须逐项保持。两个Native模式diff3/virtual6、各41world已构图/reset核245原核心调用及参数pointer/scalar完全一致，其中40mjw.step/4shared-reference updates；没有使用旧RecordedEnv，没有删除shared-reference或缩小diag。两种模式capture与只读核未改protected model/control/state/reference/target数组，trace/mask/count/window由env显式持有。

112列记录区分pre/post轮中心及solver接触层，实际passive-chain wheel_center与CPU forward位置对照过；50Hz全程与geometryx窗口±.20m2kHz采样，远侧移和零转矩不阻断窗口采样。first-terminal冻结、再次terminal不覆盖、explicit reset清stats/mask已用synthetic状态验证，无实际integration/评价。contacts按world扫描保留candidate count、normal load、垂直normal分量及最大normal/vertical目标接触点/normal/material/dim/dist/slip，legacy bump与额外terrain都列入。normal support分量不是包含friction的总反力；pre solver contact/force与post移动轮pose时间层分开。

初force unit使用int32 array而接口需vec2i，未通过准入且没有run；失败日志及修复前源保留。修正synthetic fixture类型后CPU mj_forward/mj_contactForce与GPU提取对照通过，补test_task_mode_recorder.py覆盖elliptic/pyramidal×condim3/4/6×两种geom order共12组合，当前模型cone0/geom condim3被覆盖；输入不改与采样规则通过。source_contract冻结runtime/原parent源和installed contact API，admission_review另绑定supplemental source/result，physicalrollouts/evaluations/learning均0；构图/CPUforward计算不作零计算成本声称。

162只在再次核source/proposal/admission/unit/模型/RMS/接口hash后执行已登记205首回合唯一队列。Generic evaluator沿原RouteState39与savedNormalization，日志真值不进入policy；运行时再核model/更新计数与RMS不变、geometry/trace/原完整task与counts及所有replay变化。163必须解释真实方式并作下一必要干预去留，165深审清理。工程准入不当205数据已产生、loaded越障、独立泛化、方法优势或核心录用证明；完整六类证据目标仍未达成。

第160轮方向深审、数据解释补验与清理（2026-10-06，下一165，goal active）：

**方向是否值得继续。** 156–159完成2448首回合GPU评价（同136发展case重复，记录36496976首回合子步，非全部autoreset仿真工作量）、0新学习，两个unit正常终态且payload/source/unit/model/RMS契约和原门核过。当前hip-coupling恢复及component-wheel小幅收益分支明确closed，不追加gain/weight/budget或五seed。学习方向仍有8低高度V6原判据见证，但整体强优势失败、5案例无已观察成功；这些支持进一步验证实际控制方式，不支持casewise oracle、已完成创新或直接长PPO。

**方法是否得当。** 事前有限预算/原门/失败保留/显式buffer持有/接受量和时间层分清的方法保持。原判据的reduce_contacts只按几何候选设置legacy/terrain位，没有normal force正值/顶面载荷或完整轮路径条件；legacy compiled bump盒size=[.25,.035,height/2]，宽70mm。已有success只能按原判据解释，尚不能由单个触碰位断言完整加载越障，也不能未经数据断言绕边/跳过。Nom边界roll-reference room已实读：.115≈.296mm、.38=0，不给动态失败直接定因。旧trace_failure_chain.RecordedEnv会把diag重建为21维并重新捕获图，不含当前shared-reference更新；不能直接用于当前38维/49源基线。仅复用其纯接触力函数/速度函数并核当前接口，不复用旧ownerless采集器。

**补什么实验及为何。** 只登记task_mode_witness_v1：完整controlled40（含easy中高度、两侧双向/10与20mm）+唯一未解决regular mixed6300092=41已见case，全部3个V6末200k frozen models及B0/B1-route两个固定经典，共205首回合、0training，目前source未准入/0执行。选择按整个controlled组及唯一未解决mixed公开规则，不选赢家seed/丢失败；41×5不是205独立case、5控制器不是5训练seed。

记录实际passive-chain轮中心、真实geometry/candidate contact/positive solver normal force及其时间层、路径/飞行/间隙证据，另记固定height/desired-clipped roll reference、raw和accepted Nom wheel common/diff、filtered/accepted/actual request。完整episode50Hz，目标geometry纵向范围±.20m内2kHz，按geometry/x/time取样，不按force/y/success裁剪绕边轨迹。当前true position/force/terrain只作eval，绝不进Actor/Critic/controller。保留原Native图/diag/shared-reference/动作/奖励/观测，不改原contact success；source/geometry/force接口/只读输入/owners/terminal/RMS资格后唯一队列，所有old-to-replay状态变化都保留，不称跨GPU bitwise。

163必须就实际方式和数据充分性作去留：原成功可能是加载经过/边缘路径/腾空或混合、也可能证据不足，均不自动判作弊或修改旧success。只有几何/路径/载荷证据支持相应主张；若需更严格的traversal任务，另行明确protocol并保留旧结果。再按所得事实选择一个共同reference可行性或受控authority假设，不自动Nom/commonheight修复或newPPO；不足/接口不合格就停止本诊断，无样本/阈值/预算扫描。161最小source准入、162执行、163决定、164有依据的具体后续、165深审清理。

**清理。** 删除3份git忽略/fuser无人使用且完整code-object与instruction byte和源码编译一致的pyc，共26513B。初轮marshal序列化字节检查失败、删除前改用完整代码等价验收，不因此认定源/缓存损坏。round160_cleanup保全部source/cache SHA，缓存可再生不称永久节省。src/models/RMS/raw trajectories/独有失败/provenance/封存与CPU-GPU基准全部保留。

round160_direction_review绑定156/158closures与reviews、159物理参考/任务审计、原学习门和新proposal/cleanup；全核心纯仿真目标仍active：具体新贡献、稳定强经典学习优势、正式5seed/关键消融、新独立ID/组合/参数-延迟OOD/易任务、端到端成本与完整新稿仍缺，不以负材料或新的诊断数量替代完成。

第159轮共享任务可改善空间与参考约束核对（2026-10-06，goal active）：

audit_shared_task_opportunity.py复用原flags与load_rows，对主study/zero/classic和逐轮研究全部60panels4080评价按同136case/scenario重建success及失败flag，不采新评价/学习；所有source/input SHA核过。四类分类unit通过，Nom实际5world构图/reset取参考数组，环境记录计数/time保持0；不把构图内部CPU名义设计求解计入或声称总计算为0。

**可改善任务仍存在。** classical原B0/B1/B1-route成功并集123/136；其余有8个原判据成功见证全来自V6-main，case6301001/3/5/7（.115m）和6301009/11/13/15（.16m），皆20mm单侧legacy，左右/双向都有。6D策略虽整体差，却在这些原门案例实际完成过；因此不能把全部剩余问题判为纯不可达或原任务已无学习空间，也不能因此证明更多维度/某一channel是唯一解。case-wise见证集合不等于一个可执行切换oracle，8例不是新独立测试或方法优势；V6仍在易案例丢成功，当前性能门继续失败。

**仍无已观察成功的5例。** 6300092为约.150732m mixed；6301033/35/37/39为.38m单侧20mm controlled。所有被检查的主学习/经典/当前分配干预均未通过这些原完整任务门（不是所有可能控制器/预算或不可达证明）。.38四例主要legacy roll超5deg，其中两例yaw也超5deg；不能从重叠flag推唯一因果。既有V6低高度成功也只证明原成功判据，当前contact bit/退出判据不是完整逐步路径、加载支撑或越障方式证明，不先断言实际绕边/跳过/真实爬越。

**实际Nom参考。** 5height源码/运行数组ref[2:7]核表：lower_reference=.1147044660616607m、upper=.38m，symmetric room=max(0,min(h-lower,upper-h))，所有roll-room clamp启用。h=.115时room=.00029553393833930963m（约.296mm），.16=.0452955m、.24=.1252955m、.30=.08m、.38=0。原roll desired offset=.3*roll+.12*filtered gyro_x并先限±.035m，再受该room截断；Nom left/right target围绕固定mean reference作差动，边界参考确实受限。这是内部setpoint事实，不是实际两腿长度恒等、动态可行域、失败根因或修复后的性能证明；实际joint/contact/force与迟滞仍有影响。

**160方向审查的具体依据。** 停止当前轮clip和腿轮耦合收益分支继续回放/增budget。优先判断一次有限的真实wheel路径/接触/支撑与参考饱和检验是否必要，以区分原门成功方式、sharedNom边界reference限制和动作权限作用；再决定是否注册共同参考可行性或受控authority试验。当前不改原success/物理/设计门、不默认更难terrain、newnetwork/reward/PPO或直接推commonheight修复。全投稿目标的主贡献、合格三seed→正式五seed、新独立泛化/压力/成本与新稿仍保留，160按约深审清理。

第158轮逐轮GPU干预闭合（2026-10-06，goal active）：

gpu_wheel_component.py仅可选实验selector：原controller→leg-room→wheel selector→physics，mode0不改command/diag，mode1只更新wheel ctrl/accepted diag；原hips/Nom计算/filter memory/input与invalid保护保持。新wheel accepted值由原reward/smooth penalty/历史obs和物理检查消费；old global lambda字段保持原统计，parent steering的scalar identity deviation在component下预期非零，不能误当新比例/错误。追踪数组显式env持有，已终态world仍应用所选控制但停止首回合记录，explicit reset重置stats/mask。

128×2 syntheticGPU与NumPycommand/diag/energy一致、原模式逐值身份、zero/nonbinding float32身份、invalid拒绝可选投影保留原保护输出、其余输入不改、异步terminal/reset通过；136world实际construct/reset/graphorder/owner检查0积分过。source_contract冻结49源/proposal/unit/owner SHA后仅一次flock排他wheelleg-wheel-component-v1.service（初PID17524），24jobs1632首回合评价、记录24331229物理子步（非vector自动重置后的全部仿真工作量）、0learning/model/RMS更新，unit正常exit0/inactive/MainPID0，无中断/重试。

review_wheel_component.py复用既有load_rows核全任务、physics/design、case/scenario/summary与所有SHA，独立检查新记录的接受语义、浮点box、共同/差模正交能量、旧接受量/原lambda/执行量与parent字段对应；24panel physical/design各96/40全过、记录qualified全过。pure component比concurrent pure uniform成功数各seed/panel均不变，Jpsi regular分别降.001365/.002137/.001774°，controlled降.004755/.009651/.005249°；共同增量RMS约.002996..008926Nm。相对concurrent uniform无注册非退化失败，不能把强B0比较差异唯一归因于投影。

development参考门两panel均False：regular在多参考/重复中有lost-success3及legacy姿态flag6，controlled legacy姿态flag16（比较记录有重叠，不当独立失败case数）。完整失败case/reference/flags在review中。Frozen learned-leg在component之上regular成功+1/+1/0却yaw全部worse；controlled成功0/-1/0、yaw两正一负，consistent leg increment两panel均False。当前任务收益方向按原rule closed，不调gain/weight/common罚项或预算追阳性，不autoPPO/替换baseline/旧模型晋升；这不是新学习或独立泛化证据，原15%且.05deg与原28/正式门不动。

round158_closure保存真实终态/source/review绑定，六类论文数据支撑出口仍缺方法贡献/稳定学习优势/新独立泛化。159依据已闭合结果收束实际可改善任务和学习必要性缺口，给160深审形成有限可证伪去留依据；不继续同一逐轮分配收益分支。160按约深审清理。

第157轮轮内分配代数与有限任务干预登记（2026-10-06，goal active）：

**可检验边界。** 第156轮所有记录中global lambda=wheel-only lambda且额外hip loss为0，旧腿轮分组恢复方向保持关闭。令轮accepted Nom为b、原filtered steering为s∈[-.3,.3]Nm，r=(s,-s)，原轮command box为|u_i|≤B_i。共同/差模坐标m=(uL+uR)/2、d=(uL-uR)/2，轮box即|m+d|≤BL、|m-d|≤BR。若要求m=m0且沿原请求线段u=b+alpha*r、alpha∈[0,1]，wheel-only scalar lambda已最大；这不是任意动作/闭环全局最优结论。

**区别于旧分组的候选。** 普通逐轮box投影u_i=clip(b_i+r_i,-B_i,B_i)，可视为min||u-(b+r)||²的闭式解。令各轮request fraction为alphaL、alphaR，则Delta_m=(alphaL-alphaR)*s/2、Delta_d=(alphaL+alphaR)*s/2，abs(Delta_m)≤.15Nm；在同状态signed differential command不小于scalar版本。Nom=(4.5,4.5)、r=(.3,-.3)、B=(4.5,4.5)时，scalar只能(4.5,4.5)，逐轮得到(4.5,4.2)，Delta_m=-.15、Delta_d=.15Nm。更多差动请求以改变共同分量为代价，不能称保持推进/平衡、不改变轮Nom执行或保证yaw加速度更好；这正是下一实际任务干预要检验的权衡。

wheel_component_projection.py仅NumPy代数原型，不接入运行控制器；1024synthetic vectors核box、normal-cone最优性、signed differential、common bound、零/非绑定float32身份、输入不改与无效输入拒绝全部过。algebra_check保存源码SHA和见证，0新physics/learning。基础box投影不称创新：Harkegard《Dynamic control allocation using constrained quadratic programming》技术报告2594（封面2004）摘要/绪论已讨论约束与饱和下控制重新分配，作者机构PDF可读：https://www.diva-portal.org/smash/get/diva2:316757/FULLTEXT01.pdf 。轮式饱和分配论文出版页本次403，仅有搜索元数据，不称取得全文；检索/近邻记录已更新。

**一次有限任务验证。** wheel_component_projection_v1/proposal登记3冻结models×2leg on/off with fixed assist×2allocators original/component×136同发展cases=1632首回合评价/0learning，GPUsource尚未准入/0执行。并发原版本作为因果比较，不用旧重放小幅翻转当收益；zero-leg的3标签是numerical repeats，非3新训练。只改变wheel输出投影及相应accepted diagnostics供原reward/obs/physics消费，hips/Nom计算/原filter/raw39 fixed辅助/请求cap/limits保持，保留invalid保护。original lambda仍只能叫原投影统计，不能解释为逐轮的新接受比例；必须分开原始/新diag与实际float32、共同/差模与任务统计。

预定development reference门：各panel/重复都complete、physical/design干净，保留concurrent uniform zero-leg及原B0/B1/B1-route成功case和原velocity/arrival/legacy roll-pitch阈值，yaw至少2/3重复lower且平均positive。单列learned-leg相对pure component固定参考的逐case/seed增量与全部代价；不把固定投影收益归RL。该门只判断非学习参考是否值得独立资格验证，原learned strong-superiority的15%且.05deg门不变，原28回归与封存/正式资格尚未评价、不声称通过。失败关该任务收益方向，不调gain/权重/共同分量罚项或预算追阳性；阳性也不替换原baseline/旧模型晋升/自动PPO。

158最小GPU实现与owner/reset/输入/接受语义准入后唯一队列，159完整原门去留，160深审清理；全方法贡献、正式5seed、消融、新独立泛化/压力/成本与新稿要求继续，当前没有合格新PPO优势。

第156轮只读轮余量诊断闭合（2026-10-06，goal active）：

wheel_headroom_attribution.py复用已冻结fixed_steering_leg.collect，新增只读record kernel、显式env buffer持有、first-terminal冻结/显式reset；原控制器、策略/RMS、滤波、room与任务未改。128synthetic GPU与独立NumPy、所有输入逐值不改、异步terminal/reset、136world实际构图/reset通过；source_contract冻结47源与proposal/unit/owner SHA后才一次启动flock排他的wheelleg-wheel-headroom-v1.service（初PID11974）。完整12jobs/816回合、首回合记录12165747有效子步（并行环境提前结束后可能自动重置，本计数不作仿真总工作量）、0学习正常exit0/inactive/MainPID0，无中断/重试。

review_wheel_headroom.py核原完整success/physics/design、case及payload/source/hash、有效子步与旧steering/room统计匹配、global≤wheel-only lambda、请求分解/绝对损失恒等及包络，全816/12panel通过，physical/design各96或40全过。全部seed/panel/condition的extra hip loss最大值/能量/L1及可回收RMS严格为0；authority_screen=False，regular/controlled eligible均空。wheel-only loss RMS约0.005935..0.017884Nm（准确值见review），当前压缩由轮自身box解释，不是leg constraints额外压缩。按原proposal关闭当前coupling-recovery假设，不恢复grouped、不调阈值/预算或自动PPO。

本次固定重放有1个成功状态变化：1762/learned-room-leg+assist/controlled case6301029由原失败变成功（该panel26→27）；其余成功状态保持。新增GPU图kernel输入只读与字段恒等不等于跨CUDA轨迹逐值身份，完整保留该变化及原结果，不覆写旧152门/来源或称重复回放独立证据。round156_closure绑定全部结果与单位终态，816仍是同136发展case重复。

用户新增充足论文数据要求已落实六类验收表，当前机制/强经典优势和独立泛化仍不足。下一157依据此诊断推导轮自身box内推进/航向权限是否存在区别于旧腿轮分组的可检验协调假设，先明确前后共同/差模变化及完整任务非退化；无明确机会即停止对应方向，不无限补只读回放或盲重训。尚无新控制器干预注册/执行。下次160深审清理。

实验数据支撑验收表（用户新增要求，2026-10-06）：

用户要求保证有充足实验数据支撑至少核心期刊论文。按本项目原计划落实以下完整证据出口；这是项目验收要求，不是统一期刊录用标准。现有回放数量不能认定为新方法/独立泛化已完成，发表仍取决于具体贡献和目标刊评审。

| 证据类别 | 必须交付 | 当前资格 |
|---|---|---|
| 主对照与学习必要性 | 合格方法与共同Nom零残差、强经典、普通残差PPO及适用非学习对照，冻结信息/时序/权限/预算/选模规则；按原计划至少5独立训练seed完成正式主对照 | 现三seed发展研究未过优势门，尚不扩正式训练 |
| 机制消融 | 至多两项能识别核心贡献的事前单因素消融，明确是训练干预、固定策略干预还是同状态counterfactual；报告请求与实际执行量 | 旧固定组合阴性；余量归因正在有限诊断，不能代替新机制阳性证据 |
| 独立泛化 | 新ID同域、左右交替/坡接台阶等未见组合、温和几何与质量/摩擦/驱动/延迟OOD，全部高度0.115–0.38m、左右镜像/双向速度分层；冻结模型后一次评价 | 136当前案例已用于发展诊断，不再作为独立测试；新测试尚未准入 |
| 压力与能力保持 | 参数/扰动/延迟分项和必要组合、易任务保留、完整接触/到达/姿态/航向/停车/高度/设计/物理失败分解；不隐藏未完成轨迹 | 原压力/旧能力可作历史参照，不能认证新方法 |
| 统计与计算成本 | 每训练seed配对效果及不确定性、场景与seed层级明确；保留全部seed/失败，不将重复案例当独立样本；实测PPO端到端墙钟，物理吞吐另列 | 当前只支持有限发展结果，主统计需对应合格方法及新独立评价 |
| 推导、复现和论文 | 方法假设/近邻区别/适用范围、全部结果图表与失败、环境/源码/配置/模型/RMS/hash/运行命令和完整新稿 | 旧限定材料保留；新的方法贡献与完整证据尚未就绪 |

执行顺序保持：当前有限诊断作明确去留→有区别性预测的候选机制/同信息强对照准入→三seed资格→正式五seed/消融/独立泛化/压力/成本→完整稿件与逐项验收。失败分支关闭并选择有依据的新问题或方法，不以更多同案例回放补齐尚缺的证据类别。每五轮深审/清理继续，下一160。

第155轮方向深审与冗余清理（2026-10-06，下一160，goal active）：

**继续价值与方法取舍。** 151–154完成接口/所有权准入、1632固定策略GPU因子回合、奖励目标与控制时序核对；三seed重复同136发展案例，0新学习。原room机制/强经典优势门及旧腿策略固定辅助增量门失败，继续这些配置的五seed、预算扩展或更难场景不值得。完整任务与设计/物理分开、保留失败和强经典比较的方法保留；不能将工程清理改称新控制贡献。全目标的具体机制、稳定强基线优势、独立泛化与完整新论文仍未完成。

**关键新判断。** fixed诊断中零腿残差的accepted/filtered steering RMS已为regular约.840–.847、controlled约.767–.770；即使没有腿请求，轮自身余量也会压缩辅助。learned-leg对应比例有时反而更高，不能凭global lambda存在就断言腿抑制航向是根因。旧grouped_projection_probe_v2_20260927的ABBA分组方案亦未过其能力门，继续保持拒绝/默认关闭；普通组间box分配本身不构成创新。153初态终奖折扣和154训练context差异也没有给改奖励/重训提供因果结论。

**只补一次可停止的诊断。** wheel_headroom_attribution_v1/proposal.json登记3冻结模型×zero-leg/learned-leg with fixed assist×96+40发展案例，共816GPU回合、0学习，目前source未准入/0执行。每0.5ms只读accepted Nom、filtered steering、当前wheel speed/actuator gains和原lambda，用原command_bounds求同状态wheel-only lambda；将未接受请求严格分成wheel-only loss与额外hip-coupling loss。绝对损失可加，RMS/平方能量不可直接相加；counterfactual使用diagnostic扭矩语义，不能冒称最终float32轮命令或新轨迹。原控制器/滤波/room/物理任务与model/RMS不动，所有记录buffer须强持有、首次terminal冻结、NumPy合成值与输入不改校对，不能复用旧缺所有权collector。

该项只筛查是否值得登记一次分配干预：learned-leg在regular与controlled各至少2/3seed同时达到可回收left steering RMS≥.01Nm、额外耦合占绝对拒绝请求≥20%，全部身份/记录/原门核验通过；.01Nm=原.3Nm请求上限的1/30，两值是事前authority筛查而非任务门。失败即关闭本当前耦合恢复假设，不调阈值/预算，不恢复旧grouped选项。通过也只允许登记一个同上下文原完整任务/物理/设计非退化的有限干预，不自动新PPO或方法晋升。156先工程准入，过后执行唯一队列；终态下一轮必须去留，不延长只读审计链。

**清理与证据。** 删除2份source marshal完全相同、git忽略且fuser无人使用的可再生pyc，共14668B；清单/hash在round155_cleanup.json。全部源码/模型/归一器/raw state/轨迹/独有失败/封存集保留，不称永久空间节省。round155_direction_review.json绑定原主门、24fixed结果、153审计、旧分组失败及新proposal哈希；三研究unit核inactive/MainPID0/exit0，0本轮新增physics/learning。下次深审与清理160。

第154轮控制上下文与研究主张收束（2026-10-06，既有源码/契约审计，goal active）：

**控制信号的“同信息”须分层说明。** 所有臂保留共同VMC/LQR名义控制与物理限幅，但Actor每20 ms接收39维包：前32维物理量按场景延迟，32–37维为当前已滤波的自身请求状态，38维为已知零起点、由延迟yaw/body velocity积分得到的路线记忆。机体速度是仿真理想观测，不是已经验证的轮式里程计。名义控制每0.5 ms读取当前仿真q/v和传感器；room层在同一频率直接读取当前q和固定设计参考。因此“相同Actor信息”不能扩写成“各决策层使用完全相同的观测与延迟”，也不能据此声称可测实现或实机验证完成。源码依据：training_contract.py:15、native/environment.py:194、route_state.py:14、native/controller.py:140、gpu_reference_budget.py:9（均位于wheelleg_warp）。

| 已有比较 | 实际动作/上下文 | 可支持的范围与限制 |
|---|---|---|
| L2-plain / L2-room / L2-constant | 同2输出F/H、轮请求恒0、同初始Actor/Critic/RMS和世界；分配分别无收缩、当前q余量收缩、固定0.9收缩 | 有限同预算分配机制比较；学习后闭环状态、归一统计及策略不同，不是同一策略下只改一个乘数；room使用额外的分配反馈 |
| V6-plain 对L2 | 六输出独立左右F/H与轮请求；自身请求状态六通道也不同于L2的差模展开 | 原始物理量/路线来源相同，但权限、维数、探索协方差和请求上下文不同；只能作一般动作权限对照，不能单独证明room机制或缺轮是唯一根因 |
| B0 / B1 / B1-route 对L2 | 共同名义控制；B1类在20 ms包上计算固定wheel PD/路线辅助，L2没有learned wheel请求 | 保留强经典效能基准；不是同动作子空间的学习必要性消融；固定辅助产生的收益不能归于RL |
| 第152轮腿on/off × 固定辅助on/off | 冻结旧room策略、归一器、原任务；辅助用raw39，仍经过原slew/filter/global-lambda | 能否定该旧组合的稳定增量假设；策略在无辅助条件训练，不能外推否定所有匹配上下文的新学习；共同wheel请求不等于共同接受wheel扭矩 |

**当前取舍。** 第148轮同预算room机制门/强经典优势门均失败，第152轮旧腿策略在固定辅助上无跨seed一致增量，第153轮总回报与航向排名不一致只提供描述证据。当前不能将“余量层清理设计违规”升级为动态安全证明、方法新颖性或学习必要性，不能将普通固定PD/路线辅助包装成新学习贡献。也不能因200k已完成就宣称优化充分或加预算可解决；尚缺训练上下文匹配的因果证据、可检验贡献及独立泛化。全部原失败门、CPU/GPU基准、模型与封存集保留。

**155深审必须作出的决定。** 首先判断地面不对称接触下是否存在一个需要学习、可用同信息强经典对照检验的具体剩余机制；如只提出更多网络/维数/奖励/极端地形而无区别性预测，停止这些增益分支。若有新假设，必须事前说明共同名义反馈/分配层时序、Actor及Critic输入、请求域与接受量、同条件固定或非学习对照、有限预算及原完整任务/物理/设计非退化门，先有限准入再学习，不把旧context干预直接晋升新训练。独立泛化、五seed主结果、方法推导和完整论文仍是全目标缺项，不以本收束文档替代。155按约审查并仅删除确认冗余文件，下一160。

**本轮验收。** 仅核对当前源码与两份冻结proposal/source契约（主study45源、fixed诊断46源全部SHA一致），复用已核评价；0新物理步、0学习更新，未修改控制代码。契约SHA：

- reference_budget_learning_v1/trainer_contract.json：`8ecb988f475a38de0175ecd0f79ca79c8cf5c5f3ae36a4906b1f5609c81b0094`。
- fixed_steering_leg_diagnostic_v1/source_contract.json：`0de82deb871f97d9722c4ea69a2707ee4e0dee4524ebf6d17d853da2730cd431`。

第153轮目标与回报一致性（2026-10-06，既有数据审计，goal active）：

原dense奖励是速度exp、三轴角度平方及acceptedresidual幅度/平滑成本，末±10按完整success；storedepisode.r不是Jpsi或PPOdiscounted objective。1632main末评价/408zero的case配对重建可见higherreturn却worseyaw（same-success另列），是不同控制状态/时长/成本的描述性排名，不是唯一奖励因果、critic失准或新优化器理由。所有输入/源/hash及policy长度公式核过，0新增physics/learning。

原gamma.99每.02s的e-fold1.99s，finalduration约6.08..9.62s，初态terminal权重median.02427（.00803..04758）；不能把这个当GAE/bootstrapping真实训练优势或“后果完全无法传递”。未MC/value校准或奖励对照，不直接改gamma/奖励主指标。当前gate失败/旧fixedhybrid无一致收益保持；工程设计作用不等于主贡献，独立泛化未准入。

154仅既有权限/共同controllercontext与主张必要性收束，155深审清理后决定具体可证伪假设，不自动新PPO或budget。objective_alignment与research_eligibility完整列限制与输入，wholepapergoal仍active、oldgate/final保护。

第152轮固定航向/腿策略因子诊断（2026-10-06，全1632闭合，goal active）：

3冻结room末模型×4condition×136development cases完整GPU回放，权重/RMS不更新；unit正常exit0，原任务/physics/design/identity/counters/Jpsi、source/payload SHA与steer/leg累计统计校对通过，24panels physical/design全pass。请求协议zero/B0/B1route单位等价保持，不声称跨CUDAtrajectorybitwise。

纯fixedsteer的yaw收益regular约.126..130°、controlled约.263..265°，controlled成功仍28，regular由95降94；不是RL收益。旧learnedleg+fixedassist regular success+1/+1/0但yaw两个seed变差，controlled success0/−2/0且yaw两个seed变差（1762丢2成功case），consistentleg增量两panel均False。按原限额152去留关闭该旧组合直接learning/promotion假设，不加PPO/参数/预算，也不恢复旧stronggate。

filtered/raw与diagaccepted/filtered明显非1，原global-lambdacoupling存在，不等请求即等接受量/insideNom替换；executed/diag近1也不是state安全证明。Policysource曾无assist，contextshift限制negative解释，不能定所有relearning无效；但现证据不授权新训练。153仅已有证据资格归纳，155深审/confirmedcache清理，所有旧baseline/数据/risk和全论文贡献/新独立泛化目标保留。

第151轮固定航向分配接口（2026-10-05，source准入通过，diagnostic未run，goal active）：

LegSteering缓存raw RouteState39于VecNormalize之前；learnerinput正常saved归一，F/H on/off从episode起始，wheelassist仅existingactions(rawpacket,B1route)。四condition scriptedzero/传腿/原请求等价、异步terminal/reset/归一输入隔离过，136world真实GPU构图/reset强统计owner过；0physical回放/学习，未改原Nom/legacy source。128synthetic readonlysteering GPUstats/NumPy与所有inputs保持校对过。

source_contract冻结proposal+allparent/current代码hash和接口/ownerproof，allSHA/budget核过，仍无runs。Steering记录raw/filtered/diagaccepted/executed增量和lambda并与legroom分报；global-lambda coupling保留，不叫insideNom独立替换或同接受wheel量，不做跨CUDA轨迹bitwise声明。152执行唯一1632fixedpolicy预算再原fullrules审阅/去留，models/RMS不更新、无newPPO。155按约深审清理，原失败和完整论文缺项不变。

第150轮方向深审/必要补验与清理（2026-10-05，goal active，下一155）：

**当前学习收益分支closed，工程作用不替代主贡献。** 全12run/2.4M、2448dev评价的机制与强优势门均失败，room改善design且不zeroActor，但yaw/态失败与fixed.9/强class/zero不能形成新增性能。保留可选工程层、全部模型与negative，不扩五seed/改primary/延budget，也不以工程negative材料结束新论文目标。

**只登记一次固定策略因子诊断，无新PPO。** 三冻结room末模型，learnedleg on/off与B1-route fixedwheel assist on/off，全部96regular+40controlled共1632GPU回合，0learning；同136旧发展case，不是新独立测试。151先freeze raw39缓存/归一顺序/zero action接口、bufferowner/reset与source/physical限制，152结果去留。zeroLeg从始终F/H输入0，不沿旧postfilter残留（两条件不混评分），purezero_noassist与B0 actioninterface、purezero_assist与B1-route等价必验。enabledleg使用原room算子，oldlearners未在assist下训练的contextshift须披露。

Fixedwheel复用原公开actions helper、normalized±1/原.3Nm/滤波slew与globalphysicslambda，不给truepose/map/mu。它是fixedsteering allocation，**不能称inside-Nom独立加强或共同请求等于共同接受量**：leg/wheel经global-lambda耦合，requested/accepted/lambda要分报；收益若steering alone解释、leg无一致增量就停该假设，positive仍非oldmodelpromotion或自动retraining，必须newindependent资格与新pairedprotocol。41yaw/44attitude、V6更差只支撑此隔离，不定唯一causal或credit/critic问题。

HybridLMC作者摘要（https://arxiv.org/abs/2204.03159）与PMLR动作表示摘要（https://proceedings.mlr.press/v270/esser25a.html）复核，hybrid/action/init类已有研究，普通fixedfeedback/room标量不单独作newmethod。五seed/独立ID组合geometry/paramsdelayOOD/复现稿件仍需真正qualified贡献。删除3closedsource等价无占用ignoredcache33718B，可再生，source/所有raw、训练与risk/unique失败保留。Fullgoalactive，source/run尚未准入，155再深审清理。

第149轮既有结果缺口诊断（2026-10-05，no新learning/physics，goal active）：

复用原flags重建2448eval全部success/身份/source，候选44失败全部有attitude、41有yaw，余physical/design/contact/velocity/height/stop/尾速/completion失败0；overlapping flags不是唯一因果标签。failure分布40legacy/3mixed/1cross_slope，115、160、380及中间height均见，不只下端。配对成功得失与各组件明列failure_component_audit，原metrics/gates保持不变。

候选leg authority retainedRMS各panel .9169..9278、appliedmean.9113..9271，大量命令改变、invalid0，不能以“全关闭Actor”解释表现。静态source表明L2无learnedwheel requests，B1route有fixedwheel/path feedback，需分开Nom强度/动作权限与学习价值；V6已有wheel authority仍差，不能仅说加维就解决。Terminal观察和value预测不是Critic校准/credit或历史的因果证据，本轮不新增这类变量。

150深审与确认清理：以此已有证据决定commonstrongNom与受控wheel分配是否需一次同信息隔离，未注册/实施新训练或修改原候选。Closedprimary性能分支保持停止，不改设计/yaw门或blind加场景网络/budget。整体新论文主贡献与独立泛化仍缺，goal active，oldgate/final封存。

第148轮有限研究全量闭合（2026-10-05，原两门均未过，goal active）：

main12run正常exit0，共2.4M、每run400PPOepoch/8000Adam，120checkpoint及所有trainingepisode/课程/末评价与源/hash审阅通过；registeredzero408随后独立GPU回放、model/RMS不更新，unit exit0。12models×136=1632、classic408、zero408，共2448development eval，136唯一case。全量原判据核过，不改partial gate或丢失failedrow。

**当前配置停学习优势/扩展分支。** room三seedphysical/design全96+40，通过原工程的作用保持；regular成功94/95/94、controlled27/26/28，plain91/92/92和26/25/26且18设计失败→room0，constant仍10设计失败。但regular Jpsi常数.9三seed均略好于room，controlledstate-aware收益不一致；机制门regular/control均False，与strongestB0/B1/B1route/zero的逐seed成功/方向/mean15%+.05deg及原velocity/arrival/legacy姿态非退化优势门均False。不能因design改善扩五seed/延budget或换primary。

完整输出、training违规与原fixedfinal规则留存，state room不是状态不变性证明，nonlearning reference仍强，不能声称learning必要或新算法优势。149先从现有记录拆解失败/动作authority/Nom条件/credit等方法缺口，150按约深审清理，再明确有证据的新hypothesis；不默认追加新矩阵/field或打开sealedfinal。整体论文方法+独立泛化目标仍active，negative工程报告不替代用户全目标。

第147轮最终门审核准备（2026-10-05，main active，goal active）：

同进程255100继续，初读9run metadata=1.8M、L2room/1763；报告时10closed=2M、L2constant进行；未所有12run或zero408，绝不启动最终score/zero并宣称学习结论。no retry/重启/budget/primary变化，150仍定期深审清理。

新只读score_reference_learning.py锁定原门，需12closedreview+main2.4M完+zero408才能运行，逐案例重建原physics/design/success/Jpsi；state-room对plain/constant两panel三seed的成功/≥2direction/mean收益，与bestB0/B1/B1route/zero的成功保持、逐seed方向、mean15%且.05deg、原velocity/arrival/legacy姿态非退化检查。所有loss/swap显式记录，no成功条件均值或新primary。纯helper阈值/单seed反向/missing-score检查过，score未执行/no额外physics或learning。完成study只判机制/失效，原145强扩展失败和完整新论文缺项保持。

第146轮增量闭合工件审核（2026-10-05，main active，goal active）：

同unit255100继续，当前前两seed四arms8closedmetadata=1.6M、第三seedL2plain开始；审核snapshot为7closed/1.4M/70checkpoints，与读期间自然进度分开。未改预算/学习轨迹，仍需完整12run与registeredzero408；当前strong扩展已由145原逐seed门排除，不能以平均结果救回。

审阅工具新增独立output，旧144proof保持原SHA，已过前两run仅在verification/episode/checkpoints/末panel完整hash不变后复用；新五run50checkpoint及原末指标/身份/实际训练计数/ActorOptimizerRMS与源只读核过，不影响GPU学习。146独立snapshot与旧proofhash引用归档，缓存proof不当外部physics证明。全3seed未齐不产生机制结论；150按约深审清理，oldgate/final仍封存。

第145轮方向深审与确认清理（2026-10-05，main仍running，goal active，下一150）：

**强优势扩展已排除，不能等待后seed“平均救回”。** 已完成primaryL2room/1761 regular94/96/control27/40，低于best非学习95/28；Jpsi.443425/1.194542高于best classical.271039/.922085。原门要求逐seed成功保持且yaw方向，已经有不可逆失败；当前配置不扩五seed/预算/OOD、不改候选或primary。它依然physical/design全96+40通过、较plain成功91/26改善，说明工程合法作用；constant.9 regularJpsi.438015略低于room，state-aware机制收益未能单seed证明。

**继续的是已登记有限机制矩阵，不是追求strong门阳性。** 同unit active255100，review时5run metadata闭合1M、seed1762在进行（变化应下轮重读）。固定12run/2.4M/seed与末模型规则保持，仅为机制与种子波动估计完成既定比较；无新增实验budget、无超时重训或半程stop/restart。全12及registeredzero408/full工件和原门必须如实报告，不能把已失败配置重新称qualified。不存在新的三seed结论或independent泛化结论，完整论文目标仍未完成。

**方法与新增实验必要性：** 初始化/标准PPO/公开39信息/同Nom及state/constant同维对照保留；instant command凸收缩不当joint-state安全。先完成原定查验，完成后才从已有数据分离wheel authority/commonNom strength/credit/任务目标等可检验解释，不默认再加数据、网络、接触真值或换主指标。新贡献/五seed/OOD要新的真实qualified方法，本轮不授权救当前配置的新预算。

删除3个已闭合contact诊断source-equivalent无占用cache21792B，source/所有原始/模型/科学失败与main活跃文件保留；cache可重生。详见round145_direction_review/cleanup，下一150按约深审清理，oldgate/final封存保持。

第144轮闭合运行只读审阅（2026-10-05，main running，goal active）：

同一unit active/PID255100；seed1761三L2 run已闭合metadata600k，现进入V6plain。只读审阅起始snapshot为前plain/room两run400k，进度自然前移与审核范围分开，未重新发起queue/切learn/改参数门或选checkpoint。最新pending及checkpoint下界存progress_evidence，下轮仍要live handle确认。

review_reference_learning.py已对前两run20checkpoint做hash/CPU保存文件加载、参数/Adam/RMS/counters及原CUDAmoments/保存不改trajectory校对；两run episode/真实终态curriculum、末96/40原身份/physics/design/success/Jpsi及未完成规则审阅过。它不修改活跃模型/不进main源码；新completed metadata不是自动已审核，完整3seed及zero尚待，不以partial分数判方法。

读取期间pending从constant变V6令旧静态预期assert失败，只发生在读侧、无训练修改；同handle重验active且继续，无错误复训。下一145如期深审方向/方法/补验及confirmed清理，活跃run/log/progress不删；所有预登记门与完整论文目标不缩小，goal active。

第143轮有限CUDA主队列启动（2026-10-05，running，goal active）：

冻结源与准入复核、确认未已有run/进程后启动唯一wheelleg-reference-learning-main-v1；seed固定1761/62/63、四arm各continuous200k，总2.4M，不warmstartengineering、无frame/physical restart。unit当前active，初始PID255100；下轮须同handle/main_progress核live/terminal，不能因观察超时另开queue。

首L2plain/1761已完成200k/400PPOepochs/8000Adam和末regular96/controlled40；先前80k checkpoint保存文件只读核权重改变、CUDAAdam/RMS有限、源/载荷/保存不改轨迹，检查加载到CPU不改变训练device。队列现进入L2room/1761。progress_evidence是时点，不是全2.4M完成；不能用首seed分数改增益/预算/primary候选或宣称方法优势。

144继续固定队列/源/evidence核验，全部12run+registeredzero408+原门完毕才判断学习收益；145按约深审清理，活跃model/progress/log绝不删。完整论文目标仍active，旧gate/final不读。

第142轮主研究准入（2026-10-05，冻结通过，main尚未启动，goal active）：

100world/4arm/3newseed共12初始化（0学习/physical rollout）核过，各seed L2三arm初始actor+critic/network/world/RMS SHA同、feature函数保持、optimizer空与CUDA/owner一致。与141四arm短训分开，V6维度/初始化effort差异披露。新mainfile复用已验make_env/continuous ledger，不改旧冻结源。

408fresh经典发展回放全部complete/physical/design通过：B0 regular95/96,Jpsi.399168/control28/40,1.181368；B1 regular94,.302660/control28,.963310；B1-route regular94,.271039/control28,.922085。strongsame-information route入最强对照，原15%且.05deg要求意味着candidate mean≤.221039 regular与≤.783772 controlled（registerednominalzero若更强仍比较）；数值门可达非RL可达证明。只读原判据/hash/Jpsi核过，两admission unit退出0。

trainer_contract绑定完整工程/init/classic与scientific source，sourcefreeze后才准入；reference_learning_zero.py作为注册408zero evaluator同时冻结，全部main完成后才能执行。主pipeline每run continuous200k、400PPOepoch/8000Adam，固定final输出，Actor/RMS评估冻结，engineeringweights不复用。mainqueue当前未启动/runs不存在，下一143按固定12run开始2.4M CUDA finite学习。原机制/最强B0B1routezero及非退化门保持，145深审/cleanup，未取得newmethod/独立泛化或global安全结论。

第141轮有限工程准入（2026-10-05，48k已通过，main未启动，goal active）：

新可选训练包装复用CurriculumEnv与route/两维接口和postupdate保存，state-room/单一constant.9/none与原Nom轮命令/filter/lambda语义保持；episode统计明确持有、终态读取后reset，stage只真实episode结束换。常数GPU128输入NumPy/凸界身份过，L2各臂actor+critic+world初始化SHA相同且新增y列0，128packet±3feature初始Actor/Value函数逐值同；旧sources不修改。

四arm各12k/n10/seed1761，共48k，独立工程milestones4000/8000用于exercise而非主学习20k/100k变更，连续learn每臂240epoch/480Adam，模型/Value/Adam/physics实际CUDA且有限。每臂26完成回合、10世界最终stage3；24checkpoint重新加载的policy/Adam/RMS/counters完整核验，保存不改physics/route/RNG。104已完成回合物理/design过不证明全局安全；全部统计sample/reset/convex/zero/lambda一致，room与fixed真实改变命令，plainobserver不改变。unit退出0，engineering_review归档。

此为工程更新，不算主试验、无score选择、无旧模型warmstart或权重promote。主2.4M尚未开始/trainer尚未准入；142先n100/其他seed初始与source、freshB0/B1/B1-route及数值headroom，再科学源freeze与主study。一切新训练违规/终态/非退化按原规则，prototype无coupled-state安全定理。下一145定期深审清理，全论文优势/独立泛化与正式条件保持。

第140轮方向深审与有限学习计划（2026-10-05，goal active，下一145）：

**值得继续的内容是一次新学习检验，不是扩大已证实优势。** 余量层已经去掉旧固定策略的测得design失败、保留原成功和多数腿authority；但yaw仍差B1且相对zero成功−2/0/+1，学习必要性未成立。接触模型控制路线/限额诊断保持关闭，不因GPU工程过门扩五seed或复杂模型。控制资格不证明随机探索全局安全，所有训练违规需记录。

**事前新矩阵：** L2plain、L2room、L2constant.9、V6plain各3个新seed1761/1762/1763，各200k，总2.4M策略步，全部CUDA学习；固定.9幅度是单一已声明development-informed对照，不扫或事后match分数。primary仅L2room；V6是全动作authority上下文对照而非差模因果证明。L2三arm同seed初始actor/critic/std/RMS/场景完全同，新增y首层列在Actor和Critic均置0、核初始函数，继承旧std/PPO参数而不按成绩重新校准。所有39信息/名义控制/原奖励/任务与physical/design1.4条件共同，维数/执行covariance不同之处披露。

**实际准入在141：** separate工程4arm×12k=48k、n10，检查optional state/constant/none GPU层、encoder/obs/episode reset/curriculum/owner、初始函数、实际CUDA actor/value/Adam和checkpoint等；工程无科学score、不warmstartmain。完整source/engineer资格冻结后才每arm连续200k；中断保消费与失败，不悄然接轨重跑。当前仅proposal，0新学习或physics。

**评价与停止：** 每seed/阶段100新固定trainbank6200000 namespace，课程20k/100k只episode-end换；final200k唯一科学比较。regular96新6300000发展采样，controlled40新ID仍重复结构化旧几何，不当新独立随机case。12模型×136、B0/B1/B1-route×136、3room末模型zero-leg×136共2448发展eval；不得仅比较弱B1或旧未调节模型。机制门要求全三room物理/designclean、对paired plain/constant任务非退化、对两对照至少2/3seed Jyaw收益且mean正；强门要求完整任务/原velocity-arrival-attitude等非退化，成功保持最强reference/zero，meanJyaw比best非学习分别regular/controlled至少15%且.05deg、逐seed方向。只机制或控制资格通过不得扩训，失败不选checkpoint/候选/primary指标或加budget。

**全论文出口未缩小：** 成功新pilot后才五seed主对照、至多两项识别主贡献消融、新ID/未见组合/温和几何及参数延迟OOD、易能力保留、实际端到端墙钟与完整methods/statistics/failedcases。state-dependent bounds/action/init已有先例，标量归一化不是独立创新证据。删除3个等价无占用cache16508B，source/模型/所有raw与unique失败保留，cache可再生。review/清理与未执行学习proposal已登记。

第139轮余量调节GPU控制资格（2026-10-05，通过，goal active，下一140深审清理）：

可选postfilter/postphysical-project GPU层与603×三模式静态identity/NumPy/凸命令/非法输入检查通过；184world factory/reset明确持有statistics/terminal mask buffers。Nom/wheel/originallambda/filtermemory保持，actual accepted-leg诊断随命令正确变化，原Actor39包的维度/时序不增。source运行前冻结，原控制器未改。全184既有development cases×B1184+三固定L2×三条件1656=1840GPU回合完成，权重/counters/RMS无更新，unit正常退出0。

资格门全True：物理各184；original设计174/176/180→候选各184，成功161/165/167→166/168/169且未丢原成功；B1成功166/design184，零腿残差各168。每seed/五高度桶同state leg增量RMS保留均≥.792681，高于预定.10，115/380端点另列。并非全关Actor，但是候选−zero−2/0/+1无一致正收益，Jpsi候选.660606/.642999/.629269仍差于B1 .478097；不能声称学习必要或方法优于经典。

review按原success/physics/design/计数/身份、源/模型/结果SHA及GPU累计性质核过，额外只读Jpsi重算及lambda均值/端点复核过，lambda均值逐值同Native。没有raw全轨迹外部重建、状态安全证明或新独立泛化。当前只控制资格，不自动PPO/五seed/重新开启旧门。140集中判断下一learning值得性、Nom公平条件、等actor+critic初始函数和必要机制对照，再在价值明确后注册有限新三seed；按约清理确认冗余，目标全范围保持。

第138轮分支去留与同信息候选（2026-10-05，goal active，140深审清理）：

**明确关闭接触预测控制器/因果诊断分支。** 7236限额已用完，CPU改善依赖actual contact point/fullframe-gap及真实参数，超出现有可得信息；GPU、continuum、状态不变性与边界资格仍未通过。保留证据，不再为此增加字段/normal选法或预算，也不以负结果收尾替代新论文的贡献/学习/泛化目标。

**新的最小定义仅到代数原型：** 用public neutral standing qref(h)的设计room归一当前主动joint room：rho=clip(min_j[(1.4−abs(q_j))/(1.4−abs(qref_j))],0,1)，reference room非正拒绝；腿执行command=Nom+rho*(accepted−Nom)，轮command保留。站姿reference来自现有Native约定（.3时0，否则sim.ik），编码时序与Nom相同，不读contact/passive/base/true mass-mu或future map，Actor39及原filter memory保持。可证明same-state命令端点凸界，不证明耦合next-q、Nom可恢复、poststep转速力矩或不变集；不得称新算法/安全屏障。适用条件及可能抑制有益残差须真实回合测试。

**验收与后续：** reference_residual_budget.py可运行最小检查通过，266height与603static输入的单调/端点/wheel/zero identity/convex检查过；发展meanrho.9013、旧critical fixture.0050不是轨迹安全或有用行动证明。加权端点旧式有4.44e−16 zero identity失败，旧源/说明保留，改差分形式后逐值身份通过，容差未改。0新physics/GPU/学习，尚未集成控制。

139先freeze可选GPU operator/strong buffer ownership、每步Nom/zero/wheel/原lambda与rho记录资格，再按proposal全部184既有development cases做固定控制资格：B1control184+3L2×3conditions×184=1840episode。须原physical/design全部过、任务不差于original、强B1/zero分报；各seed五高度桶腿增量RMS保留≥.10同state未调节增量的预注册非退化proxy、115/380另报，未定义不算通过。候选失败不扫reference room或阈值、不过边界不扩PPO；即使通过还需强经典新增收益与等初始actor/critic同信息的新三seed预试验，五seed与新独立ID/组合/OOD/消融任务保持。当前无新rollout/actor/RMS更新。

第137轮接触四臂因果诊断（2026-10-05，全量重算通过，goal active）：

freeze position/frame-gap transfer后执行全部603×四臂，producer2412+review2412CPU步；与136identity合7236预算用完，另1260forward、0GPU回放/学习。固定geom-order右手frame变换，无符号择分/参数/阈值扫描，原积分流程保持；每臂pos/frame/gap在constraint与implicit后逐值保留，override不改q/v/ctrl，未改臂全字段与标准step一致。两unit正常退出0。

新576最大active-q相对actualgeometry CPU oracle误差：原6.36942e−5rad、只位置1.13630e−8、只frame-gap仍6.36942e−5、两者3.45365e−9；原1e−12容差分别满足263/518/263/519，两者仍57未满足。旧27static fixture四臂全q/v同，不重新认证其GPU采集事件。位置受控替换对本批主要CPU表述差异有显著作用；不能断言全部误差唯一来源。不同fixture组与实际参数特权条件分开报告，不冒充新独立泛化。

改写的是actual current-contact oracle字段，未改变Actor/真实控制/原ULP/1.4门，也不是新可部署预测器。原14GPU/9CPU漏包、GPU数值/信息可得性/连续域与真边界安全仍未通过；138必须依据此一次限额诊断作去留决定，不能因CPU误差变小继续加privileged fields或扩大预算。下一140按约深审清理，论文贡献和学习/泛化整体验收尚未完成。

第136轮因果试验pipeline身份门（2026-10-05，通过，goal active）：

contact_manifold_causal.py冻结源后，全部603状态×两模型×两种独立执行流程共2412CPU步。fresh标准mj_step与显式fwdPosition→make/projectConstraint→fwdVelocity/Actuation/Acceleration/Constraint→implicit结果全q17/v16、qacc/warm、ctrl/actuator_force/constraint_force逐值相同，time/contact/constraint计数同，所有max0；原integrator3及1e−12不改。compatibility_review复核全部1206配对/源/载荷，无额外physics重算；unit正常退出0，0GPU回放/学习。这只资格化未修改pipeline，不是四臂干预或控制安全准入。

旧27fixture的capture源也有pre/stats owner未显式持有风险。保存的27全hinge积分关系一致不足以消除此风险；本轮只用这些值作static CPU临界夹具，不把原GPU-event provenance重新认证。新576采用已验证owner的v2。旧资料保留，不盲重跑或重用旧capture，也不把两组拼成新独立测试。

下一137冻结contact-transfer source/固定geom-order右手frame转换，执行全部603×四臂和完整review，剩余4824CPU积分步；逐臂断言改写point/frame/gap在constraint/integration后保留，不能被再次collision覆盖。未实际改contact字段或产生四臂score，不能提前判position是唯一原因。138按计划去留决定、140深审清理；原预测失败与整体论文方法/学习/泛化目标保持。

第135轮深审与冗余清理（2026-10-05，goal active，下一140）：

**方向判断：整体论文计划值得继续，当前预测控制候选停止直接推进。** 131–134得到独立状态覆盖失败与接触表述差异，价值在于排除错误假设；没有新方法优势。48唯一场景中B1成功47/48、design48，L2三seed成功及design45/45/47。576状态的14GPU/9CPU漏包、额外理想fullstate/contact需求和未经控制时序验收的CPU影子程序，不支持部署。0 false-safe来自远边界样本，不等于安全证明；已读留出现在只能作development诊断。继续扩大地形/网络/训练预算，或给旧包络调reserve，均不适当。

**方法判断与限额补验：** 停止normal-gap→无限plane→16corners当安全层，不追逐更多字段。仅保留一次位置信息/接触frame-gap的2×2因果诊断：全部新576+旧27critical fixture共603，两组分开；两模型未改pipeline与fresh standard mj_step全q17/v16身份先过，再四臂共2412步及全量review2412步，identity2412，总预算7236CPU步、0GPU回放/学习。固定geom顺序与右手frame转换，保留integrator3；不按成绩翻normal、调force阈值或更换样本。136–137执行、138强制作去留决定。API身份失败即停止；即使oracle接触数据可复现CPU，仍不证明这些字段可测、GPU误差或连续参数鲁棒性。此为离线非物理counterfactual诊断，不是新控制器或实验准入。

**返回论文主线的条件：** 若只是可复现实现错误，修正后需新独立资格验证；若依赖不可得完整manifold，则关闭该控制分支，回到可得关节状态/控制历史下的最小共同Nom修复和残差权限设计，先推导coupled-state限制与同权限实测。不自动实现候选，不把状态相关幅度调节称首次提出：PMLR动作表示论文已研究初始化/步间行为（https://proceedings.mlr.press/v270/esser25a.html），UGent2026官方摘要已有状态相关残差界/双环（https://biblio.ugent.be/publication/01KVA83TWHK288BDR6SKJF16C6，DOI10.1016/j.engappai.2026.115343，全文仍UGent only）。须找到本机可检验的新增作用与相对强经典收益。

**必要补充仍保留原范围：** 全高度与真实临界状态/非退化、强B1和零残差/旧机制、共同信息/权限及actor+critic等初始函数；新三seed实际优势后才五seed主对照、至多两项识别贡献的消融、全新同域/组合/温和几何+质量摩擦驱动延迟OOD和易场景保持、实际端到端墙钟。还需要方法推导、失败统计与复现材料；本轮不以负结果摘要或工程报告结束新投稿目标。

**清理与成本：** 原四轮384GPU诊断回合（192无效保留）、21888CPU积分步含重算、2304forward+1kinematics、0新学习；不声称在线速度或PPO加速。删除3个source完全等价且无占用的ignored pyc共29965B，可重生；source/模型/归一化/原始数据/全部独有失败保留。证据及去留在round135_direction_review，预算计划在contact_manifold_causal_v1/registration；新实验未执行，136先source准入。

第134轮接触表述审计（2026-10-05，独立核验通过，goal active）：

事前冻结后对全部576状态、actualgeometry/旧current-plane、相同真实参数做mj_forward；q/v逐值不动。Producer与review各1152forward，另1kinematics见证，0积分/新GPU/学习；1345wheel contact全对应、无缺失/多匹配或nonwheel。约束维数、存在状态及friction/solref/solreffriction/solimp无差异。1.5s全部接触与主动qacc相同，3.5s法向/间隙/位置已有差异；首次报告接触的法向/间隙几乎相同，却位置最大99.8678mm、切向最大2.58825mm，主动qacc差峰254.777rad/s²。这是接触数据位置差，不是机器人位移，不能从描述性比较证明单一因果。

最大位置见证B1/5700014/event2：原normal按geom顺序定向，不能假定总指向轮心。actual/plane contact位于轮下/上两侧，normal/gap同；实际force0、plane有载荷，仅1/1345此类转变。官方法向定义文档与installed3.12字段交叉记录在normal_and_load_semantics_audit，未将3.6文档当3.12源码证明、未判定符号bug或GJK唯一根因。constraint存在不等于承重，理想collision descriptor不自动等于可测触觉。

原132预测包络gate失败保持，不部署/扩PPO或见holdout调reserve。下一135集中深审131–134的科学价值、方法/信息/计算成本与独立边界证据，决定停分支、替代方法或必要有限因果试验，不能机械追求更多字段/更复杂模型；同时删除确认冗余，所有唯一失败/模型/状态保留。

第133轮误差分解（2026-10-05，独立核验通过，goal active）：

冻结后对全部576状态做两新offline分支：当前接触模型+真实参数+warm0，实际几何/真实参数+保存warm；复用132审核过的实际几何warm0与GPU。原16角点/ULP与1.4门不改，真实参数与保存warm不提供给控制器。独立重建两分支、源/载荷/身份与有符号分解通过；producer1152+review1152=2304CPU步，0新GPU/学习，两unit正常退出0。

576真实参数contact点均在原raw corner内，有限点不能证明连续域；warm差峰1.73472e−18rad，不解释本批失败。支撑表述差峰6.36942e−5rad，GPU与实际CPU savedwarm残差峰6.92334e−5rad；1.5s支撑差0/残差≤6.66643e−8，3.5s与首次报告接触差异更大。有符号恒等GPU-contact=(CPU0-contact)+(CPUwarm-CPU0)+(GPU-CPUwarm)逐点成立，不把不同状态max相加作鲁棒界、也不当唯一物理因果比例。支撑差包括接触构成/几何，backend残差不只浮点噪声。原132的14/9漏包和拒绝部署/训练保持，全14miss因子只读登记，不重新选分。

下一134限于接触位置/法向间隙/参数与求解器表述审计，先区分可修实现与缺失信息，不放宽预留或扩失败PPO。135必须集中评判本预测方向的继续价值、信息/实现成本、是否需替代方法及边界样本补验，并按约清理确认冗余。核心方法优势、递归安全和独立泛化尚未成立。

第132轮全量新状态预测（2026-10-05，独立重算通过，goal active）：

先冻结旧current-contact/16公共角点/原ULP reserve的全部实现源，再对v2全部576状态执行9216CPU corner预测与576actualgeometry offline对照；actual参数只给对照，不挑corner，0新GPU/0学习。Producer全量GPU14漏包、CPU9漏包；1.5s192均覆盖，3.5sGPU6/CPU5，首次solver报告接触GPU8/CPU4，各192。最大excess1.01679e−4rad与5.95410e−5rad。48真实场景参数全在原声明域，固定预测包络gate=False，不扩大reserve/选模型/晋升或扩PPO。0 false-safe因这批actual sampled-state最小设计余量仍.03805rad，不能作边界安全或完整回合成功证据。

接触时序需修正论文表述：MuJoCo Warp先在pre-q做forward/collision再积分，native在step后读取的contact buffer并非从post-q重新碰撞。因此event2仅为首次报告目标接触步，pre-q理想描述可能已包含此接触，不能当预测未观测future新接触的证据。原样本和评分保留，installed backend源码与SHA在contact_event_timing_audit。

原producer在全部hashed batch完成后因NumPy int64无法写summary JSON而退出1，预登记finalizer只转换计数格式，不重复物理/更改source/分数。独立review已完整重建9792步及全部域、描述、身份与汇总，通过576状态且确认原14/9漏包；unit正常退出0/inactive/MainPID0，closure总成本producer9792+review9792=19584CPU步、0新GPU/0学习。5条仅GPU漏包，CPU-GPU active-q差峰6.92334e−5rad，不把所有误差单因果归于跨后端。第132轮关闭，整体目标仍active。下一133有限误差分解：支撑近似/角点是否包住真实参数model/跨后端分别核，真实参数仍offline，失败当前包络保持。135方向深审必须判断新增信息与模型复杂度是否值得继续、是否需替代方法和补充边界实验，再清理确认冗余。

第131轮独立状态采集及记录器修正（2026-10-05，goal active，下一135深审清理）：

只读采集原B1+三固定L2在48已登记场景的三类事件；v1虽完成192回合，却因CUDA graph不持Python数组所有权，pre/ever释放而失真，全部snapshot拒绝预测使用。weakref证明、原源码/数据/合同/日志与rejection保留。env持有全部4buffer后预登记v2同条件追加192纠错回合，未使用预测得分或调参；累计384GPU回合、0学习，v1不拼入有效成绩。

v2已完成192回合/576状态/0缺失，每控制器三事件各48。可运行自检核事件与真实factory/reset生命周期；独立review核原身份、success/physical/design、源/模型/载荷SHA、一次性事件时间窗/首接触old0-current>0、控制分解与主动关节float32离散积分逐值一致。B1 success/design47/48与48/48；L2三seed success及design45/45/47（各48），physical全部48。仍有设计失败，不是控制准入或新方法优势。审核器曾错把B1轮残差当零，保留失败源/log，按原B1 yaw契约修正审核、无采集重跑。

下一132仅冻结不变公共16角点/原ULP reserve实现并对全部新state核预测覆盖，区分scheduled和首次接触创建、actualGPU和offline actualgeometry；原1.4门不改、不依留出得分加buffer或选模型。理想fullstate/contact信息仍强于Actor39，未来控制比较需同给；零漏包也不能称连续域保证/递归安全。可信工作域与误差假设后才共同Nom修复/Actor余量、有限同权限回归和新三seed优势门，formal五seed/OOD待准入。只有仿真已确认；论文核心贡献、强经典优势和独立泛化仍未闭合。

第130轮集中方向/方法深审与独立state注册（2026-10-05，goal active，下一135）：

**方向值得继续的条件：** current ideal normal/gap在27点复现actualgeometry作用，public参数corner有限覆盖支持独立验证；不再扩大失败的点Nom/20ms cone/PPO分支，也不把216interior无漏包当连续域证明。测量假设强于旧Actor39、oracle全景泄漏、CPU-GPU/接触切换与递归viability都是未闭合项；新核心论文的方法贡献/强经典优势/独立鲁棒泛化仍未成立，旧初稿ready不替代。

**新实验已事前注册：** contact_state_holdout_v1固定新5700000..47的48场景，九地形均覆盖、目标高度轮循0.115/.16/.24/.30/.38；原B1和三固定L2末模型，max192GPU回放/max576snapshot/0学习，事件为pre-time≥1.5、3.5s和first post目标接触创建保存pre-state。后者刻意检查当前接触描述无法预知新接触的问题，不按failure分数选case。旧namespace无重叠，仅为同固定控制器条件下独立state/case采样，不把四控制器重复48当192独立环境。当前未采集新score，collector/source须先准入冻结。

**下一131开始：** 实现并冻结只读事件记录器和不变角点/ULP模型，保留所有任务失败/状态；分scheduled与contact-creation预测覆盖，并以actualGPU和offline actualgeometry分别核每个漏包，不改原1.4门、ULP或模型选法。若失败停止对应假设，不能见holdout后增buffer；无漏包也只是有限覆盖，需声明实际信息/连续参数证明界限。若支持明确工作子域与误差假设，才推共同Nom修复和Actor余量，有限同权限控制/非退化与初始化机制消融后新三seed学习，过门才正式五seed/OOD及solver/contact敏感性。

**冗余删除：** 3个编译code与当前source完全等价且无占用的再生cache20563B，source/数据/checkpoints/所有unique失败不删；round130_cleanup.json保留hash并核已删，import可能重生成不当永久空间节省。五轮证据/停止及继续条件在round130_direction_review.json，下一135按约深审清理。

第129轮公共参数域有限包络检查（2026-10-05，goal active）：

先登记mass7–7.5、左右mu .6–1、drive差±.03（原声明训练域），16角点+seed129001生成8固定内部点，在27旧事件状态/current理想接触描述下做648预测、216actualgeometry oracle对照，warm0、0训练/0新GPU。角点全算，预测不读取true param选择角点；内部标签仅offline验证。数值reserve为原1.4位置的一float32 ULP，属于跨算术比较，不是控制门放宽或接触模型鲁棒证书。

独立review_contact_uncertainty.py重算全部864状态命令/域/参数生成与输出/SHA：216内部预测均raw corner覆盖，216actualgeometry oracle均ULP-reserved corner覆盖，27原GPU全覆盖，contact与actualgeometry同内部param difference0，max包络宽2.55202e−6rad。零漏包只限固定8内部点和同27state，不证明连续域极值在corners、更不证明递归状态安全或独立场景泛化。Disposition不准部署/新PPO。

下一130集中深审应明确contact-informed区间建模是否值得继续，独立state/接触模式与实际测量契约如何验证，能否在所需输入可得/不使用future map/unknown true parameters前提推共同NomActor工作域层；经验corner极值不偷变数学上界。原全部控制/方法门与失败保持，不扩已失败训练；确认冗余清理按约执行。

第128轮当前接触表述条件测试（2026-10-05，goal active）：

先登记27CPU步（既有27保存状态），以actual collision仅offline模拟当前触觉：轮侧别/接触法向/有符号间隙，不保留障碍标签/未来geometry。预测器构造接口仅该列表+声明的ideal current q17/v16/ctrl；Nom7kg/.8/drive0/warm0，ellipse支撑函数按当前normal/gap建局部plane，explicit pair仅相应轮，原floor/bump禁碰，避免高轮无限plane影响低轮。descriptor是新增理想测量假设，不等于旧Actor39能取得；如果未来用于控制，必须同样供给全部方法，不称硬件触觉认证。

独立review_contact_response.py重建descriptor、nom物理参数、pairs/碰撞隔离、原27预测和SHA；与actual-geometry/fixedparameters一步结果逐值相同，difference0。这27当前状态中不必完整未来map即可复现几何作用，但不能推广至未观测接触切换/曲率。误差max8.8887e−7rad、false-safe0；原ULP=1.19209e−7预算仍失败，development_gate=False。无学习/新GPU，只有27CPU影子步，原1.4门不豁免，disposition拒绝部署或扩训。

下一129有限分析公共未知mass/mu/drive域的contact-informed响应区间，预注册corner以及独立interior参数测试，不传真实参数为预测输入，也不把corner/经验误差直接当连续域鲁棒上界；独立状态留出/CPU-GPU和接触切换仍须验证后再进入共同NomActor可行层。新goal继续，130深审清理，原模型/数据/失败/CPU/封存均保持。

第127轮预测偏差全因素诊断（2026-10-05，goal active）：

事前27状态×2³组合注册216CPU步：actual geometry或state-derived lower-flat、actual grouped mass/mu/drive或fixed7/.8/0、actual warm或0；全部组合和状态纳入，actual只是offline诊断真值，不给Actor/控制器，也不作为部署优势。review_prediction_factors.py重建全部输出/因子/原门baseline及先前oracle，hash通过，0学习/0新GPU。

| 几何 | 动力参数 | 最大q误差（微rad） | 假安全 |
|---|---|---:|---:|
| FK-flat | fixed | 12.315 | 3 |
| FK-flat | actual | 12.259 | 3 |
| actual | fixed | .889 | 0 |
| actual | actual | .08156 | 0 |

每种warm0/actual输出逐值相同，只限本solver100/27状态。主要误差与支撑几何表述关联，unknown动力参数为其次；actual geometry包含完整场景，不能据此认定currentcontact一定足够，params因素合并亦不单说mass是根因。无“挑最佳”预测器推广，旧false-safe和原1.4/ULP门保留。

下一128登记current-contact-only表述的有限模型诊断，保留fixed design、不用未来map；理想当前contact descriptor若使用是新增信息假设，必须声明并同给所有未来比较方法，不能包装旧Actor39已知。先区分softgap/normal观测需求及unknown参数影响，独立保留误差后才能构造Nom修复和Actor余量，未准入之前不新长PPO。目标继续，130深审清理。

第126轮固定设计可得模型未准入（2026-10-05，goal active）：

先登记再CPU54步：固定mass7/friction.8/drive0/solver100/dt.0005、warm0，构模接口只接受当前q17/v16/已知ctrl及两预定支撑hypothesis，不传实际case参数/未来障碍geometry。当前完整状态是假设理想encoder/被动角及base pose/velocity estimator，强于旧Actor39，若实际使用须同样供给所有比较方法，不当旧proprio已经具备。Nom FK当前wheel-bottom下建立common lower horizontal plane或two current horizontal pads，只是支撑假设不是真实terrain/contact观测。

27状态每hypothesis q误差max1.23155e−5/1.23081e−5rad，各3false-safe；事前预算一个float32位置ULP=1.19209e−7rad、false-safe0均失败。review_nominal_response.py独立重建54预测/固定参数/warm0/实际输入与SHA/原门过，disposition拒绝部署和新PPO，原1.4门不改。不可用前一actual-geometry Oracle LP可达性冒充本模型有效，更不可把这27经验max转为鲁棒上界。

下一有限诊断需分别考察支撑/软接触表述、unknown dynamics参数、warmstart因素，确定额外可测支撑状态是否必要；actual geometry/contact只作明确offline Oracle对照，不能给Actor/Supervisor偷加truth，也不扫ground offset/mass选通过者。没有模型与独立误差准入前不做新的工作域控制层或长学习。目标继续，130深审清理，原数据/失败/CPU/旧门/封存均保持。

第125轮集中方向/方法深审及可达性诊断（2026-10-05，goal active，下一130）：

**方向决定：** 不扩停止的2.4M/20ms方向cone分支，不调guard或豁免微小越界；条件性继续共同Nom修复与Actor状态可行余量的响应模型。力矩方向只给有限功率条件，不能作加速度或状态域保证；actual-CPU oracle只作诊断，不能泄漏到控制器/Actor或当鲁棒误差证书。新方法优势、可部署信息模型与独立泛化仍未齐，不能用旧初稿ready结束goal。

**本轮新证据：** audit_joint_reachability.py在已存27状态，按原2D坐标±1和动态TorqueBox，以±.01差模小扰动的局部响应拟合LP最大化一步最小关节余量；角度行只缩放1e6以避免优化器容差吞没原1.4门，不更改门或制造额外控制权限。27预测及27额外CPU shadow验证全内侧，min margin3.38684e−6rad，pred与CPU误差max6.30718e−13rad；全部候选超.01局部半径，max坐标变化1.41878，逐个CPU重验原物理模型，不将局部斜率直接泛化。0新GPU/0训练，实际geometry/参数仅offline。样本中权限可支持一步内侧，但未证明递归可行/未知接触鲁棒或实际任务改善。

**下一项补充：** 先注册可部署信息契约（固定设计7kg、可得encoder/IMU/commands/控制器memory、support/contact假设），比较其一步响应与oracle状态并用新的保留误差案例检验；信息不足须明确需可测信号，不偷偷加真实terrain/mass/friction。可信模型及不确定性假设之后才做共同Nom纠正与Actor余量层，解释不可行状态/空集合，保留原执行权限和任务门；再全高度/非退化/机制消融与新三seed学习，通过实际强经典门才五seed/OOD/接触和CPU-GPU敏感性。不要把拟合样本最大误差当前瞻上界，或用一个LP存在点包装已完成鲁棒方法。

**冗余清理：** 三closed progress marker149B删除；内容/原SHA与completion/review证据SHA在round125_cleanup.json保留。三unit inactive/MainPID0、无fuser占用，现auditor不读progress，dashboard只读无关旧目录；模型/轨迹/原注册失败/所有独有科学失败/封存集不删。完整五轮证据/值得继续的条件和额外实验先后见round125_direction_review.json，下一130深审清理。

第124轮同状态联合响应与跨后端偏差测量（2026-10-05，goal active）：

事前登记三固定L2末模型/停止的cone控制/全部27低高度发展案例，81真实GPU回放、0学习；只读capture pre完整q17/v16/warm16/执行NomActor命令与post q17/v16，first运动学预测/near1e−5/first实际跨界事件最多243，实际27。无actuator activation/mocap/applied forces，时间为步计数派生，本静态直接力矩模型可用，不声称通用状态或同轨迹恢复。

同snapshot状态按实际模拟场景CPU MuJoCo做actual、Nom-only、±0.01 Fdiff/Hdiff六分支，共162 shadow步；保持差模、原坐标和speed/torque Box。Shadow知晓geometry/参数仅离线诊断，Actor和控制器没有获得它们，不能当部署名义预测器或公平的oracle控制成绩。review_joint_response.py独立重建状态/命令/所有分支及原81任务/物理计数、source/model/hash通过；审核初import缺tools路径的源/log保留，修复未改实验或重训。

CPU-GPU q误差max8.156e−8/7.678e−8/5.308e−8rad，v误差max1.465e−4/1.458e−4/4.049e−5rad/s；小扰动q中点残差约6.30e−13rad，命令中点偏差5.70e−8Nm包含float32基准命令差，因此不把局部对称性直接升级全域线性/状态安全证明。三新捕获跨界中actual-CPU均越1.4，Nom-only两例内侧、一例仍外侧，说明只抑制Actor可能不足；freshGPU有一原1610边界例未再越，原四失败不删、不挑回放分数。

结论限于同状态有限输入效应和实际跨后端偏差，未获得可信在线加速度上界或方法优势。下一125深审应决定共同Nom/Actor响应模型是否值得推进、如何只用可得状态与固定设计而不偷用未来terrain/参数，并安排独立误差/动态工作域验证；经验最大值不当鲁棒证书，不扩旧PPO/扫guard参数。原数据/失败/CPU/封存保持。

第123轮离散可行性缺口确认（2026-10-05，goal active）：

只读现243回合数据，audit_boundary_discrete.py核全部14,568,480主动hinge更新：q_next=float32(q+float32(dt*v_next))逐值完全一致，9组split mismatch0；若改成融合/实数计算则有微小误差，max≈5.97e−8rad，全部落在乘法/位置加法舍入预算。已安装MuJoCo Warp源码_next_position/_advance确认先速度后位置的顺序并留SHA；不将hinge公式套自由关节四元数。未重跑物理/训练或改原约束。

剩余4例未加位置舍入时也均越界，unrounded overshoot=.24175/.20006/.06258/.21177微rad，round项−.02717/+.01452/+.03279/+.00280微rad；不能以“数值误差”删除/豁免失败。3例pre向内而post向外，事后有效加速度项超过位置余量，线性20ms外推未触发；一例guard已触发/Actor该关节≈0仍越界。证明当前方向约束与速度外推并不是状态域保证，而非已证明某个动力学控制器解决了问题。

一步条件：对s=±1，需s*(q+dt*v)+dt²*A_s+round_bound≤1.4（A_s≥s*a_eff）。a_eff是执行后的有效速度增量；其前瞻上界必须覆盖候选Nom+Actor总输入、闭链/接触、未知参数及数值顺序。发展迹中的事后有限差分或最大值不能当在线控制量/鲁棒上界。第124轮先做有限联合输入响应/不确定性测量，明确可得状态和预测假设后再构造、检验状态可行域层；不调旧horizon/buffer或放宽门，不启动新长PPO。既定目标继续，125方向深审清理。

第122轮方向保护投影受控检验／停止该候选（2026-10-05，goal active）：

候选在原滤波与物理投影后、physics前作用于两腿有效F/H，guard为abs(q)+max(0,sign(q)*qdot)*.02≥1.4且sign(q)*Nom<0。半空间sign(q)*M_j*u≤0、原动态力矩Box和坐标±1，最小四腿电机命令增量距离，2D face/vertex枚举；零可行，Nom/轮不变，未触发与零残差身份保持。所选向外运动关节增量功率非正，不推出耦合加速度/工作域不变。15实测状态与SciPy QP对照和Box/最优代价/身份通过，最终unit_v2源码hash冻结。类型API及独立FK dtype差的失败原源/log保留，修复只匹配实际API/精度，原1e−9不放宽。

先登记再GPU243回合：三固定L2×原/候选/滤波后零腿增量×全部27低高度案例，0学习；原成功19/20/19→cone20/21/20→zero21/21/21，设计20/22/24→26/25/26→zero全27，物理全过。独立核3,642,120物理样本、原success/yaw/时间、q-v连续、命令/FK/坐标/Box/cone条件；投影908/880/292物理步，max方向误差<4e−17Nm，未增加物理Torque越界，但仍4工作域失败。continue_gate=False，不进入新训练，不扫horizon/缓冲或放宽1.4rad门。

剩余4例不是简单宣称保护有效：一个guard已触发、该关节Actor增量≈0仍越界；三个pre速度朝内，线性外推guard未触发而post越界。max超界约.21–5.10微rad仍按原门失败，不以数值噪声豁免。第123轮先读现有迹核半隐式积分/前后速度变化、触发/恢复时序及数值分辨率，再决定带可检验动力学/离散条件的状态可行性机制；不靠力矩符号继续宣称安全或用更长PPO追阳性。原数据/失败/模型/旧门保留，125深审清理。

第121轮低高度0.5ms设计边界测量（2026-10-05，goal active）：

已事前登记全部发展h<.16案例（19regular+8controlled），原B0/B1与固定三个L2末模型，共135回合、0学习。pre记录原控制前active q/qdot，post记录physics/contact/physical证据之后、native.after之前的q/qdot、accepted Nom/Actor/total及actual actuator torque、lambda、target-contact mask、设计域瞬时/累计余量及filtered/target请求。记录kernel只写独立buffer，原控制、物理操作顺序、全部成功门/原源不改；FreshGPU回放不称原轨迹位级相同。

review_design_boundary.py核全部2,023,387采样时标/q-v连续/命令分解/工作域余量、原task/physics/design/yaw与全部载荷/源/固定模型；weights/counter及RMS无更新。B0/B1成功21/27、设计27/27，L2三seed成功19/20/19、设计20/22/24，物理全27/27。首次越界15个全部已向外运动，Nom方向内侧、Actor附加方向外侧；其中2个早于首次目标障碍mask，最早1.8215与1.8105s。Mask原语义不包含普通地面支持，不能据此排除接触/外力。FK Jacobian同pre-state重建Actor增量1e−9内匹配，未平滑目标映射在15个跨界时也均外侧，无“目标内侧实际外侧”例，不凭请求坐标变号认定滤波是根因。

下一受控干预：在实际滤波输出层验证近工作域边界时Actor不抵消Nom恢复作用的一个候选，与原执行和零腿增量对照；原1.4rad工作域、原物理输入权限/成功门保持，全部低高度案例和三模型纳入，不扫阈值。须明确力矩方向不等于耦合加速度，Actor非正所选关节功率条件也不保证闭链整体状态域不变；当前记录不是单独Actor因果或动态安全证明。先证明实际输出/投影数学条件并做有限固定策略实验，再决定是否新学习，不直接扩5seed或另跑长PPO。压缩完整采样/失败保留，unit退出0，下一深审清理125。

第120轮集中方向/方法深审与清理（2026-10-05，goal active，下一125）：

**方向决定：** 原路线/动作pilot的预注册信息与正式扩训门失败保持，不扩5seed/预算/极端场景。条件性继续共同Nom/Actor设计工作域与执行可行性机制，暂不堆网络、换PPO或增加特权Critic。旧初稿可写不等于新核心投稿目标完成；方法贡献/优势与新独立泛化仍待。

**新增证据：** 对1632学习发展回合做完整已有数据分组，设计失败60（regular28/controlled32）全部在目标腿长<.16m，57个在.115端点，物理代理均通过。当前controller.py输出投影只含力矩/速度约束；名义姿态参考投影不等于保持实际1.4rad工作域。此为明确合同缺口，不是已证明越界由Actor单独造成或已获得动态安全算法。

**初始化审查：** 复现三seed初权重SHA，888固定参考包/初始identity RMS，只改新增归一化特征−3/0/+3，初始Actor与Critic已变（最大轮请求差.002134Nm、value差.808228）；将两网络第一层新增列置0后Actor/value各样本恢复完全相同初函数。0学习/0物理，静态假设输入不是训练占用或负门因果证据。重建首次线程未匹配使初SHA失败，保留源/log，按训练torch单线程后精确匹配，不改科学条件。若后续信息分支再做学习，必须先匹配初函数，不与约束层修改混合。

**下一项补充实验：** 先事前登记固定三个L2末模型和不变强经典，按低高度而非最差分数选择发展子集；记录全部0.5ms主动q/qdot、accepted Nom/Actor力矩、lambda、工作域余量及首次越界，核封装/记录不改变发令和全部成功门。先判断Nom、残差方向、接触或时序贡献；耦合闭链中力矩符号不是加速度符号，不能凭终态交叠标志直接构造并宣称CBF安全。之后才推导一个共同可行域机制及模型/接触假设并做受控验收，再考虑有限新学习。当前原预试验全保留，不扫阈值追阳性。

**清理：** 删除4个字节相同、无占用的闭合工程progress marker276B，其内容/原SHA和完整verification/checkpoint证明SHA保存在round120_cleanup.json；审计与现dashboard无读取依赖，原模型/归一化/源码/回合/失败/封存集不删。证据、研究值得继续的条件、方法限制和实验顺序记录于round120_direction_review.json及mechanism_audit_120.json；每五轮继续深审/清理。

第119轮全预试验独立验收与停止扩训（2026-10-05，goal active）：

review_route_pilot.py独立核12run/2.4M、120重载CUDA模型/Adam/39RMS/真实步与epoch、初始权重和世界配对、5772训练回合身份/物理计数及实际课程切换、2040发展评价逐例原success/物理设计/yaw重建、source/载荷SHA及bootstrap/时钟/保存状态。训练episode未存末FK，未冒称其全部success可独立重建；发展评价末FK完整。原始新源/协议、错误登记更正及失败保留。审核表字段重名导致第一次写表TypeError，原源/log归档，改design_failed只修输出键后完整readonly重载通过，不改判据或重跑物理。

| 方法 | regular成功/288 | controlled成功/120 | regular设计通过/288 | controlled设计通过/120 |
|---|---:|---:|---:|---:|
| M3-zero | 88 | 41 | 281 | 110 |
| M3-route | 38 | 29 | 278 | 108 |
| V6-route | 191 | 76 | 284 | 118 |
| L2-route | 275 | 81 | 281 | 112 |

各物理288/120全过，所有发展轨迹自然completed。信息配对regular−27/+14/−37，均值−17.3611pp/仅1正，controlled−10pp，设计多3/2失败；information_gate=False。正式候选M3-route/L2-route保留经典/controlled增益/航向等原门均未通过，formal_expansion=False；不启动5seed、不加预算追阳性。L2最接近经典但regular95.486%仍低于B0 96.875%，controlled67.5%低于三经典70%，不能只挑B1较弱成功率作优势。

接触/yaw/design失败谓词有交叠，不能直接当机制因果；regular接触证据缺189/244/82/0、yaw47/19/12/9、design7/10/4/7，controlled接触62/82/8/0、yaw30/12/32/34、design10/12/2/8。三个训练seed簇和同136案例重复需分清，不把408方法案例当独立环境。底层评价记录Native38/diff3，外层真正39及L2动作2已直接从加载模型/封装源确认，不作旧模型mask或全面Markov论断。

study_review.json、development_summary.csv及disposition.json给出核验数值、case swaps与明确停止该pilot扩训；不把工程成功或完整训练当核心方法就绪。第120轮必须深审研究价值/初始化与输入契约、隐藏控制和不可恢复任务历史、动态设计约束及接触/航向证据，再决定一个有根据的补充机制实验，同时清理确认冗余；不跳到另一次盲长PPO。新goal active，核心投稿方法优势和独立泛化仍未关闭。

第118轮收尾：同unit正常退出0/MainPID0/inactive，12/12全200k/400epoch/8000Adam、总2.4M，queue complete且冻结源hash一致；以下1.8M为较早进度快照。下一轮独立核全部工件/发展结果及原继续门，未据训练完成晋升五seed。

第118轮四臂三seed新学习已启动（2026-10-05，goal active）：

已注册route-action-pilot-v1.1：M3-zero/M3-route同39输入架构的零/路线信息对照，V6-route同信息六维虚拟残差，L2-route真正两输出腿F/H从零学习（原diff3接口补wheel0，非旧三输出策略mask）。固定seed1609/1610/1611、每臂100环境连续200k，总2.4M；原PPO、原100世界/stage/seed、原名义7kg/物理设计/任务门保持，本机制预试验不同时扩地形/改奖励。末200k唯一选模点，所有12run均纳入，不因首seed分数改预算或队列。

发展评价为新4700000..95 regular96、已公开controlled40，各末模型一次；三固定经典B0/B1/B1-route各两面板，共408经典+1632学习=2040评价，仍属发展而非独立测试。信息门为M3-route−M3-zero常规均值≥5pp、至少2正seed、controlled及平均物理/设计非退化；该门通过不自动扩训。正式扩展候选M3-route/L2-route需各seed保留三经典每个成功/物理/设计案例、两面板轨迹完整、controlled成功均值较最强经典≥5pp且regular Jψ≤B1+.05°；V6/zero对照独立报告。动作维数、rank/covariance及执行投影差异不能单独归因于物理先验；L2继承原两腿sigma，不制造完整分布等价。

工程4×1000CUDA steps、各20epochs/40Adam及500/1000post-update checkpoint过，物理/RNG/路线状态保存不变，L2原wheel请求/Actor轮残差每步严格0；独立重新加载8工件核source/维数/Adam/归一化/计数和初始配对。888×256旧v8正确状态库静态执行尺度另核，M3-zero/route相同，L2轮RMS0且lambda均值1、M3 .918/V6 .938，明确参考状态诊断不是实际训练分布/全协方差相同。初版误指27轮状态库，hash保护在任何model/学习前拒绝（0消费）；源码/proposal/错误日志保留，显式v1.1仅修正到29轮与source记录，不改科学条件或预算。

新经典结果regular成功B0/B1/B1-route=93/91/92，Jψ .454047/.351284/.310922°；controlled各28/40，Jψ1.183880/.966481/.922205°，全408完整且物理/设计通过。controlled增量数值空间30pp，5pp门不是上限矛盾，但不证明候选能达到；regular不设胜经典5pp门。工程/初始尺度/经典/独立admission的SHA已冻结trainer_contract。

正式GPU队列已由wheelleg-route-pilot-training-v1-1.service启动，独占.git/project-write.lock；本轮快照9/12完整、确认1.8M，M3-route/1611 pending、其partial消费见live progress。所有完整run为200k/400epoch/8000Adam、连续无强制物理或路线重置。首1609四臂regular29/2/54/90、controlled14/2/23/26，physical均96/40而设计有退化；不能以单seed判整体或更换主门。继续同unit到全12及独立原门核验，不静默重启、不挑模型；5seed/独立组合OOD/敏感性仍需新门，120轮方向深审清理不变。

第117轮固定路径参考实测与学习问题收敛（2026-10-05，goal active）：

由名义小角度y_dot≈v*psi、忽略惯性的kd*psi_dot+kp*psi≈kp*psi_ref取tau=(2+B1kd)/(.4+B1kp)=2.875s、lookahead=abs(command)*tau；psi_ref=clip(-arm*sign(command)*atan2(累计y,lookahead),±3°)，命令≤.05时0。B1原未裁剪PD加(.4+B1kp)*psi_ref后统一原±.3Nm请求限幅，经原过滤/投影执行。全部用同39包，真y仅发令后评价；名义近似不含接触/惯性/饱和，不当稳定/安全证明。参考3°不保证实测yaw≤5°。一次固定公式，不扫增益；原B1等价、双向/镜像/界自检过。

先登记再GPU216回合：原公开regular32×原B1/正确/反向；新增controlled40为五高度0.115/.16/.24/.30/.38、双向.7、左右单侧10/20mm，质量7、摩擦.8、无drive差/delay的完整笛卡尔网格，不按成绩挑案例。常规成功32/31/32、Jψ .242410/.215938/.309360°、平均最大侧偏15.462/9.728/21.544mm；controlled各28/40、Jψ .965475/.923436/1.013528°、最大侧偏均值41.630/39.899/43.400mm。全216轨迹完成且物理/设计均通过。高度分组均为4/8、4/8、8/8、8/8、4/8，路径改善没有改变控制网格成功率。正确参考丢regular4500029，yaw峰从4.998063°变5.351860°；保留原5°门。freshGPU的原B1由旧31→32为新回放边界变化，未为通过重跑或覆写旧分数。

独立review_route_feedback.py按全trace重算请求公式/float32输出、身份/源/载荷、物理/设计指标、原完整success/yaw/侧偏/分组和事前门，通过；unit退出0。fixed_law_continue_gate=False，停止此固定律晋升，不调tau/gain/cap追阳性，不以侧偏/Jψ改善替代任务主门，也不推出路线信息全部无价值。

下一块按计划进行有限新学习，而非更多手工路径律：拟注册三训练seed、同39架构的zero/route三维差模、同信息六维虚拟残差及真正二腿动作从零学习（区别固定mask），匹配原PPO/物理执行预算并显式记录初始化/动作域差异；先验二维接口和物理初始尺度及强经典新发展门可达性，再冻结预算/场景/末模型规则与继续门。信息干预收益和相对强经典收益分开，普通积分/降维不称算法首创。此矩阵尚未正式登记或运行，不据现有数据承诺学习优势；通过新预试验门才扩五seed/独立组合OOD/敏感性。原CPU/旧门/最终仍保留，下一深审清理120。

第116轮路线状态接口与CUDA学习工程（2026-10-05，goal active）：

新增独立RouteState（route_state.py），原冻结38输入/控制/物理不改，仅追加一个原包yaw/body vx/vy的已知零起点累计y，route与zero同39维架构；积分发生在VecNormalize之前，zero只删除输出特征、不改变原包/动作。终态使用本回合部分20ms长度得到旧回合terminal39，然后按底层自动reset/新课程包重置y和上一速度；各world独立。输入/信息和局限显式版本packet_route_y_v1，未知初始位置仍混淆，未宣称完整Markov、传感器认证或新算法。后续经典与学习对照获得同样路线信号；零特征为明确删信息消融。

验收：脚本检查异步done、部分终态长度、terminal与reset隔离、课程新速度和原动作/奖励保持；真实GPU两mode各1368actor步/32完整回合，10world全经历1→2→3（20切换），每步output-stage qpos/qvel/ctrl/task/controller/param逐值不变，原38/reward/done/terminal38一致。route最大特征66.025mm/32非零终态，zero全0；分别运行不当位级配对物理轨迹。

CUDA短训练仅工程：zero/route同初始化各1000steps，共2000，每臂2更新/20epochs/40Adam，actor/value/Adam在CUDA且有限；route新增列actor/value改变.010209/.011238，zero为0/0。39维归一化、权重、Adam和计数保存恢复逐值核，独立artifact_review重新加载两个工件并核载荷/source/counters/RMS通过。审核脚本首次Path类型错误的源/错误记录保留，修复只影响审核写hash，未重跑物理或学习。两unit正常退出0。短模型不晋升、不作为性能/收敛证据；恢复限于权重/Adam/RMS/计数，未保存物理/路线回合状态，不承诺同轨迹续接。

下一块：用已验证39接口推导、注册并检验一个同信息强经典路径/轮腿协调律与原B1，保持原物理/设计/全任务门，先核全高度及不对称发展可达性/真实可改善空间；随后才能登记三seed学习、必要机制/同维消融。原门和最终保护，五seed扩训/独立组合OOD与敏感性仍待，不能以本工程通过结束新投稿补充goal。下一集中深审/清理120。

第115轮方向深审与补充信号审计（2026-10-05，goal active）：

1. **是否值得继续：有条件继续路线状态与接触/执行约束协调，停止简单sector滤波晋升。** 固定禁轮的三seed改善不能证明重新训练二腿动作更优，更不能独立归因于差模先验；必须以后续从零学习消融区分。原三seed奖励塑形门仍失败，不回头扩大预算。当前强B1新32成功31，成功率可提高空间仅3.125pp，不能登记5/10pp成功收益门或削弱经典来制造空间；额外收益可以是预先定义且不退化的航向/路径精度、鲁棒性或样本效率，不能事后换主指标。
2. **方法是否得当：保留公平控制与完整任务判据；缺的是信息/任务机制和方法贡献。** sector只作用于延迟观测PD方向，并非内部滤波Nom或2kHz屏障；固定mask改变动作域和闭环轨迹，不能当新PPO方法。当前38包有yaw及机体vx/vy、无累计位置，已知零起点时可用历史积分恢复侧偏；未知初始偏移仍混淆。标量积分、历史输入和路径PD本身已有通用基础，不称创新，需具体协调推导、近邻区别及实际相对强基线收益。
3. **实际补充实验已完成：** probe_packet_odometry.py在已公开新发展32上固定三original末200k模型及强B1，GPU128/4、0学习；v_y=sin(yaw)*vx+cos(yaw)*vy，20ms梯形积分，仅用实际Actor包（含归一化clip逆变换），不读隐藏延迟/参数/接触/路线真值，估计不进入控制。四组最大非终态误差1.160/1.494/1.622/1.129mm，全部过事前5mm工程预算；只证明本理想仿真包、已知初位置和有限轨迹内信号可用，不证明实机定位、接触安全或控制改善。review_packet_odometry.py逐步重建46889采样、时间边界/身份/source/model/载荷/原success与yaw，通过；进程正常退出0。首次离线检查误用float64重算B1float32运算的约1e−10m差失败，修复实际dtype重建后保留原1e−12容差和5mm门，旧失败源与log保留，无物理重跑。
4. **下一块：** 独立版本39维接口（原38+累计y）、零特征同架构对照，验证episode reset/terminal/curriculum和估计不用时的控制物理身份；给全部经典与RL相同可得路线信号。再推导、验收一个无未来地形真值的名义路径/轮腿协调律并核全高度与左右不对称发展任务可达性、强经典可改善空间；之后登记三seed新学习和实际机制/同维消融，门通过才五seed正式、独立组合/OOD/物理敏感性。旧稿可写不替代新核心投稿目标，当前新方法贡献、完整新学习与泛化仍未齐。
5. **冗余清理：** 两份源码完全等价且无占用的再生bytecode删除17986B，源码和全部原始回合/模型/轨迹及独有失败不删；记录round115_cleanup.json。方向证据、限制和实验先后在round115_direction_review.json逐条记录；下一次集中深评/清理第120轮。

第114轮新补充目标执行／轮请求方向约束机制（2026-10-05）：用户授权按第113轮计划进行，goal active，每五轮审查/清理继续，下一次115。本轮先检验是否值得训练一个最小方向约束机制，不把旧材料就绪当新目标完成。

在新公开发展32案例（4500000..31、覆盖九类地形和低中高8/16/8）及三个事前固定original末200k模型上登记all/sector/legs_only与强B1；sector仅把与−.4*延迟Actor yaw−2*延迟Actor gyro_z相反的wheel请求设0，腿请求不变、仍经原rate limit/全输出投影。它只使用实际归一化/裁剪包可逆所得信息，无接触/路线真值或未来输入；不等于内部Nom滤波PD，也不保证2kHz非干扰/闭环安全。本次只做320/10固定模型发展评价，0学习，不宣称方法首创或独立测试。

结果：B1 31/32；三个seed all12/14/0、sector23/27/24、legs30/32/32。方向约束相对all均改善，但相对legs分别−7/−5/−8；物理各32/32，设计各条件31/32/32不变，接触缺失all16/18/29、sector6/5/7、legs全0。事前条件要求三个sector−all正、无额外物理/设计失败且至少两个sector胜legs；最后项失败，filter_training_candidate_gate=False。停止这个具体滤波候选，不扫阈值或将改善直接当PPO方法优势；仍不能由其失败否定整个协调/信息研究方向。

验收：probe_yaw_sector.py自检符号/腿请求不变/镜像，review_yaw_sector.py独立按source/model/payload/hash、32身份、每物理步计数、实际腿/关节/力矩界、原完整任务success、yaw汇总及配对门重算过；320全部完成，unit正常退出0、inactive。原生产源码/所有旧模型/旧门/封存集未变，全部结果与失败保留于wheelleg_warp/results/paper_recovery_20261004/yaw_sector_v1/。下一轮115深评应明确现在证据支持“轮通道请求有害、简单符号限制不足”，并判断训练轮通道/轮腿协调/接触路径信息哪个有足够可辨识增量，选择一个新机制后再登记三seed学习，而不是立即扩五seed长训。

第113轮北大核心目标与纯仿真补充建议（2026-10-05，用户确认只有仿真）：

- 用户要求至少以北大核心认可范围为投稿目标；材料就绪≠达到指定期刊录用标准。北大核心为期刊评价目录，并无统一PPO步数/场景数/种子数合格线（北京大学图书馆说明：https://lib.pku.edu.cn/publish/bjdxtsg/5jlhd/56cgfb/561hxqkym/index.htm）。目标刊尚未确定，不能承诺录用或2026目录资格。机器人官网投稿指南/收录页本次未返回正文，不当已核验门槛；该刊已发表导纳与RL融合文章的官方摘要有具体新控制结构、稳定性结论和多对照Baxter实验（https://robot.sia.cn/article/doi/10.13973/j.cnki.robot.250163），只作已发表案例，不推出所有稿件必须实机或该证明。
- 当前投稿竞争力判断：限定固定系统通道审计可起草，但原差模/PPO方法优势未成立、奖励塑形预试验失败、禁轮固定模型虽改善仍未稳定超强经典且设计有代价。纯仿真方法论文优先补“具体机制+公平收益+泛化边界”，仅增加地形/步数、换算法名或加历史不能证明创新。候选问题为不对称接触下受约束轮腿残差的航向/路径保持与通道分配，先与已有动作表示/残差/轮腿控制近邻辨别；具体方法尚未设计或证明，不把候选历史/接触/分配机制称为新算法。
- 必要训练内容（建议，未执行）：在0.115–0.38m固定目标高度内分层覆盖端点及中间高度，左右镜像障碍/坡/摩擦差、双向速度与入口扰动；由单事件扩到左右交替和坡接台阶，变化障碍位置/间距及预登记场景银行或可复现重采样。先以小幅单事件建立任务可达性、观测/执行和强经典基线，再课程增加组合；动态变高度/转弯和跳跃不自动进入原地面航向主线。
- 建议最小对照矩阵：新候选方法、普通六维虚拟残差PPO、针对实际贡献选择的同维/去机制学习对照，先三seed同预算预试验；B0/B1认真调参且信息/执行预算一致，无需学习训练。预试验原主成功及航向/接触等收益方向跨seed一致、无设计/旧能力代价隐藏且机制有证据后，正式核心方法建议五seed同预算；最多两项直接识别主张的消融各至少三seed。若主张物理差模贡献，同维非差模必须；若主张历史或自适应分配，使用去历史/固定分配消融。五/三种子是项目建议非刊物硬规定，不让纯PPO/更多算法挤占必要对照。
- 独立实验必补：新同域测试、未见地形顺序/间距组合、分别列出的温和几何OOD和质量/摩擦差/驱动差/15–20ms延迟压力，以及易场景能力保持。已读压力案例只能发展诊断，不重复冒充新独立测试，最终3000封存。报告每seed成功率/航向及横向误差/接触/停车/力矩/关节设计余量、学习效率和实际端到端墙钟；按训练seed和独立case识别重复层级，不能把大量相同案例重复当独立样本。
- 非训练必补：闭链运动学和虚拟力映射、动作子空间/可达域、真实执行器界与设计约束区别的明确推导；解释被测状态/延迟/历史与隐藏地形信息，全部基线同信息。校对同机器人CPU/GPU少量代表工况及求解步长/接触设置敏感性，区分数值影响与方法效果。纯仿真可研究，但不可声称实机泛化或由静态限幅推出全局安全；GPU后端实现不是控制方法贡献。
- 停止与实施范围：本轮仅评估/记录，不授权恢复旧失败分支或启动新长训练。先冻结一个可证伪候选机制与预试验协议，再决定长训练；若强经典已达天花板或候选三seed未产生稳定增量，停止该收益分支，调整问题/期刊定位，不加预算追阳性。只有仿真已记为真实条件，不要求提供实体平台作为所有工作前置条件。

第112轮复杂场景必要性评估（2026-10-05，用户新任务）：

- 结论：有必要定向增加泛化覆盖，但现有瓶颈不支持直接提高地形幅度并重开长PPO。当前论文是有限系统通道/执行与任务审计，已有学习、独立常规/压力/旧能力证据支持该限定主张；更广泛复杂地形控制主张需要另行登记训练与独立评价，不能从手动演示或新增地图推出。
- 覆盖事实：冻结v3后两课程含九类地形，常规坡1–3°、粗糙2–6mm、单台阶5–20mm，多级阶梯峰值可到24mm；OOD生成器坡3–5°/粗糙6–10mm/单台阶20–30mm，不等于已纳入全部独立证据。每stage/seed为固定100世界，回合内高度固定；v4单侧坡/非对称粗糙另有入口且部分过渡/宽度不同，不能当原v3训练或简单同难度加两类。
- 独立失败复核：channel_validation_v1强B1常规93/96、压力36/48。单侧25–30mm八案例均失败，但七个已完成且要求的legacy接触位满足；八个航向峰值8.97–94.15°均超过原5°门，物理/设计8/8，只有一个timeout/terrain未过。用analyze_reward_failures.py原flags只读重建一致；不能把0/8当几何不可达，亦不能把包络全过当任务成功。旧学习接触缺失与横向漂移也保留为另一类实际问题，不混为单一因果。
- 优先顺序（建议，尚未执行）：①在发展测试中分离单侧坡/非对称粗糙、入口左右偏移与接触恢复，核路径信息和同权限强基线；②验证单障碍成功后再增加坡接台阶、连续左右交替及障碍间距变化，分项报告接触/到达/姿态/航向/停车；③若目标升级为泛化控制，在新协议中扩大场景银行或预登记可复现重采样、平衡左右/高度/速度覆盖，并比较同预算原场景与扩展场景、至少三配对seed和独立保留集；④再逐级扩大温和OOD及参数组合，保留易场景能力。只加极端幅度会混合控制缺陷与分布外难度，收益需实测。
- 动态转向/回合内变高度可为后续运动能力任务；跳跃涉及离地与落地恢复，独立任务/课程/指标与回归，不直接并入旧地面直线PPO。几何、观测/权限、奖励/成功和物理模型需先一致；调整信息必须同样提供给经典对照。任何扩展均不覆写CPU/冻结GPU基线、旧64门或封存最终3000，不据已有压力结果将其再冒充新独立测试。
- 参考：Rudin等《Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning》，PMLR164（2022），https://proceedings.mlr.press/v164/rudin22a.html；证明四足课程训练路线有研究依据，不证明本机器人自动受益或可复用其训练时长。本轮只读分析与文档记录，无训练或场景源码修改。

第111轮论文写作材料就绪验收（2026-10-04）：完整中文实证初稿已具摘要/问题近邻/方法公式参数/配对学习/独立表/统计复现/讨论结论，六原出口由audit_paper_readiness.py逐条对应实际证据并核source/model/hash/数值/表行/引用/结构，paper_readiness_audit.json ready=True。1.2M三seed同信息学习、4472/78固定模型确认/压力/旧回归/强经典、全部负门/失败与图脚本保留；五轮90/95/100/105/110深评清理齐。可开始正式写作的主张为限定系统通道与任务执行审计，不是PPO/新PBRS/全局安全或录用保证，不能以仅负摘要代替数据。材料目标达到；作者期刊语言编辑属后续。内置compiler缺sandbox未确认PDF，明确保留排版限制；若继续新项目工作，下次115深评清理。

第110轮独立完成/深评（2026-10-04）：4472/78全source/原门/identity/physical/hash审计过，unit正常退出，22early保留。六regular禁轮−全均正、有限均值+62.5pp但设计多5；pressure成功81/288→210/288设计多3，旧回归97/168→167/168，分别报告非独立重复。强B1未被禁轮腿常规均值超越，不称新算法或全面安全。完整独立表图已入paper.tex，六出口实验与限定机制齐、完整贡献讨论结论与引用审阅未齐，goal active，不训练追阳性。清理两语义重复engine marker142B，完整值hash保留；下次115深评。下一项完成完整稿件论述/引用复现再全审就绪，编译器环境限制保留。

第109轮自动表/增量核验（2026-10-04）：53jobs/3068快照过，四个完整regular配对legs−all+57/+52/+37/+88伴设计变化+4/−2/+1/+1，六总体仍null；live3192继续同unit。新增只读自动CSV/table_report，161行保留early/None航向，压力汇总与因素不叠加，六模型齐才算主估计、4472/78齐才讨论整体，无训练显著性或连续压力域保证。草稿補表格分母/复现规则，下一轮110深评/清理，论文目标active。

第108轮快照复核/统计草稿（2026-10-04）：42jobs/2408记录原门、源/参数/物理及ledger快照过；三已齐模型regular腿−全+57/+52/+37，design变化+4/−2/+1，不作全六总体效应或训练显著性。实际2552仍运行，保持同unit观察。paper.tex补逐模型差/六模型平均、三seed簇、重复96非独立、压力8点/旧28/early航向不可算、source复现顺序与mask/预测限制，结果留待全数据。源结构检查过，编译器环境缺失未声称PDF成功。目标active，第110轮深评清理。

第107轮独立审计/方法草稿（2026-10-04）：21jobs/1204记录按原success、summary、CaseID/物理计数及载荷SHA复核，同字节ledger快照防运行中更新错配。original1609 regular腿−全成功+57但设计多4失败，不称无代价修复；全六模型效果仍未齐。压力单侧30mm案例4420404为真实timeout未到达，包络真/任务false；Yaw不完整留None，不补0或删失败。unit仍active，实际已完成1864；继续同进程，不因等待重启。paper.tex保存中文方法/协议/完整开发表初稿、独立结果留空并声明未就绪；内置编译因Codex sandbox执行文件缺失失败，未确认编译通过，仅结构检查。下一项完成独立及统计/贡献边界后补稿，第110轮深评清理。

第106轮独立通道评价启动（2026-10-04）：评价器和原基线/辅助源在新评分前冻结，4472/78jobs/0训练，常规96/压力48/旧28分开，压力六因素单列，固定六末模型四masks及B0/B1，无模型/case选择或隐式重试。unit已核active/PID187039，B0首172回合文件hash/physical计数过，正在B1。当前结果不作全部模型结论；下一项核同unit真实进度/终态，完成后独立审计通道/压力/非退化，再限定可写主张。旧门最终保护、论文目标active，第110轮深评清理。

第105轮集中深评与独立确认登记（2026-10-04）：最差Phi1610固定模型全96四mask干预成功all6/legs93/wheel8/zero93，terrain缺74/0/73/0，物理全过；只证明该模型有限公开案例里禁用wheel请求的正闭环效应，改变请求子空间与反馈轨迹，不是同权限算法或Phi训练因果。奖励收益分支不扩训练或调幅度，论文六项仍缺贡献/独立/压力/稿件；方向收敛到执行/接触任务/通道审计，不称新PPO/PBRS。已事前登记channel_validation_v1：全部六末模型×四0/1mask与固定强B0/B1，fresh常规96/六因素压力48/原公开28，4472评价、0训练，主报告legs−all成功效果/三seed聚类，三类分开不选case/model，旧回归不独立；旧64门/3000封存，尚无新评价。下一项先冻结评价源再执行有限确认，继续可写主张审计。清理五重复日志51418字节，保留全部独有失败/模型/版本；下一次110深评清理，目标active。

第104轮固定轨迹诊断（2026-10-04）：探索性选最大退化seed1610、末两200k模型/全部新96，192固定回放完整且无更新；root中心进度平均|y|原3.765/Phi7.293cm、最大|y|5.899/11.636cm，移动态中心前meanYaw1.0567/2.2191°，方向校正偏移95/96与96/96为正。freshGPU成功仍65/6、接触缺26/74，Jψ数值不同不覆写原结果。需要terrain的81组接触过/缺仍有位移重叠，body代理/50Hz时序不当轮胎几何证明，Value预测不当MC或训练因果。源/身份/时间/动作/终态/RMS冻结核过，可复现图保存。下一轮105深评可得路线/历史与固定腿轮通道受控检验，定位偏置再注册，停止奖励收益分支及旧门封存保持，论文目标active。

第103轮失败重建（2026-10-04）：六run末96共576方法案例的原success全部按完整判据重建一致；缺terrain证据raw45/26/73、Phi52/74/38，仅该项失败34/25/49对42/66/32。seed1610 Phi6成功遍及低中高组均退化、丢59得0，非仅低高度设计问题；六run速度/停车/尾速/高度门0失败，物理全过。Phi谓词没有覆盖主要接触失败，不足以解释全部任务；重叠标志非互斥因果。bootstrap只有预测值，无MC目标和完整状态，不能宣称Critic校准失效。来源/重建过、结构图保存。下一项最大退化seed1610固定末两模型全96只读轨迹检查（探索性选seed），核横向/接触时机及动作/值，不作随机Critic校准或调参；之后再限定可写主张和新独立验证。第105轮深评清理，原门封存，目标active。

第102轮三配对预试验结论（2026-10-04）：六run完整1.2M，各200k/400epoch/8000Adam及初始/60检查点/真实bootstrap/Phi边界/末96身份与原summary通过独立审计，队列正常退出。Raw44/65/7、Phi38/6/48成功，总116/288对92/288、均值差−8.3333pp，仅1正seed；设计多3失败、物理全过，Jψ均值差+.0140142°。预注册继续门false，停止奖励收益分支，不扩2M、调幅值/预算或用旧门最终追阳性。保留种子异质性、所有工件/失败、可复现图，288不是独立场景。下一项核失败结构和价值/历史状态机制及新的可写主张边界，不能以完整负摘要替代论文就绪；第105轮深评清理，目标active。

第102轮首奖励配对（2026-10-04）：两臂1609各200k完整/400epoch/8000Adam，初始化hash相同；新96 raw44/Phi38，成功差−6.25pp，Jψ1.6352058/1.5540850°虽略好但设计93/90更差、物理均96。不能以yaw代替成功主指标。只读审计20检查点/counter/真实bootstrap/Phi边界/源载荷/96身份及模型有限值过；当前2/6、1/3配对、确认400k，整体门未判定。剩余四run按1610/1611 original→potential顺序unit已启动，错误即停、不重试，保持总1.2M和固定200k末模型规则。旧门最终封存，论文未就绪，第105轮深评清理。

第101轮收尾补核：首original/1609实际连续200k已完成，unit正常退出0、400epoch/8000Adam，固定末权重新96为44成功/Jψ1.6352058°、物理96/设计93。随后启动同seed potential对照，确认unit正在运行；只有单臂开发结果，尚无奖励效应或独立门结论。当前1/6完成、确认200k消耗，保持配对初始hash、预算/源冻结和不挑中间权重。

第101轮连续trainer冻结/首预试验（2026-10-04）：on_rollout_start保存上一轮完成训练的权重，末保存置于learn返回后，采样与训练计数分开；记录真实更新前bootstrap/done、非终态Phi/折扣边界，保存不改物理/Phi/RNG。连续v1/v2各12k工程通过，v2区分28正常回合reset与0强制重启，六checkpoint counter/source/真实bootstrap/部分边界核过，trainer_contract已冻结。首original/1609/100世界200k用户unit已启动，检查到采样125k/训练120k（240epoch/4800Adam），不做中间选模或旧门最终评估。进程观察超时不重启，真实中断记不完整/消耗和真实工件，不能静默物理恢复。下一项核200k末模型/新96并推进同seed配对；六run1.2M预算、原门和未完就绪条件保持，第105轮深评清理。

第100轮集中深评与有限预试验（2026-10-04）：六项论文就绪仍未齐；新奖励的接口/代数/CUDA短训过，不等于收益。近邻HPRS2025已覆盖约束机器人PBRS/PPO等，不能称新算法；学习变化还可能是近似Critic/价值目标效应，Phi仅针对所列历史约束，不解释全部失败。先冻结有限M3 raw/potential两臂×三seed×200k（上限1.2M），100世界原38/超参/同初始化，连续run不计划物理恢复；固定最终200k新公开96比较完整成功，对raw均值≥5pp/至少2正seed，物理设计不退化/yaw≤raw+.05°才再评更长研究，不挑中间best。新公开96经典B0/B1均94成功、物理设计全过；对经典最多2.0833pp，不能混成5pp门或弱化基线。新方案草案及192经典回合已归档，学习尚0；下一项先完成post-update检查点/末端Phi与bootstrap日志及源码冻结。任何中断保留工件和消耗，不能静默物理reset作目标等价；旧门/最终封存。清理8份确认重复99031字节，保留独有失败/模型/原轨迹；详见round100_direction_review.json/round100_cleanup.json，下一次第105轮。

第99轮CUDA短训工程（2026-10-04）：原/势函数M3两臂同seed1609/10世界/原50步-batch250-10epochs及初始权重/world设置，各6000+恢复6000（共24k），实际CUDA actor/value/Adam/物理、各240epoch/480Adam、有限值与权重变化过，最终全stage3；势函数实际1041非零转换/20完整回合折扣舍入界过。运行时权重/Adam/观测RMS保存恢复逐值过，独立四个工件重新加载均核观测/奖励RMS与模型计数过。只做工程，不评价学习优势或晋升权重。物理/Phi/RNG分段重启，完整回合恒等不能覆盖被丢弃的部分回合；新正式研究恢复必须明确gamma^L*Phi与bootstrap边界/配对分段或真实状态恢复。下一轮第100轮深评、方向/协议决定和确认冗余清理；原门/最终封存，论文就绪未达。

第98轮在线势函数工程（2026-10-04）：可选FailurePotential保持原38输入/物理/控制/全部门和原episode.r，仅重排奖励；128固定策略公开首回合和128重置后单步过，每transition奖励处理前后物理与观测逐值相同、RMS/策略无更新，不能作两条独立轨迹或学习效果。折扣恒等精确误差3.60e−15，PPOfloat32约3.94e−8在实测舍入界内，所有50种rollout对齐/非终态边界及终态/重置过。原生float64与VecNormalize即使norm_reward=False仍转float32的两个精度检查失败来源/日志和部分B1数据保留，按实际管线修复后严格过，不改科学门。新GPU回放Jψ数值不同，不声称位级复现或包装改善。下一项同初始化同世界的原/塑形实际CUDA短PPO与保存恢复工程验证，再新独立协议/三学习seed；旧门最终封存，第100轮深评清理，目标active。

第97轮奖励时序审计（2026-10-04）：全部128原公开居中回放中49失败，只有8条提前约束标志（7低高度设计域/1高高度yaw），其后dense回报均正且终奖晚72～226步；超50步采样窗口不等于Critic不能bootstrap，不能作全部失败或训练因果解释。简单前移−10改变折扣目标。独立离线测试中间Phi=−10*历史违规、端点0，r'=r+γPhi_next−Phi_current，128折扣恒等误差<5e−15；同步V'=V−Phi时TD恒等，但原近似Critic未必能表示隐藏Phi，学习效果未证。势函数方法不是新算法，不用塑形未折扣回报代替原成绩。下一项同原38信息/物理/全部门的在线端点和reset/rollout无更新配对检查，再注册新学习干预；旧门最终不重用，第100轮深评清理，论文目标active。

第96轮期限坐标检验（2026-10-04）：仅一次事前注册的名义轮心+75mm修正，remaining=max(1.5−direction*(body_x+.075),.05)，原B1/参考5°/请求±.3Nm/全部成功门保持；正确组要求双向12/12，失败不扫参。36完整且物理/设计全过，三组4/8/4，正确仍正向2/6反向6/6、yaw峰4.386684°、Jψ1.542671°，未过原门，结束该期限参考修正分支，不晋升/不推全部权限不可达。源hash/原summary/证据与逐步发令核过；初次核验忽略float32加法的精度失败脚本和日志保留，按原运行运算重建严格检查过，不改科学门或重跑物理。下一项分析已有居中初始化公开回放中历史不可恢复违规后的回报/反馈时序，再注册同信息单因素干预；避免人工移位或额外障碍信息的因果混淆，旧门最终不重用，论文目标active，第100轮深评清理。

第95轮集中深评与方向调整（2026-10-04）：五轮242完整公开诊断、无新学习；瞬时侧移混淆和具体反馈的几何分离已证，但±120mm初始化不属旧居中训练分布、期限参考理想位置/障碍前边多于旧Actor，不能解释全部旧失败或宣称同信息学习优势。新贡献、三学习seed、独立测试/压力/旧能力和完整稿件仍未齐，目标active。文献补核官方摘要确认历史/循环策略及轮腿导航已有，不把加位置/记忆/参考作新算法。候选方向收敛到信息/任务/实际执行一致性的实证审计；下一项只做一次注册的名义轮心+75mm期限坐标修正，原增益/门不变，失败不扫参；之后先冻结不读未来地形的可得路线/历史契约，强经典与全部RL同信息，机制/公开全高度检查后再新独立注册。保留旧门失败与最终封存。删除6份确认重复载荷61375字节、保留路径hash及全部独有失败，详见round95_direction_review.json/round95_cleanup.json；下一次第100轮深评清理。

第94轮有界期限参考（2026-10-04）：公开台阶几何推导body目标1.8−.25−.05=1.5m，psi_ref=clip(atan2(-direction*y,max(1.5-direction*x,.05)),±5°)，原B1加(.4+kp)*参考并保持原±.3Nm请求，先注册后运行，不扫增益或改门。36回合全部完整、原物理/设计36过；原B1/正确/反向参考4/8/4成功，正确参考反向6/6、正向2/6，实际yaw峰≤4.425272°，Jψ均值1.539300°没有经典航向优势。逐步参考/动作和原summary独立核过，仅局部构造任务恢复，不晋升通用方法。已有实际初始轮心相对body_x约+75mm，body期限不等于两方向同一轮接触期限；作为下一核查线索，不当唯一原因证明。第95轮集中深评文献/研究价值、信息契约及方向并清理确认冗余；旧门/最终仍封存，论文就绪未达。

第93轮实际轮轨迹诊断（2026-10-04）：复用第92轮同36条件/不改控制，只读记录533497个2kHz物理样本，原物理/设计36过、三组任务仍4/12。正确反馈漏接触轮在纵向AABB重叠的整个采样区间仍横向分离20.906～24.357mm，原B1约78mm、反向约120～127mm；被动链轮心终态与独立CPU forward差<1e-7m，原接触位和计数/源码/summary/离线间隙重算过，图已生成。支持“这条固定反馈来不及进入接触通道”，不证明整个权限域不可达/连续时间碰撞结论/学习方法贡献。首次launch包装缺省inputs初始化失败的源码和日志保留，未执行step；修复调用语义后运行。下一项由公开几何和原航向门推导有明确纠偏期限的最小路径参考，先受控经典机制证据再新PPO注册，第95轮深评清理不变。

第92轮受控侧向干预（2026-10-04）：固定增益/同原差模权限的正确反馈、原B1和反向同增益对照共36回合，全部完整、物理/设计36过，三组任务均4/12、仅中心成功。理想横向里程计反馈降低终态侧移至约37mm，但body到台阶前边时仍80～85mm，缺一轮terrain证据；原B1/正确/反向组Jψ .115176/.901258/1.039974°，不晋升。源hash、原summary、计数、请求和轨迹由只读verify.py核过。静态台阶半宽160mm、轮心±150mm、轮胎半宽27.5mm，名义两轮有横向重叠要求body侧移约≤37.5mm；实际轮心/姿态/接触时机尚需记录，50Hz body代理不能作动态碰撞证明。下一项先核轮轨迹和纠偏期限，不扫增益、不启动长训练；旧门/最终不重用，第95轮深评清理不变。

第91轮机制检查（2026-10-04）：公开固定台阶场景中，−.12/0/+.12m侧移的实际其他初始状态/原38包精确相同、同shape策略输出相同，但原B1中心两重复成功、侧移四重复各缺一轮terrain接触，物理设计六过。构造了任务相关初始混淆，不解释全部旧失败或证明最优纠偏；20ms replay/批量NN零差的3次失败检查均保留，未松容差、积分后位级等同未宣称。下一项同物理公开纠偏干预与新信息契约检验，旧门最终不重用，不启动长训练，论文就绪未达。

## 新持续目标：推进到论文可写就绪（2026-10-04，第90轮起）

用户最新要求继续工作直到可以写论文，并每五轮客观深评、方向调整和冗余清理。旧18M方案与原独立门的失败是保留事实，不等于本新目标完成，也不重新打开旧门追阳性。新目标的出口是可审阅的论文材料与明确可证实主张：文献重合/贡献边界清楚，模型/可得观测/约束契约一致，强经典和RL公平对照、至少3个训练种子及新的独立测试/压力/旧能力保持证据完整，机制解释经受控检验，推导/图表/全部失败/统计/源码数据足以起草；不以“可以写一份负结果摘要”或短更新通过替代，不承诺录用。

当前文献已经覆盖残差学习、动作空间、监督初始化和状态相关边界，单纯缩维、换PPO或加一项观测不是创新证明。先恢复研究问题与机制证据，暂不启动新长训练或堆网络。候选方向为“观测/任务几何和反馈时序如何影响受约束轮腿残差学习”；这是待验证问题，不是已成立方法贡献。

第90轮探索诊断在运行前固定B1和原三方法最终2M/seed1610、全部原32公开开发场景，仅加50Hz读出，原物理/奖励/控制保持2kHz。128回合B1/M3/V/B2成功32/17/12/18；缺地形证据病例最大横向偏移均值约7.5～8.2cm，比有证据组高，原38观测有yaw/v_y但无绝对横向位置。历史不可恢复约束违规至终奖延迟约1.4～2.4s。该关联未证明原因，bodyy不是精确wheel/slab交集，50Hz首次违规时间有20ms括界，且没有真实训练成功标签；旧模型GPU回放不保证位级一致。这些新增公开诊断不算旧100次选模，不改变旧结论或旧研究门。

下一轮先检验H1横向状态混淆：在合法公开号况下构造不同侧移但相同可见包/动作历史的状态，检查模型/观测与接触路径区别；再做同物理/同预算公开干预，不能只凭散点认定原因。H2是历史违规终局反馈的信用分配，需时序/受控反馈检查，GAE代数不是因果证明。支持机制后才冻结一个最小新方案及全新选择/门/最终命名空间，强B1和六维基线不能弱化；所有RL同奖励/观测/约束，不改PPO损失。

第90轮为新目标首项实质工作与起步深评，累计项目轮次保持；以后95/100…每五轮深评/删除可证重复。已清理5份闭合旧任务/tmp重复1094字节，旧权重/原门/CPU基准/独有失败均保留。证据paper_recovery_20261004/round90_direction_review.json、public_path_reward_diagnostic/analysis.json及只读脚本diagnose_paper_recovery.py。论文就绪尚未达到，目标持续active。

第89轮交付完成（2026-10-04）：本轮原注册方案已完整执行并按研究门失败出口结束，逐项审计见completion_audit.json。9项18M、一次64门＋原28回归和负结论/基线/模型/失败/周期审查清理证据齐全，停止当前优势分支，不追加条件后续或使用最终3000。此处完成不指论文已发表或所有后续研究结束，新假设需独立注册，已用门不重开。

第88轮原独立门结论（2026-10-04）：未通过。九项原2M/100选择完整结束，锁定后704独立门/84旧回归回合完整，原评判和704条Jψ数学重算一致。B0/B1成功64/64；M3三seed53/60/53、Jψ均值.753058，对强B1 .230945没有原15%且.05°优势，且丢26个B0策略案例；旧28成功26/27/27、21项roll/pitch非退化失败。按原停止门结束当前差模优势分支，报告负结果，不扩条件五种子/消融/最终3000、不重开或换主指标；保留CPU原基线和全部失败。详见架构审计顶部及independent_gate_verification.json。

第87轮完整验收（2026-10-04）：九项原2M/100选择正常完成，合计18M，900身份/场景/权重RMS hash/原best及有限性逐项核过。全开发M3/V/B2成功89/89/87策略×案例、平均Jψ .785511/.677342/.646784，仍同32案例重复、非独立门结果。原9模型和B1已锁定，下一项一次64研究门及M3原28回归；最终3000尚封存，原门通过才条件后续，失败不重开。

第86轮最后电机百万（2026-10-04）：B2best仍40k31/32/.484259，last1M15/32/.970717，500k20例恢复未稳定保持；best丢1个B0成功但速度/到达32过，2000epoch/40000Adam有限。同1M三方法best29/31/31，仍无经典联合优势。当前8/9完整，最后原预算继续再原一次研究门，90轮深评清理，独立门/最终未开。

第85轮深评（2026-10-04）：当前8/9完整，M3/V完整开发均89/96策略配对、平均Jψ .785511/.677342，原差模优势无开发支持但非独立门结论。最后B2半百万20例恢复、best40k31/.484259有71训练回合记录，不能归空日志早期点；仍未超经典，异步r/l窗口非成功率/因果/稳定证明。完成最后原预算和一次门，通过才条件后续，失败结束当前优势分支，不扩预算或换主张。清理16份相同临时日志4723字节，下次深评90轮，原门先完成则及时按出口处理。

第84轮最后电机半百万（2026-10-04）：B2best仍40k31/32/.484259，last500k20/32/.825468，比200k8恢复但未超经典；物理32设计31，11地形/1设计/1姿态失败非互斥。策略/Adam/RMS有限，1000epoch/20000Adam正确；三方法同500kbest28/31/31，不混全预算或更换主贡献。当前8/9完整，第85轮深评清理，原预算继续且独立门/最终未开。

第83轮第三种子三方法同二十万（2026-10-04）：B2 best40k31/32/.484259，last200k8/32/1.392998；best丢1个B0成功但速度/到达32过，400epoch/8000Adam有限。M3/V同预算best28/.956787与31/.663466，电机早期较好但三者仍无经典联合优势，不更换主贡献。当前8/9完整，继续最后原预算，85轮深评清理，独立门/最终未开。

第82轮实际进展（2026-10-04）：恢复V/1611完整2M/100选择正常退出，best20k31/32/.663466，last19/32/.948188，best丢1个B0成功但速度/到达32过；4000epoch/80000Adam有限，物理分段无训练/优化次数缺口仍披露。M3/V三政策各89/96重复同32案例，平均Jψ .785511/.677342，三个开发配对方向均未支持M3，非原独立门结论。01:00:49新建B2/1611 CUDA，首20k31/32/0.542310真实更新通过；当前8/9完整，独立门/最终未开，第85轮深评清理。

第81轮跨重启恢复（2026-10-04）：V/1611在1040k开发评估期间收到SIGINT保存，随后主机重启，瞬态单位消失；非正常完成或EPA错误。核1040k权重/Adam/RMS、2080epoch/41600Adam和51评估、原7完整运行后，原--resume实际恢复并补pending评估；1060k2120epoch/42400Adam验证通过，无丢弃训练/预算重放，但物理回合重置和中断评估重做须披露。原百万best31/.663466、丢1个B0成功，仍无经典优势。当前7/9，生产源码/主门未改，继续剩余两项，85轮深评清理。

第80轮深评（2026-10-03）：当前7/9完整，M3 full29/31/29、平均Jψ .785511仍弱于经典；89/96为三策略重复同32案例配对数，非96独立场景或原门结论。第三V近期回报3.611→10.873而开发仍12且Jψ变差，异步无标签窗口不能作成功/因果或收敛证明。做完剩余两个原对照及一次门，通过才条件后续，失败结束优势分支，不加预算或改贡献。清理12份相同临时日志3668字节，下次深评85轮。

第79轮第三种子虚拟六维半百万（2026-10-03）：Vbest仍20k31/32/.663466，last500k12/32/1.929113；物理32设计29，18地形/10姿态/4原接触/3设计失败非互斥。策略/Adam/RMS有限，1000epoch/20000Adam正确；静态7/5328均值出界不代表滚动裁剪/因果。M3同预算best28/.956787，V早期较好但仍无经典联合优势。当前7/9完整，第80轮深评清理，原预算继续，独立门/最终未开。

第78轮第三种子同二十万（2026-10-03）：Vbest20k31/32/.663466，last200k12/32/1.456551；best丢1个B0成功但速度/到达32过，400epoch/8000Adam有限。M3同预算best28/.956787，V早期更好但双方仍未超经典联合要求，不将M3全预算混比或改主张。当前7/9完整，继续原2M及最后B2，80轮深评清理，独立门/最终未开。

第77轮实际进展（2026-10-03）：M3/1611完整2M/100选择正常退出，best840k29/32/.789470，last25/32/1.109091，best丢3个B0成功但速度/到达32过；4000epoch/80000Adam有限。三个M3 fullbest29/31/29、平均Jψ .785511，89/96为重复同32例的策略×案例，非96独立场景或原门结论。19:24:52新建V/1611 CUDA，首20k31/32/0.663466和真实更新通过；当前7/9完整，独立门/最终未开，第80轮深评清理。

第76轮第三种子百万（2026-10-03）：M3/1611 best840k29/32/.789470，last1M21/32/.970716；新best丢3个B0成功但速度/到达32过，2000epoch/40000Adam有限，当前物理32设计29。真实晚期改善仍未超经典联合要求，不用不同预算M3均值代替完整配对；当前6/9，继续原2M和V/B2，80轮深评清理，独立门/最终未开。

第75轮深评（2026-10-03）：当前6/9完整，两组全预算三方法均无经典联合优势；第三M3半百万28最佳/13当前，不混入完整配对平均或替代原独立门。 最新选模840k已29/.789470，1680epoch/33600Adam核过，仍未超经典，不沿用最好仅早期的旧判断。新核第三M3日志20k0回合、近期回报6.860→10.306和开发8→13恢复，但非稳定/因果/全训练成功证明。完成原第三种子配对及一次门，通过才条件后续，失败结束优势分支。清理12份相同临时日志3652字节，下次深评80轮。

第74轮第三种子半百万（2026-10-03）：M3/1611 best仍20k28/32/.956787，last500k13/32/1.616612，较200k8例恢复但未超经典；物理32设计31，16地形/2原接触/1姿态/1设计失败非互斥。策略/Adam/RMS有限，1000epoch/20000Adam正确，静态2/2664均值出界不代表滚动裁剪或因果。当前6/9完整，第75轮深评清理，原预算继续且独立门/最终未开。

第73轮第三种子二十万（2026-10-03）：M3/1611 best仍20k28/32/.956787，last200k8/32/2.176480；best丢4个B0成功但速度/到达32过，400epoch/8000Adam有限，当前物理/设计32过但地形22及原接触5失败非互斥。当前6/9完整，第三种子完整与配对仍待，继续原预算，75轮深评清理，独立门/最终未开。

第72轮实际进展（2026-10-03）：B2/1610完整2M/100选择正常退出，best20k30/32/.543594，last18/32/.873510，best丢2个B0成功但速度/到达32过；4000epoch/80000Adam有限。第二种子三方法同完整预算best31/30/30仍无经典联合优势。队列18:46:10新建M3/1611 CUDA，首20k28/32/0.956787、40epoch/800Adam通过；当前6/9完整，独立门/最终未开，第75轮深评清理。

第71轮第二种子三方法同百万（2026-10-03）：B2 best仍20k30/32/.543594，last1M14/32/.963992，500k22例恢复未稳定保持；best丢2个B0成功，速度/到达32过，2000epoch/40000Adam有限。M3/V同预算best31/.609470和30/.605179，仍无经典联合优势。当前5/9完整，继续原2M和第三种子，75轮深评清理，独立门/最终未开。

第70轮深评（2026-10-03）：当前5/9完整，第二种子M3/V全预算与三方法半百万仍无经典联合优势；B2从7恢复22例，约束“所有策略单调退化”的过度结论。新核B2保存点20k日志0、近期100回报3.562→12.544与开发7→22同向，但无场景成功标签，非收敛/因果证据。继续原9项和一次门，通过才条件后续，失败结束优势分支，不追加预算或换主张。清理14份相同临时日志4256字节，下一次深评75轮。

第69轮第二种子电机半百万（2026-10-03）：B2 best仍20k30/32/.543594，last500k22/32/.822211，从200k7例恢复但未超经典；物理32、设计29，3设计/7地形/1姿态失败非互斥。策略/Adam/RMS有限，1000epoch/20000Adam正确；同500k三方法best31/30/30仍无联合优势，当前5/9完整，70轮深评清理，原预算继续且门控/最终未开。

第68轮第二种子三方法同二十万（2026-10-03）：B2 best20k30/32/.543594，last200k7/32/1.531256；best丢2个B0成功但速度/到达32过，400epoch/8000Adam有限。M3/V同预算best31/.609470与30/.605179，三者均弱于经典32/.272426，不改联合主张。当前5/9完整，继续B2原2M及第三种子，70轮深评清理，独立门/最终未开。

第67轮实际进展（2026-10-03）：V/1610完整2M/100选择正常退出，best20k30/32/.605179，last12/32/1.252878，best丢2个B0成功但速度/到达32过；4000epoch/80000Adam有限。与M3同完整预算best31/.609470仍无经典联合优势。队列18:08:07新建B2/1610 CUDA原电机对照，首20k30/32/0.543594、40epoch/800Adam通过；当前5/9完整，独立门/最终未开，第70轮深评清理。

第66轮第二种子百万（2026-10-03）：V/1610 best仍20k30/32/.605179，last1M15/32/1.140305；best丢2个B0成功但速度/到达32过，2000epoch/40000Adam有限。M3同预算best31/.609470，末点10/1.305717，原失败优先选模和经典联合门不变，仍无优势。当前4/9完整，继续原2M配对，70轮深评清理，独立门/最终未开。

第65轮深评（2026-10-03）：当前4/9完整，第二M3全预算与V半百万仍无经典联合优势。新核6保存点：两方法20k均40epoch/800Adam但训练回合日志为空；后续近期训练回报与开发成绩不完全同向，窗口仅r/l且异步，不能推训练成功率或退化因果。继续原9项及一次门，原门通过才条件扩展，失败停止优势分支，不改主指标/运行中奖励/场景/预算。清理12份相同临时日志3990字节，下一次深评70轮。

第64轮第二种子半百万（2026-10-03）：V/1610 best仍20k30/32/.605179，last500k11/32/1.281248，物理/设计32过但20地形/3原接触/1姿态失败非互斥。策略/Adam/RMS有限、1000epoch/20000Adam正确；静态888状态2/5328动作均值出界不代表实际训练裁剪率或因果。M3同预算best31/.609470，仍无经典联合优势。当前4/9完整，第65轮深评清理，原2M继续，门控/最终未开。

第63轮第二种子同二十万（2026-10-03）：B2-V/1610 best20k30/32/.605179，last200k10/32/1.633070；best丢2个B0成功，速度/到达32过，400epoch/8000Adam有限。M3同预算best31/.609470，成功更多而Jψ略高，均弱于经典，不改联合主张。当前4/9完整，继续V原2M及B2配对，65轮深评清理，独立门/最终未开。

第62轮实际进展（2026-10-03）：M3/1610完整2M/100选择正常退出，best20k31/32/.609470，last17/32/1.039351，best丢1个B0成功，速度/到达32过；4000epoch/80000Adam有限。队列17:29:44新建B2-V/1610 CUDA，首20k30/32/0.605179，真实40epoch/800Adam通过，仅早期结果。当前4/9完整，继续原配对预算，独立门/最终未开，第65轮深评清理。

第61轮第二种子百万（2026-10-03）：M3/1610核50个checkpoint，best仍20k31/32/.609470，last1M10/32/1.305717；best丢1个B0成功但速度/到达32过，当前物理32、设计30、地形证据失败20。策略/Adam/RMS有限，2000epoch/40000Adam正确，仍无经典优势或收敛证明。继续原2M及配对，完整3/9，65轮深评清理，门控/最终未开。

第60轮深评（2026-10-03）：首种子完整三方法与第二种子半百万仍未支持联合优势，当前3/9完整。224条原记录独立重算Jψ均值误差0、公开目标一致、完成失败例全部计入，原始RMS亦弱于经典。继续原注册9运行和一次研究门，条件正式/消融/最终沿原门；失败停止优势分支。B2分段后少100Adam优化暴露、EPA根因未修复须披露，不扩预算/指标/运行中机制。清理8份字节相同临时日志2250字节，下次深评65轮。

第59轮第二种子半百万（2026-10-03）：M3/1610 best仍20k31/32/.609470，500k15/32/1.264741，物理32、设计30，地形14/姿态5/原障碍3/设计2失败非互斥。策略/Adam/RMS有限，1000epoch/20000Adam正确，静态888状态均值未出界不构成实际采样/退化因果结论。生产源码及主门未改，完整3/9，60轮深评清理，门控/最终未开。

第58轮第二种子阶段（2026-10-03）：M3/1610完整20万检查十个checkpoint，best20k31/32/Jψ .609470，last200k19/32/.911860；best唯一高位mixed例偏航6.315°超门，物理/设计/地形检查通过。原速度/到达32过，策略/Adam/RMS有限且400epoch/8000Adam符合预算；仍无经典优势，继续同一2M，当前3/9完整，门控/最终未开。

第57轮实际进展（2026-10-03）：B2/1609完整2M/100选择已退出0，best20k26/32/.912500，last12/32/1.093668；分段恢复后4000epoch/79900Adam有限，100次优化差异单列。首种子完整三方法均弱于经典；当前3/9完整，队列16:51:22新建M3/1610 CUDA，首20k31/32/0.609470，实际更新40epoch/800Adam。继续原三种子与研究门，门控/最终仍封存；独立核查脚本支持已审计中断计数，生产源码未变。

第56轮异常恢复（2026-10-03）：B2/1609在1782700步因EPA horizon24警告后GPU溢出停止，队列正确暂停。已核中断weights/Adam/RMS和89评估，按原--resume保留收费预算恢复，物理episode重启；1800k实际评估7/32/1.185480，3600epoch/71900Adam有限，比未中断少100小批次，必须披露分段影响。未改碰撞容量/物理/奖励/源码，故障几何未复现，不声称根因已修复；监督已接管恢复任务，研究门/最终未开。

第55轮深评（2026-10-03）：同160万预算M3/B2-V/B2最佳29/28/26成功，Jψ .957592/.763383/.912500°，均弱于经典32/.272426°；B2有限性及3200epoch/64000Adam预算核过，完整仍2/9。完成原注册三种子与一次研究门，不改运行中场景/奖励/结构或追加预算；正式扩展服从原门，失败结束优势分支。删除10份字节相同关闭临时日志3378字节，下一次集中评判第60轮。

当前场景覆盖说明（2026-10-03，第54轮）：活动GPU协议训练九类地形，常规几何为1～3°坡面、2～6mm粗糙路、5～20mm全宽台阶和115～380mm固定回合高度；后期冻结15类×200包含九类同域地形、保留组合与五因素压力。当前freeze最终九类采用development分布，未纳入ood生成器的3～5°坡面/6～10mm粗糙路/20～30mm台阶，单侧坡/非对称粗糙路及跳跃也未纳入本次PPO训练。后期压力不等于全部极端几何；保持已冻结实验，原研究门通过后按原终评报告限定鲁棒性，新增范围需独立注册。


第50轮深评（2026-10-03）：已完成M3/B2-V首种子完整预算，仍弱于经典；B2早期也未显示联合优势。仅继续完成注册三方法三种子以验证，不追加预算/场景/结构或奖励调参；原门失败停止优势分支并报告负结果。已清理可证重复临时日志，细节见架构审计顶部。

第49轮实际进展（2026-10-03）：B2-V/1609完整2M/100选择已正常退出，best1220k28/32/Jψ .763383，final15/32/1.209154，best丢4个B0成功但速度到达32过；同完整2M M3best29/32/.957592。M3成功更多、yaw更差，经典32/32/.272426仍更强，不能挑单指标宣称优势。队列16:11:14实际新建B2/1609 CUDA原电机6维对照，首20k26/32/.912500、有限性及40epoch/800Adam通过，仅早期开发结果。当前2/9完整，保留集未开，第50轮深评清理。

第45轮方向核查（2026-10-03）：首个M3完整预算仍无经典优势；百万步配对呈成功保持与yaw精度的不同折中，当前无符合原联合要求的赢家。只继续冻结的三方法三种子比较，不追加调参/结构/场景预算；奖励尺度是风险线索而非因果证明，原门失败则停止优势分支。详见[第45轮深评](../wheelleg_warp/ARCHITECTURE_CONSTRAINT_AUDIT_2026-10-01.md)。

第41轮实际进展（2026-10-03）：M3/1609完整200万策略步、100次选择评估完成且正常退出；全预算最佳82万步29/32、Jψ .957592°，最终200万步22/32、1.036518°。速度/到达原门32/32，但最佳丢3个B0成功，仍弱于B0/B1的32/32/.411152和.272426，不能宣称方法优势。4000训练epoch、80000Adam更新及有限性与所有指标/checkpoint哈希过。注册队列已在15:32:51自动新建B2-V/1609 CUDA预实验，原100世界/2M/20k评估配置；20k为22/32、1.133714°，同预算M3早期为27/32/.739438°，只作早期开发比较，不能替代完整配对结论。第45轮再深评清理，门控/最终仍封存。

第40轮方向核查（2026-10-03）：原注册受控比较继续；当前M3中期仍弱于强经典参考，不扩预算/种子、不宣称优势。固定训练世界、部分可观测、机械分布差异、奖励多目标及压力场景覆盖限制已在[深度评判](../wheelleg_warp/ARCHITECTURE_CONSTRAINT_AUDIT_2026-10-01.md)明确；完整原门失败则停止该优势分支，保留负结果。门控与最终集继续封存。

当前第36轮（2026-10-03）：用户最新授权按论文计划继续推进。当前CUDA预实验M3/1609已实际启动，服务wheelleg-height115-gpu-m3-1609.service；100世界、每20k固定32例选择评估、原2M累计上限。队列每种子M3→B2-V→B2，种子1609～1611，单GPU训练进程；不用smoke权重、不改冻结配置或原非退化/效应门。实时进度见protocol_gpu_v3/runs/M3/1609/progress.json与selection.json，启动和阶段核查见results/registered_gpu_pilots_20261003/。原准入记录只作训练前历史证据。

20万步阶段10个检查点均按同一选择身份/场景核验。阶段最优20k为27/32、Jψ .739438°；200k为11/32、1.245126°，明显弱于经典B1_3的32/32、.272426°，不能宣称收敛或方法优势。真实模型/Adam/归一化有限，240k检查点为480训练epoch、9600Adam小批量更新，标准预算继续至2M并完成同预算对照后再判断；20万检查不重置预算。64门控/最终仍未模拟，第40轮深度方向判断和冗余清理。

当前第30轮准入已通过（2026-10-03）：按用户最新要求，正式PPO策略/价值网络、Adam动量和MuJoCo Warp物理均使用CUDA。当前协议为`height115-yaw-v3-v8-gpu`，唯一活动路径`wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/`，准入见其中`readiness.json`。此前CPU策略协议及GPU迁移失败记录仅历史证据。当前100世界、50步/rollout、batch250、10epoch、每种子2M/每20k选择评估及三个预实验种子1609～1611已完成工程准入；没有启动长训练，探针权重不晋升，64例门控及最终类别200例仍封存。条件追加5种子和最终研究结论仍服从原效应门。

同版三方法各12000步/240训练epoch、精确策略/Adam/VecNormalize恢复与真实课程切换通过；正式100世界×50步单轮5000采样、200次Adam小批量更新及精确恢复也通过。CUDA初始化888状态×3方法×3种子均值峰差7.451e−9、条件机械RMS相对差7.240e−10，沿用已冻结σ，不重新校准。实际train函数GPU更新的两次中断730/1000→2000、只初始化一次、未保存步数/坏哈希拒绝通过，恢复明确重启物理episode。采样适配和归一化仍含CPU逻辑，不声称全流程零拷贝或GPU端到端加速。

原28 v8全部成功且原速度/roll-pitch非退化通过，148公共例物理/设计全通过但任务147/148；当前新增144速度边界＋64切点附近全门208/208。原32选择例、8候选同版重评选B1_3(kp.4,kd.3,roll0)，32/32、Jψ .272425528°，预设绝对.05°门有余量。GPU短探针M3/B2-V/B2评估1/4、3/4、1/4，保留航向和1.4rad设计失败；不比较CPU与GPU短探针成绩、不声称收敛、动作全分布等同、连续安全不变性或高鲁棒已证明。完整复核脚本与来源见`thirtieth_round_admission_review_20261003/check.py`及`verification.json`。

## 第28轮：比较入口可运行，旧回归退化仍阻止整体准入（2026-10-03）

当前冻结`height115-yaw-v1-v6`见[协议第三候选](../wheelleg_warp/results/twentyeighth_round_comparison_entry_20261003/protocol_v3/protocol.json)。统一初始向量、64×64、100世界、50步采样、batch250/10epoch，2M策略步和20k选择评估预算；世界参数按真实episode结束的课程切换，固定Nom设计7kg，目标回合内固定。新选择32、门控64及15类别各200最终参数先冻结，门控/最终未模拟，旧最终命名空间未动。B1原8候选在新选择完整评估，选B1_6为30/32、Jψ .295604°，主指标仍有绝对.05°余量。

三模式正式入口工程探针均12000步/240epoch，初始化正确、权重实变、精确Adam/归一化恢复及实际课程切换通过；物理轨迹重启明确披露，权重不晋升。探针仅2/4、2/4、1/4，设计/航向失败保留。新增真实终态FK核验、原solver50分组；CPU合成门控不打开真实门控。证据[第28轮](../wheelleg_warp/results/twentyeighth_round_comparison_entry_20261003/verification.json)。

原28当前B0仅20/28，全部物理/设计过但8例航向超5°，旧速度门28/28、roll/pitch27/28；原CPU28/28记录保持。不能据退化基线宣称学习优势，下一项同状态/整任务受控定位当前J Nom、原Nom与物理执行差异，并补正式中断恢复的耗费预算守卫；未满足这些及最后完整准入审计前不长训。第30轮集中评判不变。

## 第27轮历史准入（含第25轮集中评判，2026-10-03）

原生共同运行配置现已实现：height115_candidate(shared_reference=True)，v6-public-region-reference，按冻结公开工况规则选择完整控制律，在40子步图内5ms状态相位/当前J转移并遵守原请求域。原生与宿主验证标签逐例一致；原18例15/18，新128例126/128，物理/设计工程覆盖与任务困难例分别记录，不声称连续保证或高鲁棒证明。默认v5与旧基线保留。

diff3与virtual6各完成真实2000步10次标准PPO更新、精确策略/Adam/归一化恢复、200步续训及冻结统计评估。权重只用于工程探针，评估2/4、3/4不作为学习优势或正式成果，失败保留。参考来源、实现与资产均需冻结；证据[本轮验收](../wheelleg_warp/results/twentyfifth_round_native_admission_20261003/verification.json)。

方向有条件保留，不再堆无收益映射刷新、幅值权重或要求零Actor所有困难例满分。第26轮已补齐同版torque6电机B2和原8候选手工B1，288个映射fixture、同状态零Actor命令、三模式六正常及144个B1工程episode通过；B2完成真实2200步/11更新恢复联调，探针评估3/4，航向失败保留。独立GPU图回放有连续指标波动，不声称位级轨迹一致，不据工程面板选B1。第27轮完成888个同状态初始探索审计及共同规则log_std校准，拟合组最大RMS目标误差2.643%、三策略种子的开发验证整体偏差最多6.78%，但高度分组差异约24.7%，映射秩3对6仍在；全部λ/协方差/饱和和时序差异公开。只冻结候选初始化，不声称全分布或可达集合相同、没有学习。下一项是同版可执行训练/选择评估入口、应用该初始化后的真实更新/恢复联调和完整研究协议/选模规则/源码冻结（含回放波动与剩余探索差异）；在这些完成前不进行整体正式长训练。证据[第27轮审计](../wheelleg_warp/results/twentyseventh_round_initial_actions_20261003/summary.json)。证据[第26轮验收](../wheelleg_warp/results/twentysixth_round_fair_baselines_20261003/verification.json)。原指标/非退化门与最终封存集保持，下次集中评判第30轮。

## 第20轮集中评判与当前准入（2026-10-02）

第24轮已冻结公开工况选择（h<.11828125且|v|>.75使用参考，否则零修正），新边界144/144及高度速度84/84通过，原18例15/18，新开发126/128；两航向失败完整保留，物理/设计128/128。选择器仍未接入原生训练图、真实短PPO更新仍零，第25轮将优先关闭这两项与同信息协议，不再调切点，尚未整体准入。

第23轮当前J共同虚拟F/H转移使同100例从固定电机89/100提高至97/100，2kHz重映射无新增收益；零修正对照87/100且能通过参考剩余3失败。按公开高度/速度同时覆盖两参数组的工况点存在交叠，但未构造或执行连续域选择器，不能把并集称为某方法100/100。下一轮验证交叠并冻结仅用公开工况的完整控制律选择，用新样本/边界验收；不再幅值混合、按参数真值择优或缩小范围。生产v5不变、无学习，仍未准入。

第22轮状态相位试验仍未准入：同100例，加权状态版86/100，完整幅值状态版89/100，完整幅值时钟对照70/100。相位反馈相对时钟有19例局部收益，但幅值消融新增7例失败，不能当全面改进。生产v5不变、无学习；下一步核固定电机曲线在不同几何下的VMC虚拟力/力矩失配，优先几何一致转移与完整验收，不继续调相位系数或标量权重。

第21轮原生5ms参考执行链验证通过，但参考泛化规则拒绝：初版84例79/84；一次按已知零修正点重锚后旧网格84/84，而预注册加密单元仅86/100，14处训练组合设计失败。5例对照关闭参考全过、开启全失败，不能声称插值保可行。已归档失败自动配置并恢复生产v5，不继续调权重/缩窄验证区域；下一步核模板与实际状态的动态对应，统一运行与全域门仍未完成，无PPO。

保留VMC＋六状态LQR＋差模PPO方向，当前未证明方法优势或高鲁棒。近五轮真正保留的控制改动是恢复原差动阻尼；共同制动参考使外置完整面板达到15/18，但新测默认工厂仍仅4/6名义任务通过，二者不能混用。下一项优先统一原生共同参考执行/配置，再核0.115～0.38m及连续速度/训练参数域、冻结同信息对照与真实短PPO采样更新/恢复。停止继续堆在线预测器、重置层或静态过滤器。

已统一原生设计成功契约，版本v5：1.4rad主动设计门逐物理步累计，历史违例进入终态success与奖惩；1.5物理指标保持独立，Actor38维不变。历史当前姿态恢复的反例和18例全程原生/独立判定一致性通过，旧物理/接口检查保持。详见[第20轮证据](../wheelleg_warp/results/twentieth_round_admission_review_20261002/verification.json)。

第6.3工程准入、学习性能与第6.4方法继续门分开：历史经典28/32也曾通过工程链路，不能额外要求所有训练困难样本零Actor满分。原速度/姿态/设计阈值与失败计数不变；台阶失败继续保留，不宣称可学习、也不将它一项当作阻止所有短联调的充分条件。当前仍不整体训练，因为默认名义门、统一配置/连续域、新协议与真实更新均缺证。下次集中评判第25轮。

## 第15轮集中评判与当前准入（2026-10-02）

第19轮已修复共用核几何投影覆盖原差动速度阻尼的问题，版本升v4；无新增增益/动作权限/观测。v4配合第18轮冻结参考的完整18例全门15/18（名义6、训练组合5、压力4），训练组合起步越界关闭，但台阶速度RMSE .102440>.1仍失败。参考尚未自动接入默认工厂，实际短PPO更新仍为零。第20轮集中评判将核统一运行配置、连续域、全程设计验收与训练契约、原计划工程准入和学习性能边界；原真实门和失败均保留，不直接启动整体训练。

第18轮最新口径：跨参数共同参考的制动段10/10通过，但初版评分漏掉制动前主动1.4设计门，已纠正；从站立全程复验高速仅9/10，原18例完整面板14/18（名义/训练/压力6/4/4）。训练组合正向在2.038s起步末段已越界、4.5715s才开始制动，另有20mm台阶速度RMSE .101831>.1。评价器已补全全程记录和前缀拒绝，不再用停车参数修复已发生的历史失败。当前依据[全程回算](../wheelleg_warp/results/eighteenth_round_robust_reference_20261002/full_verification.json)，旧verification.json属于制动段历史口径。下一轮前移到起步/行驶段共同控制与完整回归，生产未晋升、无学习，仍未整体准入。

第17轮证实共同修正可使已投影的静态目标重新越界；但单次当前静态几何投影候选虽守住两例主动设计门，停距.616321/.617713m均失败，拒绝晋升。静态目标可行不等于动态任务可行。下一项沿原完整时域/Q/R和输入域，对名义及已见训练扰动共同设计一条参考，采用离线真实完整物理评分和独立复验，不继续在线预测初始化或加静态硬门；生产未改、无学习，整体准入仍开放。

第16轮单因素机制核查：同v3/原冻结计划24例物理24/24、任务22/24、任务＋主动1.4设计18/24，名义/仅质量/仅摩擦/仅驱动差全门6/4/4/4。质量主要引入共同关节偏移，摩擦和驱动差主要引入左右差异，均能耗尽原薄设计余量；单独质量补偿不够。下一步核额外共同电机修正处于几何参考投影之后的串联作用，在冻结状态上确认后再做一个最小公共分配修正，生产未改、无学习，仍未准入。

[第15轮深评](../wheelleg_warp/ARCHITECTURE_CONSTRAINT_AUDIT_2026-10-01.md)保留VMC＋六状态LQR＋差模PPO主线，停止继续叠加预测器初始化/完整Data快照。新增同一v2/38维工厂的五设计节点×六正常30例，物理与主动1.4设计门30/30、原任务28/30；仅0.115m正反高速停车约.663554/.666539m未过.6m。四个较高节点均6/6，不能扩大为连续区间、单侧边界或随机域通过。

第11轮v3冻结计划在参数面板仍名义6/6、训练角点3/6、压力4/6；第12～14轮全部1/96/96次在线请求逐值等于原指引，没有实际反馈改善，连续预测仍未通过原门。数值配对门不放宽，但也不把修复某个实验预测器当作所有控制方法必须包含的模块。下一轮用原v3/计划在同六正常场景隔离质量、摩擦、驱动差，定位真实设计/任务余量丧失原因，再决定一个最小状态反馈修正；不先堆求解器或继续扫增益。

当前整体训练仍未准入：最低位共同控制、同版本完整声明域工程检查、公平新协议、实际短PPO更新/优化器与归一化恢复未关闭。0次更新的形状/保存检查、旧训练成功和五节点名义检查均不能代替。压力例单独报告，不新增“所有压力必须100%成功”门；原高度、物理、任务阈值和动作权限全部保留。下次集中评判为第20轮。

## 第10轮集中评判与当前准入（2026-10-02）

第13轮过去状态/已知发令由模型自身重放，冻结首窗发令误差7.15256e−7Nm通过原门；连续分配前95次通过，第96次再次未过发令门，回合未完成。请求尚等于原指引，不称状态反馈收益。冻结失败用新13臂/1臂模型重查均通过，复用历史/重复性条件仍待隔离；生产不改、阈值不放宽，整体训练未准入。

第12轮状态相关共同分配在首个5ms未过原预测支持门：位置/速度通过，但发令误差1.79529e−4Nm>1e−5Nm；过去命令初始化对照没有改善。原输入/关节/目标门未放宽，实际首窗仍安全但回合未完成。临时预测/分配生产分支已移除，来源及冻结拒绝复核保留；下一块数值历史与高增益反馈误差支持未关闭，不能借未过门模型直接进入动态控制或训练。

第11轮已提供显式v3共同Nom边界，经典修正先归Nom并接受约束，随后计算Actorλ/实际残差，残差观测及惩罚不再包含经典量。原请求幅度和修正/增量域不变，默认v2不晋升，旧预测器不覆盖新边界即拒绝。同输入逐状态306224个物理步发令逐值对应，完整18例仍13/18全门；边界/计量已修不等于鲁棒控制，下一块处理共同动态可行性。当前仍无真实新协议PPO更新。

[第10轮深评](../wheelleg_warp/ARCHITECTURE_CONSTRAINT_AUDIT_2026-10-01.md)有条件保留纯仿真差模航向问题及标准PPO，停止叠PD/定时补偿、扫增益或逐步全时域求解。五轮实质收益为38维执行请求契约及一次名义可行序列；任务11→13只改善名义高速，训练/压力仍3/6、4/6，设计余量及扰动失败不支持高鲁棒，未证明PPO主指标优势。局部模型/力矩合法与动态安全分别判断，原范围/门/权限不变。

下一块先统一公共Nom与Actor边界，经典量不记作Actor残差或惩罚，所有对照共享；再处理一个有状态/误差支持的共同动态参考/输入分配机制。整合外置计划不等于鲁棒修复，Nom本来有反馈，不盲叠等价反馈。正常及声明训练域工程门、压力/重复性、同信息强对照冻结与真实短PPO更新/恢复仍待；0更新预检和旧范围不代用。第15轮再深评，当前不启动整体长训练。

## 第9轮及第5轮证据记录（2026-10-02）

第9轮[统一18例冻结计划复核](../wheelleg_warp/results/ninth_round_frozen_plan_panel_20261002/verification.json)在当前v2/38维候选上原样执行两份既有完整计划，其他四场景额外量零；一次统一名义六例原物理/任务/主动设计门6/6，但全18全门13/18，训练域/压力组仍3/6、4/6。名义正向关节设计余量仅0.739微弧度，扰动高速均超1.4设计域，速度/停距仍有失败；不能称高鲁棒或默认factory已通过准入。外置经典量还进入原残差监控/惩罚，未统一公共Nom/Actor契约，不晋升、不训练。此结果证明该名义原输入域存在完整可行序列，不证明冻结时序具备误差支持或在线可持续性。第10轮集中客观评判，原范围/门与主线不改。

依据[五轮方向评判](../wheelleg_warp/ARCHITECTURE_CONSTRAINT_AUDIT_2026-10-01.md)，保留纯仿真差模PPO航向精度方向，但当前整体训练不准入：统一候选在0.115m三组预注册六正常参数面板物理18/18、任务11/18，名义4/6、真实训练域组合3/6、压力组合4/6。前四轮主要改善证据与代码边界，任务性能未提高，不能称方法整体有效或高鲁棒。+.05驱动的首面板误名已更正，正式本轮训练域按原+.03重新注册测量；Actor为零，观测延迟标签不证明学习/全闭环延迟鲁棒。

下一块以统一候选的动态输入分配与原零停车目标兼容性为控制重点，并明确内部执行状态/历史与同信息对照；不再把经验PD、参考坐标替换或全时域SLSQP当现成修复。按原高度范围、原成功门及Q/R验收，不以新增诊断脚本数量为进展，不为躲失败改主指标或打开最终集。之后依次同版本范围验证、冻结协议及短预算PPO/恢复联调，达到条件才整体长训练。第10工作轮再集中评判和清理冗余。

## 2026-10-02 当前方向复核

当前主线仍为**纯仿真、VMC＋六状态LQR基础上的差动残差PPO航向精度研究**，名义腿长范围为用户指定0.115～0.38m。第4.4节允许理想仿真状态反馈；实机传感器认证、CAD干涉认证及硬件壁钟5ms期限不是这项纯仿真正式训练的新增前置条件。模拟控制周期、声明的延迟、壁钟计算开销与训练吞吐分别报告；若另称实时控制或实机安全，则必须单独验证。

复核发现推进方法仍有问题，不能按“没有问题”直接启动长训练：

- 最新全时域SLSQP是**0.115m、标准7kg平地、公共F/H/W共模动作的离线可达性诊断**，不是PPO，也不是原M3差模策略能直接表达的补偿。它的15状态矢状面LQR成本不是航向主指标Jψ；两例停车成功不能证明差模学习优势。工程共模修正若进入基控制，所有方法必须共用、重新冻结并验证；不能仅给M3额外权限。
- 当前实验入口与训练入口未统一。`train_native.py`等仍为历史固定高度任务，`train_height_capability.py`是旧0.16～0.38m能力探针；最新投影、径向保护及当前J表在独立实验中启用，默认训练没有自动继承。需要一个明确版本、可执行的height-115基线及统一六正常/声明范围验收，不能把跨版本成功拼为一个协议。
- 最新保留序列在5ms边界重新预测，正向第48次（约停车后0.2405s）未来betaR超1.4rad设计域约8.1e−7rad；当时实际关节余量0.118rad，真实A/B及物理门仍有效，不能称实际1.5rad限位越界。证据只说明紧贴名义边界的轨迹不能直接充作可持续备份，尚无对应误差支持与失效回退。不得放宽1.4或物理门掩盖失败。
- 原M3动作滤波与λ会引入观测中未显式给出的执行状态；这是POMDP/信息契约问题，不是已证的PPO损失错误。正式新协议需明确可见状态或历史，并在同信息、同执行权限下比较。基线自身的共同支撑和停车问题不能交给差模PPO或测试集调参解决。

继续顺序：**统一可实际执行的基控制及仿真信息/约束契约 → 同版本六正常及声明高度/参数范围验收 → 冻结公平比较、短预算PPO链路/恢复/评估检查 → 按既定研究门启动长训练**。同时在公开开发集确认航向指标仍有提升空间；不打开最终集，不改主指标，不改PPO损失。全时域轨迹搜索保持辅助诊断地位，不继续默认扩充为每个控制子步的在线求解器。下方旧日期的继续门和结果保留为历史，以本节及根目录最新记忆为当前入口。

**2026-09-29 当前继续门更新：** [1ms更新及1ms采样到执行的同状态对照](../wheelleg_warp/results/height_115_scheduled_guard_20260929/verification.json)中，两相位和理想即时参考均在约998ms模型无解，实际几何尚安全，不能把此前单例一秒通过当鲁棒全高度能力。GPU装配/增量LP热点中位约0.499ms但65/132超0.5ms，未完成真实实时链路。失败点1Nm域外的有界模型诊断提示需求约1.10～1.18Nm，尚非实测证据。下一步先核当前状态真实动作响应和标定域，再验证完整因果介入/恢复及全范围；不继续单靠趋势门/计时微调、不修改PPO核心或打开最终留出。详见[持续工程评估](../wheelleg_warp/HEIGHT_115_LOCAL_FEASIBILITY.md)。下列条目为既往阶段证据。

**2026-09-29 当前状态滚动制动实测：** [反向平地当前求解三臂10 ms](../wheelleg_warp/results/height_115_live_braking_20260929/verification.json)中，一次解和每0.5ms滚动解均守住真实A链/八关节等局部门、零臂失败；估计名义加速度时扣掉上一安全动作，避免重复计入力矩作用。但[固定候选延长至20 ms](../wheelleg_warp/results/height_115_live_braking_20ms_20260929/verification.json)在12.5 ms因主动FK收紧模型距离约−0.085 µm而停止，真实A链仍高代理线约41 µm；不能称真实几何失败，也不能称递归模型安全。当前Python求解中位1.17 ms、未含GPU传输，超0.5 ms预算；无独立加速度误差界。下一步须独立标定加速度/延迟并纳入滚动约束，再选择实时求解器和六正常整回合试验，不能把这一个10 ms例子当0.115～0.38 m任务、PPO优势或实机安全。

**2026-09-29 制动预算继续门：** [固定模型、起点每电机≤1 Nm的耦合制动距离检查](../wheelleg_warp/results/height_115_braking_budget_bounded_20260929/verification.json)在首次警报状态仅反向平地1/6可行，失效前15 ms为4/6；两正向平地需要的关节制动、三地形冲击后的径向制动均超当前模型动作能力。该结论只在恒G/恒上一帧加速度假设下成立，位置误差预留不能冒充加速度误差界。唯一模型可行候选的[当前状态复核](../wheelleg_warp/results/height_115_braking_response_20260929/verification.json)因速度差分加速度变化使约束余量−0.005185、低于固定−1e−5门而停止效果评分；没有物理安全成绩。下一步必须在当前运行状态求解动作并标定加速度估计误差，建立终端制动或滚动反馈再验整回合；当前1 Nm局部模型、警报规则、归档固定脉冲均不能称安全控制器。0.115～0.38 m目标及仿真代理线保持，PPO训练/最终留出关闭。

**2026-09-29 最新问题评估：** 固定现有5 ms预测、六折预留与数值门，按当前主动q/dq及前一帧dq在失效邻域第一次报警，六正常例分别提前5/5.5/5.5/6.5/5/9 ms，警报时1 Nm模型有解5/6。为避免“原失败还未进入5 ms窗口”的虚假通过，[固定首次警报动作同图跟踪到10 ms](../wheelleg_warp/results/height_115_alarm_response_v2_20260929/verification.json)：五候选前5 ms均守线，但10 ms均失守，仅比零臂推迟0～1 ms；无动态电机裁剪，100个候选物理步接触对与零臂全同。轨迹表明终端外向速度仍大，当前位置预测约束没有建立持续制动能力。下一步必须核剩余距离/外向速度/可用制动加速度与执行延迟，设计终端制动可行域或高阶约束及滚动反馈，再测整回合；不能把短窗通过当0.115～0.38 m任务或PPO优势。[整数步时序证据](../wheelleg_warp/results/height_115_alarm_timing_integer_20260929/verification.json)只覆盖已选失效邻域，尚无整回合误报或实时延迟证明。正式训练、最终留出继续关闭。

**2026-09-28 用户新范围（height-115，待工程验收）：** 后续名义目标腿长统一改为 **0.115～0.38 m**；0.115 m 高于现有模型主动关节推得下限约0.094704 m再留20 mm的仿真代理下限约0.114704 m，仅余约0.30 mm，不能据此称实机安全。下文所有 `height-v1` 的0.16～0.38 m六档、120/48例及PPO探针均是**旧范围历史结果**，不作为新下端成功率。先为新范围独立设计0.115 m名义工作点、打通闭链初始化/观测/零残差回归及电机约束，并重建开发/最终协议与归一化；不过工程门不长训、不改PPO核心。新高度范围版本称 `height-115`，不占用此前可选前视感知的 `height-v2` 命名。

**height-115首次工程门结果：** 五节点新名义设计的0.115 m离线线性检查通过，旧四节点及旧CPU/Warp配对不变；九档平地/常规/20 mm单侧公开零残差面板在加入“2 kHz全程腿长≥0.114704 m”验收后分别为**8/9、23/27、20/72**。0.115 m在三类面板均0通过，平地最短腿0.113799 m；独立14例2 kHz轨迹全有余量越线，最短0.095438 m，部分主动关节触及并越过软限位。原任务门下的完成与平均腿长误差不能替代此安全门。完整机制和源码/轨迹哈希见[新高度域工程复核](../wheelleg_warp/HEIGHT_115_ENGINEERING.md)。**新工程准入未通过，继续停止PPO训练**；下一步先研究可得关节/IMU信息下的低位安全控制及机械约束，不改用户范围或放宽余量线。

**0.115 m下端机制复核：** 独立被动链2 kHz回放确认14/14真实轮心同样越线，动态闭链误差最大约0.135 mm；一项固定的可得关节状态预测径向反射虽实际增加髋命令、未被电机包络裁掉，六个正常场景仍0/6通过原任务＋真实轮心／关节联合门，已按门停止。详见[低位安全反射](../wheelleg_warp/HEIGHT_115_SAFETY_REFLEX.md)。问题仍优先是低位关节角与腿长的动态可行性，不据此修改PPO核心或宣称硬件极限已证。

**双约束进一步核对：** 新增一项固定关节角预测制动后，14例软关节限位均未越界，正常六例的原任务＋实际轮心＋关节联合门仍仅1/6；已依协议停止，不调该组经验阈值。见[关节角制动结果](../wheelleg_warp/HEIGHT_115_JOINT_REFLEX.md)。下一研究门转为关节角/径向长度/机身运动的统一可行域，而非PPO优化器改造。

**局部动作权限证据：** 在三个0.115 m平地首次越界前20 ms窗口，CPU/Warp一步及40步配对通过；六电机各≤0.1 Nm恒定增量的局部解在Warp冻结基础命令窗口3/3守住真实轮心/关节/姿态，而实时闭环加相同增量仅1/3，全回合仍0/3。固定起点雅可比的原M3差模子空间在这三处局部线性约束均不可行。见[局部可行性与边界](../wheelleg_warp/HEIGHT_115_LOCAL_FEASIBILITY.md)。这提示优先研究同信息共模安全控制及动作分配；既不证明时变M3无解，也不支持现在修改PPO损失或开启长训。

**共模动作与运行成本：** 同三个局部窗口用共同径向力/髋矩/轮矩三维动作在CPU与Warp冻结控制窗口均3/3过门，但该基仅在起点求一次，40步执行固定电机增量；在线重算基础控制后仍1/3且全回合0/3。当前Python实现的13×40步有限差分中位约112 ms，HiGHS 24条安全约束＋12条增量约束的LP约0.726 ms，无法逐2 kHz物理步运行。该结果仅指向需要可实时的因果共模安全分配和同信息强经典对照，不能把共模三维局部表示直接称为论文创新或已部署控制器。见[工程证据](../wheelleg_warp/HEIGHT_115_LOCAL_FEASIBILITY.md#三通道共模与实时成本)。

**最新低位方向门：** 三个平地窗口改成实时重算名义控制和逐步共模雅可比后，按每电机≤0.1 Nm与既定数值裕量，离线局部LP仅2/3可行、Warp严格非线性同接触仅1/3通过；这仍是未来窗口选出的常量系数，不是在线策略。独立发现低位横坡对称滚转补偿把一腿径向目标直接请求至代理线以下；新高度模式可选的单侧目标下限投影把横坡实际轮心最短值从0.107773 m改善至0.113564 m，但六个正常场景联合门仍0/6，故不晋升。平地关节越限时腿长仍约0.115～0.119 m，必须把共同角向、径向长度、姿态和驱动盒纳入统一可行域；优先修名义控制与同信息强经典，再比较PPO差模价值，**当前不改PPO算法或启动训练**。见[局部证据](../wheelleg_warp/HEIGHT_115_LOCAL_FEASIBILITY.md#实时名义控制下的共模窗口)、[完整工程证据](../wheelleg_warp/HEIGHT_115_ENGINEERING.md#低位名义控制的上游约束)。

**静态可行域及因果预测门：** 新[站立连通支路逆解](../wheelleg_warp/results/height_angle_envelope_20260928/verification.json)给0.115 m主动关节硬限/各留0.1 rad时腿极角半范围10.790°/8.901°；包络到约0.324 m后回缩，0.38 m边界可由A链伸直先触发，不能按高度越高越安全外推。[首失效事件的可得状态短时预测](../wheelleg_warp/results/height_115_predictability_20260928/verification.json)中，上一帧速度差分加速度外推10 ms已有13个重叠窗口起点漏报、最大乐观腿长误差0.538 mm，超过最低名义目标至仿真代理线0.296 mm空隙。该统计不含动作影响、接触模型和实机噪声，不能叫形式安全证书；先测可得状态下的动作/接触预测误差，再设计联合安全分配。新核的[变高度LQR/MPC、闭链CBF-QP及轮式双足CBF/NMPC先例](LITERATURE_MATRIX.md#主流机器人论文与官方开源项目补查)也排除了把这些组件单独当创新点的主张；最终仍需全高度实测与同信息强经典/PPO配对优势。

**接触动作模型门：** 0.115 m横坡首次接触、接触后3 ms、越线前0.5 ms三状态的[CPU/Warp小共模动作配对](../wheelleg_warp/results/height_115_contact_action_pair_20260928/verification.json)在归档基础ctrl(t)冻结、每步重算关节雅可比的条件下，源一步及5/10 ms均3/3过门、10 ms接触对420/420同，实际轮心跨后端误差≤0.098 µm，八关节裕量误差≤0.000000617 rad；接触集合相同但轮矩正负响应仍不对称。首次接触后10 ms的最佳起点约0.1 Nm电机增量小动作仍比代理线低0.222 mm。该结果仅支持这些**离线局部物理代理**并否定小动作救援，不能称在线安全可行或完整电机盒不可行；继续核可得状态对动作/接触的预测误差与实时执行，再考虑控制器和PPO训练。

**接触时序细分与动作上界停止门：** [无动作预测分层](../wheelleg_warp/results/height_115_contact_predictability_20260928/verification.json)表明10 ms腿长最大乐观误差在地形冲击尚未进入理想主动dq时可达0.538 mm，冲击后同三个地形归档窗口最大约0.0052 mm；但接触后5 ms和无地形反向平地仍有漏报，主动FK与真实A链也可差14～37 µm。若使用本体快速反馈，应显式验证动作响应、闭链误差和感知延迟。[原电机峰值盒的单次共同径向上界](../wheelleg_warp/results/height_115_radial_authority_20260928/verification.json)初始约937 N/四髋增量约±39.1 Nm，按预注册CPU/Warp误差门在7 ms停止，**未完成10 ms或整回合**；已完成段候选轮心未越代理线但腿迅速抬高、接触集合改变，不可当0.115 m能力或实机安全。下一项按六正常场景整体留出核5 ms、仅用当前q/dq/IMU/已发命令的有动作单侧预测误差与同信息电机盒可行性；过门后才设计在线联合约束，不改PPO核心。

**有动作预测留出当前结果：** [单图六世界整体留出](../wheelleg_warp/results/height_115_action_predict_single_graph_20260928/verification.json)在首次失效前15/5 ms分别测九个零/±共模/±混合动作，当前同图九臂起点逐值一致，5 ms电机无裁剪；内层世界留出定真实A链与八关节单侧误差预留。108个窗口动作在扣预留后均无漏报，但54个关键晚窗的六个零臂均真实失败、**0/54个候选具有正的联合预测安全裕量**；四个真实短暂守线的非零动作仅有微米/微弧度级薄裕量，不能称可救援。原始不扣预留的晚窗预测安全28/54中有26个实际越界，说明不能以“模型可算”取代误差门。接下来先求**连续**共模小盒在当前模型/预留内是否存在安全动作，若无则停止该小动作候选并研究动作范围、接触感知或先验几何余量；仍不能据此改PPO损失或开启长训。现有2 kHz基控读取MuJoCo姿态/速度真值，50 Hz Actor读延迟32维观测；今后PPO与强经典高层必须共享相同观测和低层安全分配，单列安全层消融，实机传感器时序尚未验收。

**连续小动作门也未过：** 在上述六个晚窗、相同留出G和真实A链/八关节误差预留下，[连续共模三变量LP](../wheelleg_warp/results/height_115_continuous_common_lp_20260928/verification.json)把动作从九个采样点扩为整个位于起点每电机±0.08 Nm的小盒，仍**0/6线性可行**；0/4号受关节角、1/2/3/5号受腿长约束。该LP未在Warp执行新动作，腿长仅局部线性化，也未纳入全过程动态电机盒，因此只关闭**当前模型和小信赖盒**；不能拿它证明0.115 m机械或任何PPO动作不可能。下一研究门是先用新的同信息公开扰动数据标定有更大制动权限时的动作响应/接触预测误差，再开发在线联合约束；不能直接把仅在0.08 Nm拟合的G外推到大扭矩或删除误差预留。

**1.0 Nm档重新标定后的继续门：** 旧线性代理所需起点电机峰值外推0.113～0.934 Nm只用于一次预先固定的1.0 Nm新标定，不借旧G/误差预留。[新同图六世界结果](../wheelleg_warp/results/height_115_action_predict_1nm_single_graph_20260929/verification.json)先过台阶晚窗预检，再完成108个5 ms九臂动作；逐步电机无裁剪，接触对相对零臂无变化。新内层世界留出腿长预留约54.4～59.1 µm、关节0.000261～0.000331 rad；扣预留后0漏报，但六个关键晚窗零臂6/6失守，54个动作只有3个被预测且真实短暂守线，不足以建立六正常任务能力。详细[问题—证据—继续门](../wheelleg_warp/HEIGHT_115_LOCAL_FEASIBILITY.md)要求先在此固定模型中求连续1.0 Nm共模盒，再以Warp非线性和实时成本复验；不能以局部薄余量或同步快照称算法创新、在线安全或0.115～0.38 m全任务完成。

**连续1.0 Nm动作验证后仍未过完整门：** [冻结六折模型的连续LP与同图Warp](../wheelleg_warp/results/height_115_continuous_common_1nm_20260929/verification.json)在原真实A链/八关节预留之外再留1 µm腿长、0.0001 rad关节的**数值**裕量，六晚窗只0/1号2/6模型可行；同图5 ms真实A链、八关节、姿态、接触和电机对照这两例均过门。平地+0.5从零臂关节越限到候选+0.000247 rad，台阶从零臂轮心低线8.12 µm到候选高线17.11 µm。两动作系数均超原九臂标定凸包，世界2/3/4/5无数值门内的模型动作；这只证明**两处公开窗口有可执行的局部组合**，未证明连续动作预测误差界、六正常整回合或0.115～0.38 m任务。下一步须从更早的可观察状态、接触/真实轮心误差与低层传感延迟重新评估可行性；再决定是否扩动作权限和构造在线经典安全层，PPO算法改造与长训继续关闭。

**2026-09-28 准确接触时序与最新继续门：** 修正上一轮把零动作首次接触所在50 Hz区间末误称为“准确接触后触发”的口径。原/新公开48例在同图整数320物理步的固定M3轮差矩下，零动作17/26、准确2 kHz接触后22/25、50 Hz区间末20/25、特权提前40/46；精确接触臂不能复现提前臂收益，但不否定其他接触反射。现有轮体力信号在名义单侧障碍中有约1 ms侧别提示，也在平地/对称台阶误触发单轮阈值，当前Actor未接收该力。详情见[准确时序与传感器复核](../wheelleg_warp/HEIGHT_V1_EXACT_CONTACT_TIMING.md)。下一步按原纯本体感知范围先核0.16 m端机构动作余量与侧别判别鲁棒性，再做同信息经典反射对照；未过门不追加PPO训练、不改PPO核心。全高度全任务仍未完成。

**2026-09-28 多高度接触前信息门（旧区间末口径）：** 独立height-v1标准PPO在预定102,400步后20 mm边界仍17/48，常规回归从120/120退至112/120，权重不晋升、不追加训练。原M3轮差矩在**模拟器特权侧别与名义提前100 ms接触区间**下，原20 mm边界由16/48分别升至39/48、40/48，新独立公开16～20 mm场景由26/48升至两轮均46/48；零动作接触所在50 Hz区间末触发分别20/48、25/48。首次接触前成对左右障碍的32维本体观测差与相同场景GPU数值背景同量级，因此不能把特权预瞄成绩归给当前Actor。详细配对、数值重排与近邻先例见[预瞄方向审核](../wheelleg_warp/HEIGHT_V1_PREVIEW_DIRECTION.md)。原无未来障碍输入的论文范围仍有效；如新增可实现前视传感器须独立height-v2协议、同信息强经典对照和新最终集，不能仅以“预瞄+PPO”自称创新。0.16～0.38 m全部任务仍未完成。

**2026-09-28 多高度要求与范围重评：** 用户指定名义腿长 0.16～0.38 m 全范围均须完成地面任务。Warp 新建独立高度条件模式：每个并行世界创建时从该范围连续采样目标并覆盖两端与 0.30 m，回合内目标固定；Actor 显式看到目标腿长；成功须同时满足原任务门、启动 1 s 后腿长 RMSE 与终态误差均不超过 0.02 m。旧 CPU/0.30 m 协议和历史分数保留，旧策略不能直接称为多高度策略。此前把**单侧27 mm零残差压力例**作为新多高度训练的前置障碍，是把第6.1节的**25/30 mm范围外推**与**0～20 mm训练范围**混用；其失效机理证据保留，但不阻挡独立 height-v1 的工程与短预算能力验证。

训练范围的两轮分层零残差检查各120/120；两组连续高度训练种子为127/128和120/128，对应固定0.30 m同地形为127/128和118/128。训练上边界20 mm单侧障碍、六档×左右侧×±0.5/±1.0 m/s两轮仅17/48、18/48，其中0.38 m均1/8，仍是必须研究的姿态缺口。128世界高度条件 PPO 独立归一化的12,800步两更新仅通过工程链路，不作效果成绩。详细协议、配对结果和限制见[多高度范围重评](../wheelleg_warp/HEIGHT_SCOPE_REASSESSMENT_2026-09-28.md)。当时建议冻结height-v1开发/留出并做同预算标准PPO对照；其后102,400步能力探针已失败，当前继续门以本文顶部为准。27 mm压力结果单独报告。逐高度和未见过的区间内高度分别报告，不以平均值掩盖失败。高度随机化本身不作为论文创新点，也不据此改 PPO 核心。

**2026-09-27 方向复核：** GitHub 与近年论文显示“PPO＋VMC/LQR”“残差叠加”“地形越障”均已有近邻；当前差动动作子空间仍是待验证假设，尚无可宣称的方法优势。不为凑创新先改 PPO 核心；优先验证接触与电机包络下的合法动作、可获得的观测及强经典/同维动作对照。新证据、检索边界和保留/转向条件见 [文献矩阵最新节](LITERATURE_MATRIX.md)。原冻结协议及历史结果不改写。

修订日期：2026-09-21。依据本目录 [jianyi.md](jianyi.md) 修订，并更新训练前工程验收进度，证据见 [PREPPO_REPORT.md](PREPPO_REPORT.md)。用户确认：纯仿真，当前主机RTX 5060，两个月内投稿；PPO优先，ROS2独立用于毕设。工程验收与短更新不代表论文优势或正式训练完成。

**2026-09-21最新执行状态：用户已取消定时任务，automation ppo已删除，后续在当前任务内推进。M3/1609的20万步工程核验已通过，性能未超过B0/B1；独立systemd服务继续原累计200万步预算，见第12.9节及v2_progress.json。该运行在8万步处有披露的恢复分段，其余方法和种子尚未启动。IEEE全文缺口作为研究限制保留，不证明创新。第6.4节v1与历史证据保持不变，门控集和最终集继续封存。**

## 1. 结论与范围

首选题目：**面向不对称接触航向精度的双轮腿机器人差动残差强化学习控制**。

英文暂名：Differential Residual Reinforcement Learning for Heading Accuracy of a Two-Wheeled Legged Robot under Asymmetric Contact。

这是待验证的研究假设，不是已证明的创新或已完成的PPO算法。保留VMC＋六状态LQR，PPO学习差动补偿；不以“使用PPO”“混合控制”“加入域随机化”为独立创新点。不同时做SLAM、导航、端到端跳跃、视觉感知、实机迁移和大模型。

主问题：在未知的左右接触差异和参数变化下，结构化残差能否比充分调参的经典控制、手工差动调度及通用残差PPO，更有效地兼顾姿态、航向、速度和样本效率？

现有控制器已经包含左右支撑调节和航向差矩。研究对象是**学习已有差动通道中固定反馈律难以表达的补偿规律**，不是首次引入差模控制。先定位固定反馈失效条件，再判断手工调度是否足够、学习增益是否跨场景和训练种子存在。三维动作是起始候选，最终保留一维、二维或三维由验证集递进消融决定。

第一篇聚焦地面滚行越障。跳跃作为冻结基线和后续扩展，不作为本篇必须完成的学习任务。不能把地面差模映射直接用于空中，宣称覆盖全部跳跃相位。

## 2. 已有基础与未完成项

从原项目复制了2026-09-15的独立基线，来源与哈希见BASELINE_MANIFEST.json。模型轮距300 mm，轮径100 mm，7 kg名义质量，上平台220 mm；六状态LQR、VMC、跳跃状态机保留。

既有证据：7 kg、±1 m/s、左右10/15 mm凸台8/8通过，最大yaw约4.925°；7/8 kg名义基础/落地急停22/22通过；380 mm腿位跳跃已修复。这些是已有经典控制证据，不是PPO成果，也不能作为不确定参数下的性能统计。

模型近似更新（2026-09-17）：保留原静态凝聚失败证据，当前采用已存在的慢模态降阶候选并重新验收。四工作点0.2～5Hz密集采样误差0.260%～0.346%，完整15状态闭环及实际物理增益插值中点稳定，14/14非线性脉冲通过。精度门槛仍为10%；详见 [修复报告](PREPPO_FIX_REPORT.md)，不扩展为全局/实机稳定证明。

9月16日实现修正：六状态控制器默认从独立名义模型缓存增益、前馈、平衡角和 `mass_scale`，环境只重建运行状态；残差已接到六状态覆盖之后的最终约束前，峰值计时统一提交一次。验收包含质量变化后的同状态命令一致性和原轨迹逐值复现。历史地形测试虽记录最低速度，但通过判据未包含速度RMSE；8/8不能直接当作第6节新协议的成功率。

已实现Gymnasium环境和五种残差动作接口，完成训练前基线/脉冲及短PPO链路检查；尚无可用的训练策略、正式学习曲线、正式论文对照实验或实机验证。当前底层使用仿真真值，电源/效率/热/回生理想化；机器人外壳及部分连杆的接触仍未完整建模。论文必须明确这些限制，禁止据此声称整机无擦碰、节省电池能耗或sim-to-real成功。

## 3. 与已有工作的区别及查新门

残差强化学习本身已有工作：Johannink等将传统控制与学习残差叠加，用于处理难以建模的接触和摩擦。[Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201)

更近的轮腿研究已涉及残差状态补偿、信赖域和不确定性约束，不能简单重复“残差＋自适应限幅”： [Residual Policy Optimization With Trust Region Constraints](https://ieeexplore.ieee.org/abstract/document/11202537)。已取得摘要，尚未完整阅读全文。按用户最新决定保留该限制，不再将取得全文设为训练前置门；不能据此宣称已排除该文全部实现与本机重合。

平台水平控制也不是空白： [Horizon-stability control for wheel-legged robot driving over unknow, rough terrain](https://www.sciencedirect.com/science/article/pii/S0094114X24003148)。跳跃和模型控制＋学习结合也已有实机研究： [Robust quadruped jumping via deep reinforcement learning](https://www.sciencedirect.com/science/article/pii/S0921889024001830)。这些文章是研究边界，不是本机器人可直接使用的参数来源。

本次按建议补充以下直接相关工作，检索截止2026-09-16。下表只记录已核对的官方摘要/元数据，不冒充完成全文复现；“对本方案的要求”是本项目据此作出的设计判断。

| 工作与来源 | 已核对内容 | 对本方案的要求 |
|---|---|---|
| [Hybrid LMC](https://arxiv.org/abs/2204.03159)，2022 | 轮式人形平台上结合LQR与集成SAC，在MuJoCo中考察参数变化和样本效率 | “LQR＋RL”和样本效率收益已有先例，不能靠改成PPO形成创新 |
| [Action Space Design in Reinforcement Learning for Robot Motor Skills](https://lamarr-institute.org/publication/action-space-design-in-reinforcement-learning-for-robot-motor-skills/)，CoRL 2024，论文集链接见作者机构页 | 涵盖轮腿平台；动作表示的影响与策略初始化、步间行为有关 | 增加同VMC映射对照，并检查初始力矩分布与平滑性度量 |
| [基于TD3-PID-VMC动态混合控制双足轮腿机器人运动优化方法](https://xk.sia.cn/en/article/doi/10.13976/j.cnki.xk.2026.0153)，2026-06-23在线，DOI：10.13976/j.cnki.xk.2026.0153 | 学习PID/VMC动态融合权重，摘要报告仿真与硬件实验 | 区分“调节控制器融合权重”与“固定基控制器上的差动残差”，不能把混合控制本身作为贡献 |

第一周优先取得上述三篇及残差信赖域论文全文，补齐动作定义、观测、基控制器、约束、训练预算与公开实现；未获得的信息标“未核实”。近期文献的不同机器人成功率不能直接与本机数值横比。

拟验证的贡献只有两项：

1. 在已有差动控制通道上构造低维PPO残差参数化，检验其是否能补偿固定反馈难以覆盖的不对称接触动态；不预先声称减小了纵向干扰。
2. 用同VMC映射、匹配信息与预算的对照和机理实验，评估该参数化整体的姿态—速度折中、样本效率及失效边界。未加入同维数非差模对照时，不把收益单独归因于“物理差模先验”，也不声称排除了普通降维的作用。

补充全文核对：[受约束残差控制系列](CRRL_FULLTEXT_REVIEW.md)。绝对/相对残差约束及串级残差已有公开研究；本机的余量λ不是学习的自适应安全边界，也不能继承这些论文的稳定性结论。

力矩余量缩放属于必要实现，不先包装成第三项创新。若查新发现相同动作分解与目标问题已有充分覆盖，或优势仅来自更强的手工增益，必须缩减主张或更换问题；不靠换名字宣称原创。

已建立 [文献矩阵](LITERATURE_MATRIX.md)，逐项区分正文、摘要及未核实内容。Hybrid LMC的关键方法已核对；动作空间设计与中文TD3-PID-VMC已完成全文核对，残差信赖域一篇仍缺全文；详见 [全文核对](LITERATURE_FULLTEXT_REVIEW.md)。方法新颖性在完成全文比较前保持“待确认”，不将工程准入通过扩大为查新完成。

## 4. 方法设计

### 4.1 名义控制器与物理模型分离

物理模型使用当前episode的质量、惯量、摩擦与驱动参数；控制器设计模型固定为7 kg名义模型。冻结全部名义增益表、平衡前馈、平衡角、`mass_scale`、支撑/摆腿/航向缩放及名义执行器参数，不能只固定K。按测得腿长查询固定增益表可以保留，按随机质量重新设计或调整前馈不可以。

审计初始化及每步控制的参数读取：随机化参数只进入物理系统和评估记录，不直接进入Actor、基控制器或余量计算；基控制器只使用约定的状态观测和固定设计参数。B1调度只依赖各方法均可见的状态。若另设“已知真实参数”控制器，只作为明确标注的oracle辅助对照，不混入未知参数主结果。

验收须包含：改变物理质量后，设计参数快照保持一致；相同观测、动作和控制器状态下，各质量模型得到相同的控制命令，随后物理响应可以不同。模型运行索引与状态仍属于各自环境，不能因复用名义设计表而共享可变控制状态。

### 4.2 候选动作及适用边界

约束前控制关系为 `τ_cmd = τ_base + λ B(q) S a`。τ_base来自冻结的VMC＋LQR；a为PPO的3维归一化动作，S是固定量纲缩放矩阵，B(q)包含差动分配与VMC映射。三个候选物理残差为：

| 动作 | 对左右侧的作用 | 研究用途 |
|---|---|---|
| ΔF | 左腿支撑力＋ΔF，右腿−ΔF | 修正不对称接触下的roll和负载差 |
| ΔT | 左腿虚拟摆腿力矩＋ΔT，右腿−ΔT | 调整左右腿差动响应 |
| Δτw | 左轮＋Δτw，右轮−Δτw | 抑制航向瞬态和轮差速 |

B(q)沿用现有VMC雅可比映射。左右腿虚拟量的和保持不变，不等于世界坐标合力/合力矩不变；接触和构型仍存在耦合，不能宣称严格解耦。

两腿构型不同时，相反虚拟输入可能产生不同实际作用；单侧离地时作用路径改变；增重或双侧同时爬升可能需要共模支撑，纯差模不能直接提供。先做小型机理实验：平地、左右高度不同、单侧接触变化三类条件下，各通道分别施加正负小脉冲，与零残差同初态配对，记录roll/yaw/pitch、速度、实际力矩与接触变化。脉冲幅度和时长在训练前固定，报告限幅事件；据此核查符号、交叉耦合和适用范围，不据单次脉冲推导全局稳定性。

### 4.3 最终输出链与更新频率

根据当前电机转速和名义扭矩边界求λ∈[0,1]，统一缩放残差，使最终六路输出在边界内；不分别裁剪六个残差导致其比例变化。基控制器先应用自身边界；若基输出不可行或映射无效，残差归零并记录事件。该机制只保证输出约束，不提供全局稳定、安全不变集或永不倾倒保证。

每个低层周期的顺序必须是：**完整基控制输出（含六状态覆盖）→残差映射与余量处理→最终执行器约束→物理步进**。基输出可行化与残差余量使用本步开始时的同一执行器状态，峰值力矩计时只按最终输出更新一次；禁止原控制与残差各计一次，也不能只统计基控制输出。日志同时保存请求残差、λ、最终输出及最终约束触发情况。若最终约束改变残差比例，必须披露，不能仍称实际输出严格保持差模。

起始残差尺度作为验证集超参数：ΔF约±10%名义单腿静态支撑力、ΔT约±1 N·m、轮差矩约±0.3 N·m。它们不是实测安全边界；先检查符号与量纲，调参后冻结，不随测试质量改尺度。虚拟动作变化率限制只约束虚拟指令，不等价于电机力矩变化率限制；各残差对照采用相同处理规则，最终力矩变化率另行记录。

PPO初版用标准实现和小型MLP（如2×64），不改PPO损失，不加Transformer、多个教师或复杂历史编码器。策略频率候选50 Hz，每20 ms产生一次动作，期间保持虚拟动作目标；基控制器、当前构型B(q)、转速余量λ与最终约束均以2 kHz更新，不能把40个物理步前的映射或力矩边界一直保持。若启用虚拟动作限速，在低层追踪所保持的目标。所有对照使用相同频率，奖励按物理步积分、姿态峰值逐物理步统计。改变低层周期须单独复验，不能直接复用K后宣称等价。

### 4.4 观测与奖励

观测统一包括：姿态/角速度、车体速度及命令误差、关节角/角速度、轮速、左右腿长/腿速、上一动作；可叠加短历史。当前版本主实验允许仿真状态观测，明确为理想状态反馈研究；不提供地形真值、摩擦真值或未来障碍信息给Actor。各对照可见信息一致，不使用特权Critic作为首版额外组件。

2026-10-02新高度候选采用`request_state_v1`，统一38维：原32维包的物理量、目标和已执行残差保持场景设定延迟；末6维为当前已完成子步的滤波虚拟请求`[F_L,F_R,H_L,H_R,W_L,W_R]`，使用原动作归一化尺度，在映射/λ投影之前记录。diff3按左右反号展开，virtual6按原六通道顺序记录；第26轮torque6末6维按`[alphaL,betaL,alphaR,betaR,wheelL,wheelR]`记录归一化滤波电机请求。各方法可见相同的自身请求信息与时序，坐标/单位分别冻结，不能说所有末6项都是虚拟输入；不用当前J逆变换给B2额外无延迟物理状态，零残差相应为零。新增量是自己的已知执行请求，不是新环境真值；λ每步重算，不是另一个持久状态。该接口修复请求被投影为零时的执行信息歧义，但延迟、未知接触/参数及未展开的基控制滤波仍使问题部分可观测，不宣称完整Markov。旧32维检查点/归一化不得直接套用，须独立冻结新版本；旧NativeEnv默认保留legacy32用于历史复现。观测接口检查不能代替完整任务、鲁棒性或训练准入。

论文若要声称“仅本体传感器”“无真值依赖”，必须把整个闭环（包括LQR）改为估计输入并复验，而非只隐藏Actor输入。传感器估计与ROS2不作为本次纯仿真研究启动前置条件。

奖励仅围绕速度/航向跟踪、平台roll/pitch、残差幅度与平滑性、倾倒/未越障惩罚。机械功率积分只作为机械做功指标，不称实际电耗；不要用人为设定的电池效率得出续航结论。

## 5. 核心对照与递进消融

| 编号 | 方法 | 必须回答的问题 |
|---|---|---|
| B0 | 当前已优化VMC＋LQR，采用9月17日修复：航向阻尼2.0、归一差矩上限0.12 | 学习是否优于真实强基线？ |
| B1 | 基于共同观测的手工反馈/阻尼调度，本轮冻结为航向PD＋roll调度，认真分配验证调参预算 | 是否简单手工调度就能解决？ |
| B2 | VMC＋LQR＋6维原始电机力矩残差PPO | 与B2-V比较虚拟动作坐标及其映射是否有帮助？ |
| B2-V | 独立输出ΔF_L、ΔF_R、ΔT_L、ΔT_R、Δτw_L、Δτw_R的6维虚拟残差PPO | 与M比较，差动子空间参数化整体是否有帮助？ |
| M | VMC＋LQR＋差模残差PPO，起始3维，最终维数由消融确定 | 本文候选方法 |
| B3（次优先级） | 纯PPO六路输出，保留相同物理执行器约束 | 经典先验的样本效率价值？ |

B2-V与M复用同一VMC映射、名义基控制器、低层更新频率、余量缩放和最终约束。B2与M直接比较同时改变维数、坐标、映射和可达集合，不能独立识别差模收益。核心资源优先用于B0/B1/B2/B2-V/M，B2-V优先于纯PPO；若预算不足删去B3，就不主张相对纯PPO的优势。

所有RL方法使用相同训练/验证/测试划分、相同可见状态、场景分布和策略频率；报告网络规模、物理步数、策略步数和墙钟时间。允许各方法在相同验证预算内调参，不能只优化本文方法。B3在相同预算失败不能解释为“纯PPO永远不可行”；报告预算限制。

公平性补充：残差幅度和平滑性在统一六路执行器空间度量，使用固定名义力矩尺度归一化，并同时记录最终总力矩变化；不直接比较3维与6维动作向量的平方和。训练前在同一组状态上检查随机初始策略的映射前后力矩分布、λ和饱和比例，按相同验证规则调整尺度/初始方差并冻结，报告无法消除的可达集合差异。检查点选择规则、验证次数和超参数预算一致，测试冻结观测归一化统计。 第27轮采用已预注册的三方法pilot RMS几何平均作为机械校准目标，同一有界求解规则只调整初始log_std；不以任务成绩选择尺度。888同状态审计核实际增量电机力、总电机力、λ及裁剪，不将固定状态20ms输出当PPO滚动状态分布。拟合与开发整体接近不消除高度分组/协方差/秩3对6差异；主要结论仍限于动作参数化整体，正式入口必须显式应用并冻结该配置，不能默认沿用std=1或称严格分布等同。

最近论文若能取得实现，加入可比设置；无法复现时明确区分自建通用基线与原论文方法，不冒充复现。

优先递进消融：M1仅Δτw → M2为ΔF＋Δτw → M3完整三维。使用相同场景、预算与约束；若ΔT没有稳定额外收益，采用二维，若ΔF也无贡献则采用一维，不为题目强留通道。根据验证结果在正式测试前选定M并冻结。

次优先级消融为关闭参数随机化、λ统一缩放对比同样有最终电机限幅的常规裁剪。B2-V与M仍存在维数和探索分布差异；两个月版本将结论限定为“差动动作参数化整体有效”。若要进一步归因于物理差模而非一般降维，再加入同维数非差模动作子空间对照。

## 6. 数据与实验协议

训练、验证、测试按场景生成种子及地形布局分离，不能将同一轨迹相邻片段分到不同集合。先冻结生成器、判据与参数范围，再训练；测试集只作最终评估。

### 6.1 场景与泛化类型

按固定课程先验证单一高度/接触时刻差，再加入左右摩擦差和驱动差，最后加入质量及延迟组合；所有RL对照使用相同课程和样本预算。保留平地、对称凸台、一致摩擦对照，检查名义性能是否退化。两侧接触时刻差由凸台纵向错位产生，记录实际触台时差，不人为指定接触真值给Actor。

建议第一周验证后冻结的仿真范围（是实验设计，不是硬件实测）：

| 因素 | 训练候选 | 独立测试/外推候选 |
|---|---|---|
| 单侧/错位凸台高度与接触时差 | 0～20 mm；左右纵向错位候选0～0.10 m，左右先接触均覆盖 | 同规则新布局为同分布；25/30 mm高度为范围外推 |
| 速度命令 | ±0.5～±1 m/s | 相同范围新组合；转向单列 |
| 左右摩擦μ_L、μ_R | 分侧取0.6～1.0，显式包含μ_L≠μ_R及相等对照 | 同规则新样本；一侧0.4或1.2、另一侧名义值的范围外推，并交换左右 |
| 总质量 | 7.0～7.5 kg | 7.6～8.0 kg；新增载荷位置变化单列 |
| 左右驱动力差 | 0～±3% | ±5%压力点 |
| Actor观测通道延迟 | 0～10 ms，按0.5 ms物理步量化的episode内固定延迟 | 15/20 ms同链路压力点；不称全闭环延迟鲁棒性 |

摩擦实验须核查最终接触对实际使用的摩擦系数，不能仅修改某个geom却被接触组合规则覆盖。延迟首版只在Actor观测输入的带时间戳物理步缓冲实现，LQR继续读取当前约定状态，动作即时到达并保持20 ms；缓冲在reset清空。若以后研究全闭环观测延迟或执行器延迟，需单列实验、对应延迟整条链路并复验，不能沿用Actor延迟结论。

测试命名必须按实际分布划分：同范围同生成规则的新种子是**同分布泛化**；**组合外推**要求训练和验证明确排除某个联合区域，例如“高凸台且大左右摩擦差”，阈值、保留区域和采样规则训练前冻结；所有参数独立连续随机化后的新组合通常仍为同分布。超出范围的是**范围外推**；接触/求解器变化是**模型敏感性**。若未真正留出组合区域，删去组合外推主张。

接触外推必须区分参数变化与模型形式变化；最终附加一种合理的接触/求解器设置敏感性复验，禁止用更松的约束或更深穿透制造提升。若未完整启用机体/连杆碰撞，评价只覆盖启用碰撞体，不声称真实机械可通过性。

### 6.2 成功判据与统计口径

原协议唯一主指标为**预先固定困难集的任务成功率**（2026-09-20用户确认改题后，由第6.4节替代主指标与继续门，本节任务成功定义保留）；约束指标为速度RMSE、完成时间与5°姿态/航向限制；解释指标为yaw/roll/pitch的RMS和峰值、恢复时间、残差/力矩饱和比例及样本效率。历史yaw最大4.925°接近门槛，因此必须同时报告连续指标，不能只展示跨过5°后的成功率变化。

直线越障协议已按首周开发校核冻结为 `preppo-v2-ramp`，见结果目录的 `protocol.json`。原候选阶跃起步/1 s停车在无障碍也失败，原结果保留为 `candidate_v1_*`；因此统一使用1 s有界速度参考斜坡和2 s停车观察，不改变5°、20%、0.60 m、0.03 m/s阈值，不改控制增益。该变化属于任务协议修正，不算控制性能提升；所有方法使用同一协议，最终测试未运行。

- 从统一站立初态执行1 s准备，再用1 s线性斜坡由0升至巡航命令，随后恒速，沿命令方向驶至预设终点。终点须位于最远障碍后至少0.5 m；d为起点到终点的有向距离。通行截止时间为 `1 s＋0.5 s＋1.5d/|v_cruise|`，未到达即失败。
- 速度合格条件为行驶段相对当时参考速度的RMSE不超过 `0.20|v_cruise|`，从斜坡开始至到达终点或失败逐物理步积分，包含全部起步和接触瞬态。不能排除减速区间；准备段及终点后的停车另行报告，不混入该RMSE。
- 全任务roll/pitch绝对值及相对初始航向的yaw误差峰值均不超过5°，且无倾倒、非法输出或已启用非轮碰撞体触地。转向任务另列，不套用零航向参考。越障必须实际接触目标障碍并通过终点；无障碍对照只要求完成行程。
- 到达终点后保持2 s停车观察，停车距离≤0.60 m、末0.5 s速度≤0.03 m/s，姿态门覆盖观察段。成功需同时满足以上条件，不能靠提前结束规避恢复或停车失败。

恢复时间以最后一次目标障碍接触结束为起点，至姿态/航向误差≤1°且速度误差≤行驶命令幅值10%持续0.5 s为止；终点停车后改用零速度参考及≤0.03 m/s速度阈值。窗口内未恢复的记录为“未恢复”，不能按0计。阈值与统计窗口同样在训练前冻结。

正式主要RL方法计划5个独立训练种子，每个策略每类200个固定测试episode；探索仅用3种子/较小样本。同一套场景用于不同方法，保留每个训练种子的独立结果和场景明细；B0/B1为固定控制器，不虚构训练重复。先按种子计算成功率及连续指标，再汇总均值与95%区间，写明区间计算方法；可按训练种子为簇重采样，不能将episode数量当作训练重复，5个种子的区间也不保证充分精确。[统计评估参考](https://arxiv.org/abs/2108.13264)

失败episode一律计入主指标分母。连续指标报告两组：所有轨迹终止前的峰值/RMS连同实际时长和失败时刻，以及完整成功轨迹的条件统计；不得把提前翻倒的短轨迹RMS直接当作优于完整轨迹的证据，也不以零填充失败后的缺失数据。完整轨迹条件统计仅用于解释，不能替代成功率和失败分析。学习曲线使用相同验证间隔，测试集不用于挑选检查点。

**2026-09-17可达性核对：修订B0在固定12例困难开发组已12/12；以下原成功率提升10个百分点的继续门在当前集合不可达。保留历史门槛，不将辅助yaw指标自动升级为主指标。长训练前须先解决研究问题与验证协议的天花板，详见CRRL_FULLTEXT_REVIEW.md第4节。原回归集和最终测试集保持不变。**

**2026-09-20开发扩展核对：** 按运行前固定的现有validation生成规则、stage=3、种子1000～1023完整评估24例，修订B0为23/24（95.83%）。唯一失败development_1023的yaw峰值5.186°；该组即使M全成功，相对B0最多提高4.17个百分点，仍不足原10个百分点门槛。此组仅为开发诊断，不替换正式验证集、不用于最终测试或挑选失败子集，也不证明总体饱和。运行前后历史基线源码哈希一致。见[预注册与复现](tools/results/development_extension_2026-09-20/PREREGISTRATION.md)、[完整结果](tools/results/development_extension_2026-09-20/baseline.json)。

第4周原定的唯一主要继续门：在预先固定的困难**验证集**上，M的平均成功率相对B0/B1/B2/B2-V中的最佳对照提高至少10个百分点，且3个预实验种子均相对B2-V方向一致，名义场景速度/姿态无明显退化。yaw RMS降低15%只作辅助证据，不能替代主门；门槛是继续投入的工程目标，不是显著性证明或录用标准。最终困难测试集继续封存，正式5种子结果报告区间，不据预实验宣称研究成功。

若仅B2-V优于原始电机力矩残差而M无稳定收益，则将贡献改为虚拟动作参数化并重新核对已有动作空间研究；若B1已达到同等性能，重新判断学习模块必要性。均不靠堆模块、测试集调参或降低主门维持原结论。

### 6.3 Gym环境的训练准入检查

正式训练前留下可重复运行的最小检查：零残差逐步复现原控制轨迹；同种子reset与rollout重现；reset清空滤波器、停车锚点、上一动作、延迟缓冲及峰值计时；并行环境不共享可变控制状态；超时截断与倾倒/任务失败终止正确区分；奖励积分、峰值与失败检测覆盖每个策略步内全部物理子步。另按第4节检查设计参数无泄漏、残差不被覆盖、限幅不被绕过和峰值不重复计时。零残差一致性在同一名义模型与原协议下验证，不与新成功阈值混为一谈。

### 6.4 航向精度协议 v1（2026-09-20，用户确认改题后冻结）

**生效范围与历史：** 用户明确选择保留PPO并改为航向精度研究。本节替代第6.2节原成功率提升10个百分点的研究继续门；第6.2节的任务流程、成功定义和失败报告规则继续有效。改题发生在正式学习前，依据已公开的B0开发结果（12/12及23/24），不是独立于开发证据提出的假设。旧结果保留，不能作为新协议的独立验证。题目收敛为“面向不对称接触航向精度的双轮腿机器人差动残差强化学习控制”。

**唯一主要问题：** 在相同场景和预算下，差动残差参数化能否在不损害任务通过能力、姿态和速度的条件下，比充分整定的经典/手工调度和通用残差降低完整越障任务的航向误差？样本效率、roll/pitch及恢复时间为辅助结果；不再承诺成功率提高10个百分点。

**主指标：固定参考时长归一化的航向误差 Jψ（°，越小越好）。** 对场景i，以第6.2节的通行截止时间加2 s停车观察作为固定Ti，即 `Ti = 3.5 + 1.5 * (center + abs(offset)/2 + 0.75) / abs(speed)`。积分覆盖准备、行驶和完整停车观察的每个0.5 ms物理步，`Jψ_i = sqrt(∫ eψ(t)^2 dt / Ti)`，eψ为相对初始零航向的误差（°）。Ti只依赖场景，不随控制器运行时间变化；这是任务累计平方误差的归一化，不称普通实际时长RMS。现有日志在当前固定0.5 ms步长下可还原为 `rms_deg[2] * sqrt(physical_steps * 0.0005 / Ti)`。普通RMS、实际时长、到达时间及峰值同时保留，不对未模拟时段补零。

**失败处理：** 24例开发结果显示reason=completed不等于success。5°等约束失败且完成全程者仍计入Jψ和失败分母；fall、timeout、invalid、非轮接触等提前终止者不生成可用于方法比较的Jψ，不删除或零填。任何待比较方法/种子在门控集有非completed轨迹，该轮不通过精度继续门，完整报告失败；不能以另一方法未完成为由宣称航向精度优势。此规则保守，可能阻止比较，但避免提前终止得到低误差。

**数据冻结：** 全部沿用 `sample_scenario('validation', seed, stage=3)`，不筛选难例、不改变参数范围、不包含组合保留区。新选择集32例：seed 2000～2031；一次性研究门控集64例：seed 3000～3063。旧validation_0～11、开发1000～1023及28例回归均为已见开发资料，不能充当新门控集。场景明细、生成器/控制源码哈希见[冻结清单](tools/results/yaw_precision_v1_2026-09-20/protocol_manifest.json)。清单只生成参数，尚未运行新场景。所有test_*最终划分不调用、不改动。

选择集用于B1整定、共同残差尺度选择及检查点选择；门控集在所有方法配置和每种子检查点锁定后只打开一次，不用于调参。不得看完B0门控结果后另选种子或门槛。首次门控失败即报告不通过；后续修改必须成为另一个明确披露的探索研究，原门控集降为开发资料。

**公平预算和选择：** 主比较仍为B0/B1/B2/B2-V/M3，共同名义基线固定；暂不同时做M1/M2选维和B3。B1至多8个预先列明的手工调度候选，零调度必须在候选中；候选只读共同可观测状态，具体公式/参数表在第一次选择集评估前记录。RL各方法仅1套共同初始超参数配置、预实验种子1609/1610/1611，各200万策略步上限，每2万步在相同32例选择集评估一次（最多100次）。前20万步是同一训练运行的链路/趋势检查点，不重启预算、不据其结果追加候选。训练频率、网络、课程与共同尺度须在开始前落盘；本节不表示训练入口已经实现或已获准跳过准入。

选择规则先排除存在非completed轨迹的检查点，再按失败数最少、32例平均Jψ最低、步数最早的顺序选择；同值按先记录者。B1使用同一规则。若没有完整检查点，报告该方法本预算内未通过，不隐藏失败。B0/B1不虚构训练种子。测试冻结归一化统计并用确定性动作。额外调参须对各RL方法同额增加预算并在门控前修订注册，不能只优化M。

**唯一精度继续门：** 每个RL训练种子先对64例等权平均Jψ，再对3个种子等权平均得到方法分数Q；B0/B1各自对64例等权平均一次。取B0/B1/B2/B2-V中最低Q作为Qref。须同时满足 `(Qref-QM)/Qref >= 15%` 且 `Qref-QM >= 0.05°`，并且3个同编号种子的M均低于B2-V。Qref=0时不通过提升门，不加任意小常数。这是预先选择的研究投入门槛，不是已有文献阈值、统计显著性或已证明的应用需求；绝对门避免在接近零误差时放大百分比。

**约束同时成立才放行：** 每个M种子在64例的成功数不得低于B0、B1及同编号B2/B2-V的最大成功数；另外B0成功的每一例M仍须成功，不能用不同失败互相抵消。M各场景的速度RMSE不得超过B0的1.05倍加0.005 m/s，到达时间不得超过B0的1.05倍加0.05 s。旧28例回归须全部成功；其速度RMSE同样不超过原修订B0的1.05倍加0.005 m/s，roll/pitch峰值不超过原B0峰值加0.1°，且原5°等硬成功阈值保持。这些是开发阶段经验非退化门，不是统计非劣效性证明。不可行基输出/最终限幅统计完整披露。

**停止与正式实验：** 门控不通过则缩题或报告负结果，不把辅助指标替换主指标。门控通过才进入原五种子正式实验；正式种子1609～1613，前3个预实验运行可作为正式运行复用，但须全部报告、不得删种子；正式配置不再基于最终集选择。各测试类别分别报告5个种子的主指标、成功率和区间；任何未完成轨迹按上述规则报告该类别主比较未通过，不用成功条件均值替代。该五种子汇总仅是条件于已选配置的证据，不是新的独立算法选择验证。M1/M2及OOD保持后续独立消融地位。

**本轮验收与下一项：** 已冻结数学定义、数据编号、效应门槛、失败处理和选择预算；未改奖励、控制器或执行器链路。下一项实现最小离线评分与检查点选择检查，冻结B1的至多8个候选及RL配置，然后只在选择集评估B0/B1，核查新主指标是否有足够的绝对误差空间；若选择集最强经典对照Q<0.05°，停止学习投入，不改门槛。门控集和最终集继续封存；全文查新缺口仍限制新颖性主张。

## 7. RTX 5060与训练预算

主机已核验为i7-14650HX、24逻辑CPU、约31 GiB内存。9月16日用户重装驱动后，主机识别RTX 5060 Laptop、约8 GB显存、驱动580.178.04；已安装PyTorch 2.14.0+cu130并通过M/B2-V的CUDA短更新、重载及CPU读取检查。早期执行环境曾限制GPU访问；9月21日已在当前环境完成CPU/CUDA配对实测，GPU实际可用，但本方案完整采样与更新仍以CPU更快，详见第12.10节。GPU就绪不等于MuJoCo物理仿真已迁移GPU。

先使用现有MuJoCo＋Gymnasium＋Stable-Baselines3，测1/4/8个CPU环境的策略步吞吐，再比较网络训练用CPU/GPU。官方文档指出非CNN的PPO常更适合CPU；有GPU不等于现有Python控制/接触仿真自动GPU并行。[SB3 PPO说明](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

`jianyi.md` 提供的单进程名义站立微基准为：模型加载约0.11 s，四工作点控制器初始化约0.185 s，2000次控制＋物理步进约0.485 s；按40物理步/动作折算约103策略步/s，200万步仅采样约5.4 h。此处为建议文件中的初测记录，本轮未复测，不作为正式训练吞吐；未包含Gym、网络推理、PPO更新、复杂地形与评估开销，也不意味着200万步能收敛。

第一周测reset、一次step、完整采样与一次PPO更新的耗时。名义LQR工作点设计放在初始化/只读缓存，episode仅重置运行状态；随机化时冻结第4节列出的整套名义设计参数，不只固定K。记录完整训练链的1/4/8环境吞吐、内存、CPU/GPU选择及所用软件版本。

先用20万策略步做闭环/数值检查，再约200万步做单种子学习曲线；正式预算由实测确定。估时：总策略步数÷实测吞吐＋优化/评估时间。所有方法和消融的总预算须能在第6周前跑完，不能凭显卡型号承诺训练天数。

只有实测完整CPU训练预算使8周计划不可行，才考虑MJX。当前官方区分MJX-Warp与MJX-JAX，不能将二者功能限制混用；MJX-JAX列有椭球与箱体等碰撞组合限制，本机恰有椭球轮胎和箱形凸台。迁移前按选定后端和版本核验闭环五连杆、该碰撞对、接触/求解器及控制计算，完成动作和物理等价检查，不能假定无成本迁移或偷偷换碰撞形状后沿用旧结果。[MuJoCo MJX文档](https://mujoco.readthedocs.io/en/stable/mjx.html)

## 8. 八周里程碑与停止门

目标为2026-11-15前完成一次合适期刊投稿，不等于两个月内录用。

| 周 | 日期 | 交付物与出口 |
|---|---|---|
| 1 | 9/15～9/21 | 直接相关论文矩阵、困难场景B0、脉冲机理检查；冻结整套名义设计参数与指标/数据划分；Gym准入检查、零残差一致性和实际吞吐 |
| 2 | 9/22～9/28 | M三维差模与B2-V六维虚拟残差均得到首条学习曲线；输出链/2 kHz映射/约束验收；保存重载、归一化冻结与确定性评估 |
| 3 | 9/29～10/5 | 充分调参B1，补B2原始力矩残差；B0/B1/B2/B2-V/M关键对照与M1→M2→M3递进消融，启动三种子预实验 |
| 4 | 10/6～10/12 | 完成三种子验证集比较，按第6.4节航向精度与非退化继续门判断；选择并冻结最终动作维数、检查点规则和正式预算；不达标缩题/改假设 |
| 5 | 10/13～10/19 | 核心方法正式五种子与必要消融；有余量才加入B3纯PPO；同步写方法和实验协议 |
| 6 | 10/20～10/26 | 同分布/明确留出的组合/范围外推/接触敏感性分别评估，逐种子区间统计与失败分析；冻结所有结果 |
| 7 | 10/27～11/2 | 中文或英文完整初稿、图表、复现清单、导师审阅 |
| 8 | 11/3～11/9 | 根据审阅修订、选刊与格式检查、准备投稿材料 |
| 缓冲 | 11/10～11/15 | 补关键实验或完成投稿；不承诺审稿周期 |

## 9. 论文结构与图表

结构：引言与相关工作→名义控制器及不对称接触问题→差动残差动作映射与PPO→实验协议→对照/递进消融/泛化→局限与结论。

必要图表：机器人与最终保留动作映射、名义设计/随机物理模型分离及最终输出控制框图、三类正负脉冲响应、接触事件与yaw/roll/速度曲线、跨种子学习曲线、包含B2-V的主结果表、左右摩擦差/障碍/质量性能曲线、递进消融、失败样例与力矩余量图。图中标注统计窗口、重复数及区间，视频只作补充，不能替代数据。

数学部分推导虚拟力映射、差动子空间、残差幅值/变化率约束；若没有严格闭环证明，不使用“全局稳定”“安全保证”等表述。不为凑理论加入未验证的Lyapunov/CBF结论。

## 10. 投稿策略

“北大核心及以上”不是SCI、EI、CSCD与北大核心之间统一可排序的等级；需以学校和导师认可目录为准。两个月可规划到投稿，录用由审稿决定。

中文优先候选《机器人》：研究主题匹配，官网已有控制与强化学习融合文章；纯仿真稿必须靠方法贡献、严谨对照和可复现性支撑，不能据此保证接收。[官网](https://robot.sia.cn/) / [近期融合控制文章](https://doi.org/10.13973/j.cnki.robot.250163)。其官方历史封面标有中文自然科学核心/Ei等收录，但本轮官网收录页未返回正文；投稿前用学校认可的最新北大核心目录核对，不将历史名单冒充2026认证。[官方历史封面](https://robot.sia.cn/fileJQR/journal/img/cover/20203ml.pdf)

《控制与决策》可作强调控制方法与统计验证的中文备选，具体收录及纯仿真适配性仍需正式核验，不先承诺级别或录用概率。若英文写作、跨种子/OOD和方法贡献充足，再评估Robotics and Autonomous Systems或Control Engineering Practice；已有相关论文常含实机，当前纯仿真条件意味着证据压力更大。这是选刊建议，不是对期刊等级或审稿时长的保证。

首周与导师确定中文/英文及学校认可范围；第4周依据实际结果选定一个主要投稿目标，避免先为多个期刊写多个版本。

## 11. 当前执行状态

已完成的工程项：原始基线冻结、名义设计参数隔离、统一最终输出及计时、五种残差动作接口、32维共同观测与Actor延迟、分侧摩擦及组合留出生成器、Gym准入检查、28场景B0开发基线、18个正负脉冲、初始动作分布、CPU吞吐与短更新链路检查。结果与限制以 [准入报告](PREPPO_REPORT.md) 为准。

未完成：残差信赖域一篇全文查新、正式200万步学习及多种子对照、B1调度整定、正式消融/OOD测试、论文稿件与实机验证。短更新模型只用于检查优化器及保存重载，不是已学成策略；最终测试集保持未使用。

工程准入已通过。2026-09-20用户确认转向航向精度研究，第6.4节已冻结替代协议；2026-09-21已完成离线评分验收、B1候选/共同RL配置冻结、32例选择集B0/B1评估以及M/B2-V/B2各4000步短更新。工程准入记录见下方；20万步链路/趋势检查和200万步学习尚未启动，下一阶段按第6.4节统一预算与选择规则执行。全文查新并行补齐，但不据缺失全文宣称原创。ROS2和实机仍不是当前纯仿真训练前置条件。

2026-09-20推进：已完成上述独立开发扩展评估，长训练继续暂停。下一项是训练前完成研究问题收敛：依据应用需求决定是否改为连续航向误差研究，并事先固定主指标、效应门槛和独立验证规则；若无合理改题依据，则缩题为经典控制与虚拟动作接口的工程验证。不能继续搜集失败样本来凑原成功率门，也不能将本轮唯一失败例升级为主评估集。残差信赖域全文缺口仍保留。

2026-09-20用户确认改题后：第6.4节航向精度协议v1已冻结，原成功率提升门保留为历史。当前未完成项为离线评分验收、B1候选/训练配置冻结与选择集B0/B1评估；在此之前不启动20万/200万步学习。64例研究门控集及最终测试集仍未评估。


### 2026-09-21 长训练前落实（已完成工程验收）

已在第一次新选择集评估前固定[共同训练配置与8个B1候选](tools/results/yaw_precision_v1_2026-09-20/training_config.json)。B1在与Actor相同的延迟观测上，以航向角/角速度PD反馈乘以roll幅值调度系数，仅使用现有轮差矩残差；50 Hz更新、2 kHz原映射/变化率/余量约束。零候选就是B0，全部候选完整评估32例，按既有失败数→Jψ→记录顺序选择。

RL统一采用8环境、每环境250步/rollout（共2000策略步）、batch=250、10 epochs、2×64网络、CPU；每2万步评估可恰好位于第10次更新完成之后，200万步恰好1000次更新，无隐含步数超额。共同课程为前2万步stage1、2万～10万步stage2、之后stage3；阶段切换在下次episode reset生效。共同奖励、初始log_std=0、残差尺度和观测保持原实现。现有稳态随机动作映射报告中的归一化力矩RMS为M 0.02782、B2-V 0.02906、B2 0.03151；不声称完全匹配可达集合，也不把稳态映射结果当真实初始策略全轨迹分布。

[评分/选择集脚本](tools/pretrain_yaw.py)和[训练入口](tools/train_yaw.py)复用同一评分及检查点顺序；[最小检查](tools/test_yaw_training.py)覆盖提前终止排除、完整失败保留、样本缺漏拒绝、检查点优先级和课程/评估时点。训练入口要求显式选择`--smoke`或`--train`，拒绝覆盖已有目录，长训练检查readiness及对应文件哈希。短更新每方法4000步，只用已见开发1000～1003验证保存重载和冻结归一化评估，不消耗32例正式检查点选择预算；不据短更新成绩调参、选模型或宣称收敛。


验收汇总：[readiness.json](tools/results/pretrain_yaw_2026-09-21/readiness.json)。8候选×32例均完成完整任务流程（completed不等于success），没有删除失败轨迹。B0为28/32、Jψ=0.762952°；按预定规则选出B1_6（kp=0.8、kd=0.3、roll_gain=1），同为28/32、Jψ=0.613519°，比B0低19.586%。二者失败编号同为2000/2001/2012/2027；B1未新增B0已成功场景的失败。其余候选与逐例指标保留在[选择集汇总](tools/results/pretrain_yaw_2026-09-21/summary.json)。Qref≥0.05°的训练前空间检查通过，但不能从B1结果推出PPO可达到继续门。

M/B2-V/B2各4000策略步、2次PPO更新均通过参数更新、有限损失、保存重载、冻结归一化和课程reset检查，共12000步仅作工程检查。采样＋更新耗时分别25.585/25.150/21.792 s；这是stage1短更新耗时，不外推为全课程/全部验证的正式训练预算。短更新仅核对旧开发1000～1003，未使用32例选择集选短更新模型。三个方法的模型和日志均标记smoke，不可替代正式学习曲线。

旧控制/环境/配置18文件哈希一致，复用已有28例回归、输出链、5类旧动作和脉冲证据；本轮未修改控制器或物理模型。新增检查覆盖评分与选择规则、更新后评估时点、未完成准入时拒绝长训练、拒绝覆盖已有输出目录。readiness绑定新增脚本、冻结配置和验收文件哈希，代码或配置变化会使长训练入口拒绝旧证据。

下一步可以开始预定学习阶段，先观察同一训练运行的20万步检查点，之后仍按既定总预算和第6.4节门控判断；本轮没有启动20万/200万步。64例门控集和最终测试集均未评估。残差信赖域全文仍因合法访问缺口未取得，方法新颖性仍待确认，工程准入不等于所有研究前置问题关闭。

复验评分/时点检查（项目根目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python wheelleg_ppo/tools/test_yaw_training.py
```

训练入口为 `tools/train_yaw.py`：必须显式指定 `--method M|B2-V|B2`、`--seed 1609|1610|1611`、`--output 新目录`，并选择 `--smoke`（4000步）或 `--train`（冻结200万步预算）。每2万步保存policy.zip、VecNormalize.pkl及完整32例选择集指标；每种子独立保存selection.json。中断留下的检查点不等于完成预算，不得将重启运行冒充无中断继续。最终集评估不在此入口中提供。

2026-09-21最新用户要求覆盖此前“下一步可以开始学习”：暂停20万/200万步，先完成文献矩阵末节的两项研究准入判断。技术入口当前只校验工程readiness，不能将该检查通过当成研究启动许可；方向获确认后才能安排训练。


## 12. 2026-09-21 贡献识别对照定义（环境及v2训练入口已验收）

本节回应用户要求的一维轮差矩和同维非差模对照。第6.4节v1、原配置与样本编号不改写；v2已单独冻结，具体生效规则见第12.5节。按用户最新要求，全文缺口转为研究限制；物理动作分布、新接口局部回归及v2短更新均已完成。不能把“设计用来排除解释”写成“已经排除了该解释”。

### 12.1 固定动作定义

令归一化虚拟输出按 `z=[F_L,F_R,T_L,T_R,w_L,w_R]` 排列，实际虚拟输出为 `diag(Fs,Fs,1,1,0.3,0.3) z`，Fs沿用名义单腿支撑力的10%。记dF=(1,−1,0,0,0,0)、dT=(0,0,1,−1,0,0)、dw=(0,0,0,0,1,−1)；cF=(1,1,0,0,0,0)、cT=(0,0,1,1,0,0)。所有动作分量为[-1,1]，不增新网络或更改PPO目标。

| 方法 | 虚拟动作映射 | 用于回答 |
|---|---|---|
| B0/B1 | 原冻结基线/选定B1_6，保持原候选结果 | 学习是否优于本预算内简单手工反馈 |
| M1 | z=dw·a，仅一维轮差矩，轮幅值仍±0.3 Nm/侧 | 学会轮差矩反馈是否已足够，腿部通道是否有额外价值 |
| M3 | z=[dF,dT,dw]a，原三维差模 | 本文候选参数化 |
| N3+ | z=[cosθ·dF+sinθ·cF, cosθ·dT+sinθ·cT, dw]a，θ=30° | 同样三维、同样轮差矩能力，但腿部同时包含共模 |
| N3− | 上式sinθ项变为负号，θ仍30° | N3+的左右镜像，避免选择对某一侧不利的基 |
| B2-V/B2 | 原六维虚拟残差/原电机残差 | 完整虚拟空间与电机坐标的背景比较 |

θ=30°在新增训练/场景评价前选定，不搜索角度，不按成绩挑正负号，N3±都保留。两者腿部系数为1.366025和−0.366025的相应左右组合；轮通道严格沿用dw，防止用没有航向控制能力的纯共模对照制造优势。左右镜像等价于交换两侧并翻转动作符号。

[矩阵清单](tools/results/contrast_design_2026-09-21/basis.json)和[可运行代数检查](tools/results/contrast_design_2026-09-21/check_basis.py)验证：三种三维矩阵均有Gram矩阵2I、秩3；N3±在腿部的共模投影非零，因此不属于原M3子空间内部的坐标旋转；轮列完全相同。检查不运行环境，也未使用门控/最终集。

### 12.2 物理匹配的边界与训练前检查

相同Gram矩阵只保证归一化虚拟空间的列能量相同。VMC雅可比、不同腿长和执行器限幅之后，电机协方差不可能因此自动相同；N3±单侧腿输出可达原单通道尺度的1.366倍。这一差异必须披露，不能再宣称各侧虚拟幅度完全一致，也不能私自逐侧裁剪以改变注册子空间。

所有方法使用同一名义设计、VMC映射、50 Hz策略、2 kHz更新、公共λ、最终限幅、奖励物理积分和观测。M1仅减少输出维数，保持隐层配置；报告准确参数量，不能称网络参数量绝对相同。物理检查复用已见的三类脉冲状态，以相同固定种子610、256个标准正态并裁至[-1,1]的动作，检查请求/实际六电机RMS、协方差、λ、饱和率及零动作一致性。M1与M3的轮随机分量成对复用。不得用32例成绩选择更有利的初始化或基；分布差异若显著，应在训练前说明或修订尺度并留档，不能在学习后补救。当前物理检查已通过实现一致性验收，具体差异与限制见第12.4节。

### 12.3 判据、预算及可写结论

新增对照拟沿用32例选择集、64例一次性门控集以及第6.4节的指标/非退化规则，保持测试封存。B1_6不再因新增方法重新整定；原8候选成绩不覆盖。M1、M3、N3±、B2-V、B2共6个RL方法若各3种子×200万步，预实验总预算为3600万策略步（原3方法为1800万）；每方法同样最多100次×32例选择集评估，总计57600个评估episode。此处只是预算明示，未授权或执行这些训练，不能声称增加对照无成本。

下一版的结构贡献继续门应将M1和N3±加入最强对照集合，仍要求M3相对最佳对照同时降低至少15%和0.05°、满足原全部非退化约束；3个配对种子对M1、N3+、N3−及B2-V均应方向一致。只有在研究准入和完整v2配置冻结后，才可执行此修订，不能把v1门控结果事后重算成另一主门。

- M1不优于B1：没有证据认为轮差矩反馈必须用RL；不宣称学习必要。
- M3不优于M1：三维腿部通道贡献不成立，缩为一维研究或报告负结果，不强留三维题目。
- M3优于B2-V却不优于N3±：可能是普通降维、探索或约束形状解释，不能归因差模。
- M3优于B1/M1、两个同维混合空间和B2-V，且物理统计没有明显未解释偏置：可支持本任务中该差模子空间的贡献；仍不能排除所有手工反馈/所有三维子空间，不能证明全领域新颖性。

新检索的非学习控制文献还提示，持续偏置应考虑积分补偿、噪声问题应考虑输出平滑。本机当前为理想状态反馈且目标是接触瞬态，不能直接把其他论文的控制器插入作为“复现”。若已见开发失败诊断显示剩余误差主要是持续偏置，应先单独注册积分/扰动补偿经典对照再学习，不能以仅8个PD候选宣称穷尽经典方法。

### 12.4 接口与物理分布验收

`ppo_env.py`增加`mixed3_plus/minus`两个三维动作模式，只在原虚拟动作构造处应用已注册的±30°混合，之后复用原VMC及最终输出链。M1复用现有`diff1`。修改前环境源码保存在[原环境](tools/results/contrast_physics_2026-09-21/ppo_env_before.py)，其哈希与v1清单一致；其余17个原清单文件未变。

验收脚本为[check_contrast_physics.py](tools/check_contrast_physics.py)，[结果](tools/results/contrast_physics_2026-09-21/check/summary.json)与当前相关源码哈希逐项匹配，故复用已完成检查，不重复运行。检查覆盖：三类实际状态中的新映射对独立virtual6参考、旧5种映射逐值一致；全部7种动作模式各80物理步的零动作轨迹/力矩/峰值计时对旧环境逐值一致；N3±各600物理步非零输入的最终限幅与状态有限性；新增模式及M1的Gym观测/奖励与非法输入拒绝。

| 已见状态 | N3+实际归一化RMS / M3 | N3−实际归一化RMS / M3 | 三者饱和比例 |
|---|---:|---:|---:|
| 平地 | 1.000000 | 1.000000 | 0 |
| 不等腿高 | 1.000215 | 0.999785 | 0 |
| 单轮接触 | 1.000106 | 0.999894 | 0 |

三类状态×6方法×256个固定样本均经过真实基控制/余量/最终限幅链，记录请求与实际残差的每电机RMS和协方差；该批静态样本λ均为1，无基输出阻塞或额外最终裁剪。M1与M3共享同批轮动作，M1没有两个腿通道，不能要求其总力矩RMS等于M3。

整体归一化RMS最大相差约0.0215%，只排除了这批状态下明显的整体幅度/饱和偏置。轮通道对总量的贡献较大；N3±单侧腿幅值和每电机协方差仍不同，这正是子空间变化的一部分，不能以整体RMS接近宣称完整探索分布匹配。该检查使用稳态保持目标，不能替代新策略初始化全轨迹、学习行为或泛化评估。未按结果调整θ、尺度或选择镜像符号。

下一项冻结v2配置、方法注册和源码快照，并对M1/N3±做共用训练入口短更新验收；旧训练入口仍按v1哈希核查，因此会因本次环境扩展拒绝旧准入记录。应建立v2证据入口，不能直接改写v1哈希使旧记录看似通过。长训练、64例门控集和最终测试集均未启动。

### 12.5 v2配置冻结与短更新验收

上述下一项现已完成：[独立协议](tools/results/yaw_precision_v2_2026-09-21/PROTOCOL.md)、[共同配置](tools/results/yaw_precision_v2_2026-09-21/training_config.json)、[源码/场景清单](tools/results/yaw_precision_v2_2026-09-21/protocol_manifest.json)与[准入记录](tools/results/yaw_precision_v2_2026-09-21/readiness.json)已落盘。v2场景与v1逐项相同；网络隐层、PPO超参数、课程、尺度及评估次数不变。M3是原M的三维模式，新加入M1/N3+/N3−；预实验总预算3600万策略步，仅为预算上限，未执行。

v2采用第12.3节扩展后的最强对照集合及配对方向门，v1其余成功/提前终止/非退化规则继承。原脚本版本归档于v2的history_v1目录，旧配置与结果未改写。`verify_sources`允许显式选择版本目录；训练入口须使用`--protocol v2`，默认v1仍会因历史源码不匹配拒绝当前环境。新入口同时要求工程准入、研究协议准入及依赖哈希有效，研究协议通过不等于方法优越性已被证实。

| 新增方法 | 策略步/更新次数 | 网络参数量（含Actor与Critic） | 采样＋更新耗时 |
|---|---:|---:|---:|
| M1 | 4000 / 2 | 12675 | 25.893 s |
| N3+ | 4000 / 2 | 12807 | 25.831 s |
| N3− | 4000 / 2 | 12807 | 26.468 s |

三者均通过权重更新、损失有限、保存重载、归一化mean/var/count冻结、课程reset及原开发1000～1003的确定性评估链路检查。这里不使用短更新成绩选超参数、镜像方向或正式检查点；smoke模型不能作为正式运行初始化。短更新训练处于stage1，stage2/3验证reset，完整课程的学习稳定性仍要在预实验中检查。

M3/B2-V/B2复用原各4000步的检查，依据是共同训练参数逐项一致、原代码归档哈希一致、旧动作映射/零动作物理轨迹回归通过，以及新版本共用入口已由上述三种方法实际走通；没有把旧模型重标为v2训练成果。旧环境之外的控制链未再修改，不额外重复无关回归。

最小检查还覆盖版本选择、源码变化拒绝、方法未注册拒绝、缺少准入时长训练拒绝、单独工程通过但研究协议未通过时拒绝，以及输出目录覆盖拒绝。完整门控比较脚本留到评估阶段用合成数据开发；不能用封存集测试统计代码。

本轮新增共12000策略步，均为smoke。下一阶段可按冻结v2配置启动预实验；每2万步保存模型、归一化与选择集指标，20万步检查属于同一200万步上限运行。实际长训练命令必须显式带`--train`，例如在项目根目录使用`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python wheelleg_ppo/tools/train_yaw.py --protocol v2 --method M3 --seed 1609 --train --output 新独立目录`。此命令本轮未执行；门控集与最终测试集仍未评估。

### 12.6 首个v2预实验已启动

用户继续授权后，于2026-09-21约01:38（北京时间）启动M3、seed 1609、CPU、8环境的正式预算预实验；从冻结的随机初始化开始，不加载smoke模型。配置见[本次运行配置](tools/results/pilot_v2_M3_seed1609_2026-09-21/run_config.json)，[日志](tools/results/pilot_v2_M3_seed1609_2026-09-21_run.log)持续写入。上限200万策略步，每2万步按固定规则评估32例选择集；20万步属于同一运行，不单独重启训练或追加预算。

当前只运行这一个方法/种子，不能称三种子比较完成或方法优势成立。64例门控集和最终测试集不在本入口中评估。进度应以本次目录中的step_*.json、selection.json和最终completed.json为准；模型文件先写入，只有对应指标JSON完整落盘才代表该次检查点评估完成。进程存在或日志有初始化输出不等于学习收敛。

北京时间01:43首个2万步检查点评估已完整落盘：32/32完成任务流程，16/32满足全部成功条件，平均Jψ=1.425673°，弱于同选择集B0的0.762952°与B1的0.613519°。所有失败均为completed后的判据失败，未丢弃失败轨迹。检查点模型和优化器张量均有限，进程仍在运行；见[首个检查点](tools/results/pilot_v2_M3_seed1609_2026-09-21/step_20000.json)及[带时间戳的启动状态](tools/results/pilot_v2_M3_seed1609_2026-09-21/launch_status.json)。这不是20万步阶段检查已通过，也不是正式比较结论；保持原预算/参数，后续按固定检查点记录趋势。

后续已完成6万步选择集评估：28/32成功，Jψ=0.972566°；成功数从2万步16/32、4万步20/32提高，但航向指标仍弱于B0/B1。以持续写入的selection.json为最新进度，本条不表示20万步阶段检查或200万步预算完成。

### 12.7 门控结果计算器的合成数据验收

训练期间补齐独立离线工具[score_yaw_gate.py](tools/score_yaw_gate.py)，复用原Jψ函数；未修改正在运行的训练程序、环境、配置或冻结清单。输入是已经采集的结果JSON，不运行仿真、不搜索或自动打开策略文件。真实封存集仍未评估。

工具核对8种方法（含两个固定经典对照）、3个RL种子、64个门控场景与28个旧回归场景的完整性和场景身份。固定控制器使用fixed键，不伪造3次训练重复；指标按场景和训练种子分层平均，M1/N3±纳入最强对照。任一门控或必要回归提前终止则不通过，不把其低误差纳入排名；完整但success=false的轨迹仍进入均值与失败分母。相对/绝对效应、配对方向、成功数及B0成功场景保护、速度/到达时间和旧回归姿态约束分别记录失败原因。CLI要求原始28例B0记录未经修改，拒绝覆盖输出文件。

[test_yaw_gate.py](tools/test_yaw_gate.py)只生成合成数据，覆盖正常通过、效应边界、仅相对或仅绝对改善不足、零参考误差、平均优势掩盖个别种子逆转、未完成轨迹、缺方法/种子/场景、重复/替换场景、非有限指标、伪造经典重复、成功标记矛盾及各项非退化失败。执行命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python wheelleg_ppo/tools/test_yaw_gate.py`。计算器完成的是数值与场景校核，不能证明日志来自指定检查点；正式门控前仍须锁定各方法配置、检查点及采集版本，再运行一次封存评估。不得将本次合成检查记为真实门控通过。

### 12.8 中断恢复与持续跟进

用户要求持续推进后，检查发现原进程14162及exec会话均已消失，原目录保留6万步完整评估及8万步模型/归一化配对文件，没有8万步评估JSON或最终completed.json；日志尾部未见Python异常。退出原因尚未确认，不解释为算法失败或成功完成。原记录、模型及失败线索全部保留。

新增[resume_yaw.py](tools/resume_yaw.py)在独立目录恢复PPO权重、优化器和VecNormalize，复用冻结的回调与评估逻辑；先补齐被中断的8万步评估，继承2/4/6万步选择记录，然后以reset_num_timesteps=False继续剩余192万步，累计上限仍为200万。已用M1 smoke的4000步检查点恢复至6000步验证新增2000步、参数更新、保存与选择记录，见[恢复验收](tools/results/yaw_precision_v2_2026-09-21/resume_smoke_M1/completed.json)。该2000步仅作恢复功能检查，不作为正式策略预训练。

**恢复边界必须披露：** 原检查点没有保存各仿真环境内部状态及完整随机数流，恢复会重置仿真并重新播种；这是一条有明确中断点的分段训练记录，不是原轨迹的逐值连续复现。不可把中断段当作独立训练种子、只挑表现较好段，或隐藏重复的部分评估开销。正式论文必须披露这次分段及其可比性限制；是否需要额外敏感性实验应独立决定，不自动追加预算。

北京时间约04:32通过独立用户服务`wheelleg-m3-1609-v2-resume1.service`启动恢复，输出位于[pilot_v2_M3_seed1609_resume1](tools/results/pilot_v2_M3_seed1609_resume1/run_config.json)，[日志](tools/results/pilot_v2_M3_seed1609_resume1.log)。`systemctl --user show wheelleg-m3-1609-v2-resume1.service -p ActiveState -p SubState -p MainPID -p Result`可检查宿主服务；不要只根据历史PID判断。服务由用户服务管理器持有，避免把长训练生命周期绑在短终端调用上。

已创建当前任务的15分钟heartbeat跟进，automation id为`ppo`。只在20万步阶段核验、单次完成、失败或需要用户动作时报告，普通进度静默。每个种子内顺序为M3→B2-V→M1→N3+→N3−→B2，种子顺序1609→1610→1611；已保存为[持续进度入口](tools/results/v2_progress.json)。每次只运行一个8环境任务，完成后启动下一项，保留所有结果和固定预算；全部18个预实验完成后汇总并暂停跟进，不自动开放64例门控/最终集或启动五种子正式扩展。运行时不得改冻结代码、超参数或主门。

恢复后8万步评估已补齐：26/32成功，Jψ=0.900551°，仍弱于B0/B1。恢复前后8万步权重、优化器状态、归一化mean/var/count逐值相同，累计步数保持80000，见[恢复核对](tools/results/pilot_v2_M3_seed1609_resume1/recovery_check.json)。这只验证保存状态恢复一致，不包含未保存的仿真/RNG状态。系统服务保持active/running，继续剩余预算；每15分钟的任务跟进依赖本机与Codex应用保持运行。

随后用户明确取消定时任务，已通过应用删除automation `ppo`，没有停止systemd训练服务。此前15分钟跟进安排作废，不能宣称仍会自动唤醒或自动调度下一方法；队列与预算继续保留，在当前任务中执行后续操作。

### 12.9 M3/1609的20万步阶段核验完成

北京时间2026-09-21约04:59完成[阶段记录](tools/results/pilot_v2_M3_seed1609_resume1/stage_200000_review.json)及[学习曲线](tools/results/pilot_v2_M3_seed1609_resume1/stage_200000_curve.png)，由[只读审查脚本](tools/review_yaw_stage.py)核对2～20万步全部10个检查点、每次固定32例场景和原评分函数。没有重新运行场景或读取封存结果。

工程检查通过：累计20万策略步对应8000次Adam小批次更新，模型及优化器张量有限，归一化mean/var有限且方差非负；观测统计计数200016.0001包含初次初始化和8万步恢复后的reset观测。10个检查点的32例均走完任务流程；completed不等于全部成功。原冻结依赖哈希仍匹配。

20万步最新策略为28/32成功、Jψ=1.007097°；按失败数→Jψ→步数的原选择规则，阶段最优为16万步，28/32成功、Jψ=0.803941°。相较同选择集B0的0.762952°、B1的0.613519°，阶段最优误差分别高约5.37%和31.04%，没有方法优势证据；18～20万步误差回升，也不能称已稳定收敛。阶段最优未丢失B0成功场景，速度/到达时间的选择集非退化检查通过。尚未对该策略执行原28例名义回归，因此不宣布完整研究门通过。

决定：工程链路可继续运行，保持现有配置和累计200万步上限，收集后续学习曲线；不按这次结果改奖励、调参或换主门。停止/继续的研究优势判据仍按冻结v2在完整比较后执行。systemd服务保持运行，定时跟进已删除；其余方法须待当前运行完成后在本任务中按队列启动。门控64例和最终集仍封存。

### 12.10 CPU/CUDA实测与设备决定

用户要求必要时切换GPU后，使用已完成的24万步检查点，在相同8环境、stage3、PPO参数、起始权重/优化器/归一化下做三组CPU/CUDA对比。[预定测试规则](tools/results/device_bench_2026-09-21/design.json)在运行前落盘：顺序CPU→CUDA、CUDA→CPU、CPU→CUDA，种子7100/7101/7102；每次2000策略步、一次完整rollout和10 epochs更新。GPU必须完整链路中位加速至少10%且三组均更快才切换，不看策略任务成绩决定设备。

执行脚本为[benchmark_yaw_device.py](tools/benchmark_yaw_device.py)，[结果](tools/results/device_bench_2026-09-21/summary.json)记录六次实际设备、采样与更新耗时、有限性及源文件哈希。模型参数确实在CUDA设备计算，未将显存上下文占用当作GPU训练证据。初始化缓存和5次预测预热在计时外；CPU/CUDA随机数流不同，因此轨迹可能不同，不声称逐步完全相同。这里比较同分布完整训练链的短时性能，不给长期耗时保证。

| 项目 | CPU中位数 | CUDA中位数 |
|---|---:|---:|
| 2000策略步采样＋更新 | 12.8635 s | 13.4557 s |
| PPO更新部分 | 0.1068 s | 0.1785 s |
| 采样及其他开销 | 12.7576 s | 13.1815 s |

三组CPU耗时/CUDA耗时比分别0.9490、0.9848、0.9280，GPU没有一组更快；按总中位数GPU耗时约多4.6%。当前更新部分占比很小，主要耗时位于采样及其他开销，因此本次保留device=cpu，不迁移当前策略或物理后端。这也符合[SB3对非CNN PPO的CPU使用建议](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)，但设备决定以本机实测为依据。

为避免两个8环境任务争抢CPU，基准服务运行前通过systemd freeze冻结原训练服务，并用ExecStopPost确保完成或失败后thaw；基准服务另设单次600 s运行上限。测试结束已确认原服务FreezerState=running、active/running且PID仍为4952，训练继续，没有新增模型重载/仿真重置分段。冻结期间的墙钟时间应作为硬件诊断停顿披露，不据包含该停顿的检查点间隔估计稳定吞吐。未恢复任何定时任务。

共12000策略步只用于硬件诊断，测试模型不保存为候选、不接回正式训练。未改冻结超参数、数据、主门或当前训练源码，门控与最终集未使用。

### 12.11 独立 MuJoCo Warp 基线与原 CPU 对照（2026-09-21）

用户明确要求新增GPU物理基线，原CPU仿真作为原始效果参照。新增入口为 `../wheelleg_warp/baseline.py`，结果与说明在 `../wheelleg_warp/`。原CPU代码、v2协议与在跑M3/1609训练不因本次切换设备；新入口也不评估64例门控或最终保留集。

先验证同模型、同原控制器的站立/非对称地面/跳跃配对轨迹，再测试256环境开环物理回放。GPU物理回放与8线程原生CPU rollout比较，倍率不代表Python闭环或PPO端到端训练倍率。详细验收与未关闭项只在根目录 `../PROJECT_MEMORY.md` 维护，原始机器可读证据在 `../wheelleg_warp/results/`，不得以“运行在GPU”直接替代原工程及研究准入。

### 12.12 CPU / Warp 双后端正式训练对照（2026-09-21，用户明确授权）

用户最新要求同时进行CPU和GPU两个基线的正式训练并最终比较效果。本次是独立后端对照，M3/diff3、种子1609/1610/1611，每后端每种子200万策略步，总新增1200万步；双方各8环境队列并行，内部依次运行种子。不扩展为本节六方法全部后端组合；已有CPU分段预实验保留但不计入这组配对。

具体协议和源码/依赖哈希冻结于 `../wheelleg_warp/results/formal_cpu_warp_v1_20260921/protocol.json`，工程入口及边界见 `../wheelleg_warp/FORMAL_TRAINING.md`。两侧用相同v2 PPO配置和CPU网络，仅物理后端分别为CPU MuJoCo及GPU Warp；CPU控制/奖励仍复用，原CPU源码不改。GPU既有非对称接触全状态校对失败保留，不能把新训练授权表述成后端等价已验证。

选模仍用共同CPU32例选择集；终评使用训练前新固定的64例IID场景（91000～91063），两侧selected与last检查点全部报告。原研究门控/最终集保持封存，本比较不证明M3相对M1/N3±等其他方法的结构贡献。两侧4000步短更新与初始权重一致性验证通过后，系统服务启动双队列；全预算完成后自动共同终评生成 `COMPARISON.md` 和 `comparison.json`，中途进度不能冒称最终效果。

### 12.13 真实训练状态可视化版本（2026-09-21）

用户选择采集训练进程原始逐帧画面，允许改造接口并重新安排训练。旧无采集双后端对照已停止，状态标为superseded且全部保留；当前 `../wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/` 从相同种子重新开始，预算、控制、奖励、选模和最终场景规则不变。只有各侧环境0增加只读状态采集，独立渲染器读取实际状态，不另跑一个策略环境冒充训练画面。

采集版CPU/GPU各4000步验证通过，CPU最终策略权重与无采集版逐值一致。新正式双队列于07:39:50启动；前端 `http://127.0.0.1:8765/` 显示实时采样、学习指标及动态ETA，完整回合轨迹和抽样GIF本地保存。所有运行、来源、复现范围与未关闭项以根目录项目记忆及 `../wheelleg_warp/dashboard/README.md` 为准。可视化不消除GPU物理差异，不使阶段进度变成最终方法结论。

### 12.14 GPU打包回读及检查点分段续训（2026-09-21）

用户要求优化GPU和动画。保留原物理方程、0.5ms步长、CPU控制/奖励，新增打包回读，移除不必要主机字段传输；逐值字段核对和12000步越障接触检查通过，4000步GPU短更新观测2.88倍采样/更新提速，仍不能宣称GPU整体优于CPU。录像50FPS与真实实时流分开，编码不再阻塞直播。

当前运行转到 `../wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/`，CPU60000、GPU20000检查点恢复模型/优化器/归一化，原预算不扩张；环境/RNG重置且恢复点不同，明确属于分段实验，不能冒称精确连续轨迹或只凭当前段时间得出全程硬件倍率。优化和旧实验全部保留；GPU跨运行权重不逐位一致的事实也已记录，详见 `../wheelleg_warp/OPTIMIZATION.md`。

### 12.15 非结构化地形扩展 terrain-v1（2026-09-22，用户明确授权）

用户要求把不平坦路面、斜坡、台阶等场景写入计划并开始训练。该实验是已完成1024原生GPU基线之后的独立扩展，原最佳开发32/32、独立终评60/64及全部失败场景保持冻结对照，不覆盖、不重新解释。

terrain-v1复用原M3策略、VMC＋六状态LQR、动作/奖励/限幅和1024世界批量执行，仅扩展统一拓扑的地形模型。训练范围为1°～3°坡道与横坡、2～6mm块状粗糙路、5～20mm全宽台阶、粗糙路叠加原左右障碍，并固定30%原场景回放。3°～5°坡面、6～10mm粗糙度和20～30mm台阶封存为停止后OOD终评；不在现有世界坐标5°姿态门下训练更陡坡，避免评价定义与任务冲突。

固定32例地形开发集与16例新IID原场景共同选模，先比较总失败数，再保护原场景，最后比较Jψ；固定64例OOD地形只在停止后打开。每轮102.4万步，中点/末点评估；第1～3轮逐步加入坡面、粗糙路、台阶和组合，之后保持完整分布；连续3轮无改善或最多8轮停止。详细范围、启动门、动画隔离和结论边界见 `../wheelleg_warp/TERRAIN_PLAN.md`。

terrain-v1完成后发现组合OOD的粗糙块可能覆盖较矮左右障碍，导致接触型成功门不可达；该0/12不作为策略失败结论。坡道、横坡、粗糙路各13/13和台阶7/13保留为有效v1证据。修订后的terrain-v2清空中央障碍走廊，使用全新固定种子并从v1最佳继续最多4轮Stage3；不得覆盖v1失败协议或重复使用已打开的v1 OOD作为v2终评。

terrain-v2按规则完成4轮，第2轮末点为开发最佳，地形20/32、原回归11/16；连续2轮无改善后停止。停止后新64例OOD全部完成，55/64成功、Jψ=0.586542°：坡道13/13、横坡12/13、粗糙路12/13、全宽台阶13/13、组合5/12。新16例原回归为11/16。终评未用于重选或追加训练；组合剩余6例未满足冻结障碍接触门，另1例姿态超门。该结果只证明terrain-v2冻结协议下的纯仿真表现。
