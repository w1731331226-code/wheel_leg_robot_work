# 长期项目记忆

维护规则：每轮大的对话结束前更新当前状态、决策理由、变更及验收证据、未关闭项；每次提交必须包含本文件的更新。此文件是跨对话入口，原始实验数据与历史计划继续保留在各自目录。自动快照条目只证明归档，不证明代码或研究结论通过。

## 当前约定（2026-09-21）

- 用户要求：新增 MuJoCo Warp GPU 基线；原 CPU 仿真继续作为原始效果校对；运行 GitHub 同步守护；为每轮实质工作更新长期记忆并及时写中文提交说明。
- 项目约束入口：`AGENTS.md`、`.agents/skills/wheelleg-project-guard/SKILL.md`。GPU 基线独立放在 `wheelleg_warp/`；CPU 控制、物理模型、环境和历史证据不覆盖。
- Git 远程：`origin` → `https://github.com/w1731331226-code/wheel_leg_robot_work.git`，工作分支 `main`。禁止 force push；远程分叉时保留本地提交并人工处理。
- 同步覆盖 Git 已跟踪文件及未忽略的新文件，连续两次30秒扫描保持稳定才自动快照；缓存、凭据及 `.gitignore` 排除内容不上传。超过95MiB的单文件拒绝提交。后台训练产生的稳定检查点属于用户授权同步范围，记录为独立自动归档提交。
- 编辑任务开始暂停 `wheelleg-git-sync.service`，结束更新此记忆、提交后恢复；可用 `.git/project-write.lock` 排他锁代替。用户暂存内容优先，不自动提交已有暂存区。
- `.githooks/commit-msg` 在本地强制检查中文标题、随提交更新本记忆、凭据路径和文件大小。新克隆须执行 `git config core.hooksPath .githooks`；本地钩子不是 GitHub 服务端规则，不允许代理主动绕过。

## 当前任务：terrain-v1非结构化地形课程已启动（2026-09-22）

- 用户要求把不平坦路、斜坡、台阶等写入计划并开始训练，随后明确保留原有训练动画。详细协议已写入 `wheelleg_warp/TERRAIN_PLAN.md` 和 `wheelleg_ppo/PAPER_PLAN.md` 第12.15节；已完成1024原生基线及60/64终评保持冻结对照。
- terrain-v1复用原最佳策略/Adam/VecNormalize、VMC＋六状态LQR、M3动作、2kHz物理、50Hz策略和成功门。训练地形为1°～3°坡道/横坡、2～6mm块状粗糙路、5～20mm全宽台阶及粗糙路+左右障碍；每轮保留约30%原场景。3°～5°坡、6～10mm粗糙度、20～30mm台阶只用于停止后OOD终评。
- GPU统一拓扑新增16个盒几何：上坡/平台/下坡、横向倾斜面、12段粗糙块或全宽台阶。它们是可控近似，不冒充真实连续随机地貌。现有世界坐标5°姿态门未改，更陡坡必须先实现地形相对指标。
- 开发集已分层冻结为32例：原场景8、坡道/横坡/粗糙路/台阶各5、组合4；另有16例新IID原场景防遗忘。64例OOD五类近似等量，训练/选模期间不读取。排序为48例总失败、原场景失败、Jpsi和。
- 工程门通过：旧NativeEnv完整回归通过；六类地形统一拓扑、有限状态、真实回合和非法动作拒绝通过；1024训练采样中原场景占30.664%。原最佳在最终固定起点评估为地形24/32、原回归13/16。
- 首次探针因测试脚本把新增步数误传累计步数而触发预算断言，失败未保存/晋升权重，保留于 `terrain_smoke_20260922`。修正探针严格新增102400步、2次PPO更新通过，8例地形4/8、4例原场景3/4，结果在 `terrain_smoke_fix_20260922`；探针权重不进入正式训练。
- 唯一正式地形目录为 `wheelleg_warp/results/terrain_v1_1024_20260922/`，服务 `wheelleg-terrain-formal-1024.service` 已实际启动并完成前三轮。Stage1/2/3的选定成功数均为地形24/32、原回归13/16；Jpsi和持续下降（1.616833→1.475449→1.431033），因此未触发停滞，第3轮为当前最佳。第4轮已开始，后续保持完整Stage3。每轮1024000步，中点/末点选模，连续3轮无改善或最多8轮停止。
- 原动画与新动画物理隔离：原基线继续在 `http://127.0.0.1:8765/`，已有20个正式录像保留；地形训练在 `http://127.0.0.1:8766/`，独立数据根 `dashboard/local_data/terrain_v1/`。两页互相链接。地形页已看到1024总览、坡道环境111真实详情和独立50FPS录像，未覆盖原文件。
- 用户随后要求正式训练时保持动画窗口打开；8766地形动画页已设为可见并保留到后续任务。修正了双页面共用环境选择入口以及中点评估误显示为整轮完成的问题；第4轮已在可见窗口展示粗糙路环境111和独立录像。
- 服务资源：地形训练24GiB且禁用换页，地形渲染3GiB，地形前端512MiB；不并行运行第二组1024环境。Git同步在本轮编辑/冻结期间暂停，完成提交后恢复。

### terrain-v1/v2最终收尾（2026-09-22）

#### terrain-v3修正后扩展与正式训练

- 已在修正后的统一16 geom GPU拓扑加入`rolling_slope`连续四段坡、`multi_step`多级台阶和`split_level`左右异高路面。新场景显式`relative_attitude=true`：局部横滚/俯仰误差与世界偏航门均5°，世界横滚/俯仰另设10°硬边界；奖励也使用局部姿态误差。旧v1/v2场景缺省保持世界5°门，历史语义不变。
- terrain-v3开发集冻结48例（mixed 12、legacy 8、其余七类各4），另有16例原回归；最终留出64例（mixed 15、其余七类各7）在训练停止前仅写入协议、从未评估。terrain-v2正式检查点起点为开发46/48、原回归15/16，64/64完整且有通过证据；新增三类均4/4，开发两个失败均为legacy偏航。
- 工程门通过：原六类语义与terrain-v3八类非legacy闭环、双轮地形接触、相对姿态字段；旧NativeEnv完整回归；1024世界Stage1/2/3分布、有限状态、逐世界静态几何和一步物理。1024初始化41.57秒。102,400步GPU探针完成2次PPO更新，开发48/48、原回归15/16、权重有限且保存重载正常；探针权重未晋升。
- terrain-v3正式训练从terrain-v2第2轮末点而非探针恢复，目录`wheelleg_warp/results/terrain_v3_1024_20260922/`；1024环境、每轮1,024,000步、连续3轮无改善停止、最多10轮。服务`wheelleg-terrain-v3-formal.service`为24GiB/禁用换页，独立仪表盘/渲染服务在8768和`dashboard/local_data/terrain_v3/`。已确认1024总览、环境111混合地形真实3D详情、约47.7 FPS/400 Hz及第1轮512,000步中点评估；动画页保持可见。第1轮选定开发47/48、原回归16/16，较起点总失败3降至1，停滞0并进入第2轮。最终64例只在停止后打开。
- 正式训练共6轮、新增6,144,000步；第3轮中点最佳为开发47/48、原回归16/16，Jψ 0.575799°/0.551306°。第4～6轮连续3轮未改善后停止，末次未覆盖最佳。停止后首次打开64例留出：47/64成功、61/64完整且有双轮地形通过证据；mixed 15/15，rough/ramp/multi-step各7/7，rolling-slope 5/7，split-level 3/7，step 2/7，cross-slope 1/7；原回归16/16、Jψ 0.545209°。3例高台阶跌倒/超时导致总体Jψ按协议为未完整；其余主要失败是横坡/异高世界偏航和连续坡局部俯仰。终评未用于重选或续训。渲染服务已停，8768页面保留最终快照；报告`wheelleg_warp/results/terrain_v3_1024_20260922/REPORT.md`。
- 用户要求继续扩展单腿上坡、复杂凹凸地面和左右不对称接触；斜坡自旋明确为上坡后水平平台机动，可不作为训练场景。terrain-v4计划已分阶段加入单侧坡道、左右交错粗糙、单侧台阶/坑槽、摩擦差、短时单轮离地、接缝/坡顶和坡上启停；平台自旋等待期望角速度/目标航向接口后只作评估。
- `single_side_ramp`和`asymmetric_rough`统一16 geom原型、指定接触门、双轮越过末端证据及真实闭环通过。新终态字段分开记录接触、轮进度、末端通过、残差受限比例和基础控制不可行步数，尚未把新增证据追溯改写旧成功门。
- 全新24横坡＋24左右异高GPU配对：零残差21/48、terrain-v2 20/48、terrain-v3 21/48；terrain-v3相对零残差仅一胜一负，横坡为8/24对7/24，说明同一PPO配方继续堆步数缺乏依据，下一步先查基础偏航控制、轮滑、差动扭矩与电机速度包络。事后复测旧64例为64/64实际接触、62/64完成并双轮越过末端；冻结47/64不改写，复测46/64有1个5°偏航边界翻转。证据在`wheelleg_warp/results/terrain_v4_diagnostics_20260922/`，terrain-v4正式训练未启动。
- 控制器级轨迹显示terrain-v3失败组已发出更大偏航动作/残差差动扭矩，但横坡失败相对成功的左右轮速差4.52/2.82rad/s、估算滑移0.099/0.048m/s、偏航角速度0.125/0.069rad/s，基础控制不可行比例2.81%/0.39%，λ 0.971/0.995；问题优先指向非对称接触下的基础偏航控制与速度包络，而非策略未请求纠偏。
- 新鲜高台阶高度×速度图：15～18mm三档速度均4/4；21mm在0.5m/s为0/4、0.7/0.9m/s均4/4；24mm为0/3/2，27mm为0/1/3，30mm三档均0/4并有跌倒/非轮接触。`readiness.json`判定terrain-v4未成熟，未启动训练；下一步扫描基础偏航P/D与残差偏航权限，并在新开发集、原回归和15～21mm能力门上复核。
- 原生偏航参数现已显式化为`(Kp,Kd,质量比例上限,残差尺度)`，默认值保持旧行为且CPU/GPU逐步峰值差4.7e-7。12组开发扫描的最佳候选`(0.8,3.0,0.24,0.30)`仅把20/48提高到21/48；独立验证默认/候选横坡＋异高均17/48，候选完整率47→46、原回归均16/16、15～21mm台阶32/36→29/36。候选已拒绝，terrain-v4训练未启动；下一步改为滑移/轮速差感知偏航控制与速度相关权限，不再继续放大固定P/D。
- 根因复核更正上述归因：横坡/左右异高以抬起的盒体构造，入口高侧形成约16～27mm单侧直立台阶；横坡320mm宽、轮距300mm，轮心边距仅10mm。48例四条件消融为原构造19/48、仅加宽18/48、仅400mm进出坡过渡44/48、过渡＋每侧350mm余量48/48；原29个失败首次偏航超门均在入口、早于或未发生掉边。独立520000起始48例17/48→48/48，原场景16/16保持；平均基础不可行步数212→39。此前“滑移权限不足”为相关性推断，不应当作根因；差动力矩优先试验增加未完成并已从源码撤下。
- 已修复v4坡面构造：显式`transition_run_m`/`lateral_margin_m`，旧场景默认0仍可复现；新v4横坡/异高带过渡和横向余量，停车目标越过完整出口坡。中央轮道高度/坡度不变，解析与GPU正反/左右8例、旧地形/进阶回归通过。真实直立台阶和原突变入口作为压力任务保留；不将地形干预改善称作策略增益。几何未改的一例台阶在混合批次中17.10°→4.07°，需报告求解/批次敏感性，不能一概说微小阈值波动。证据在`wheelleg_warp/results/terrain_root_cause_20260922/`；本轮未启动正式训练，先冻结分离坡面/突变越障的新协议。
- 修复后独立坡面组重复两次均48/48，最大偏航2.334°/2.327°；另一次接触见证确认48/48双轮均实际接触中央目标geom并越过出口。真实台阶单独重复两次均34/36，故全速度越障准入仍未通过，训练未启动。不可把修复后的平滑坡面成绩代替原突变入口能力。
- 最终1024世界11类场景初始化＋3个策略步采样通过，逐世界静态geom坐标一致、状态有限、无容量溢出，初始化及检查共41.52秒。几何解析检查覆盖8个正反/左右组合。原控制器及拒绝的差动优先方案均未晋升或改默认；恢复同步前以中文提交归档本轮修复与全部失败对照。
- 最小策略优化已完成：固定新96开发场景做通道消融、25%/50%残差强度和512000步定向PPO对照。定向训练1024并行、真实台阶约60%/原场景30%，256k台阶38/48但完整44/48，512k退至34/48且原回归15/16，均未晋升；8769页面实际展示总览/环境111物理帧，试验结束后渲染停止。
- 已拒绝的残差缩放试验`residual_scale=.5`仅在策略动作校验后整体减半，不改模型/地形/控制增益/奖励；默认仍1.0，录制与终态带尺度字段。两次开发重复原策略35、34/48，半残差39、38/48且完整45→46，其他3组均16/16。打开新测试前冻结半残差；三组独立测试合计288例，台阶96→98/144、完整均135/144、原场景均45/48、坡面/进阶均48/48，后两组平均偏航峰值约降低35%。台阶9胜7负（McNemar p=.804）且平均偏航峰值12.796→14.189°，不足以宣称普遍越障提升；不替换全任务默认、不启动长训练。失败记录`wheelleg_warp/results/policy_minimal_20260922/REPORT.md`及`rejected_configuration.json`。后续优先验证残差约束及对称台阶所需共模动作可表达性，不能继续盲目调固定增益/堆训练量。
- 用户进一步明确必须“大幅提高成功率且稳定”。撤销上条将50%残差称为优化成果的表述，全部缩放/消融/512k定向训练候选判为未通过，不推荐部署。强准入暂定困难组至少提升20个百分点且成功≥85%、完成≥95%、重复/独立批次稳定、原回归不退化。仅改变姿态均值或净增2例不算优化。下一受控方向是复用CPU已有virtual6，补足M3无法独立控制的共模支撑/腿力矩/轮驱动；先验证与M3嵌入的等价性，再用全新开发场景检验。
- 已完成virtual6受控试验：默认仍diff3；六维映射复用CPU既有虚拟力定义，原策略输出按[F,-F,M,-M,T,-T]迁移，GPU嵌入/共同轮扭矩/状态重置检查通过；动作空间改变使Adam重置，非精确续训。新开发原M3困难台阶33/48，强门要求≥43/48成功、≥46/48完整；1024环境2,048,000步/40更新的中点32/48且完整41、末点32/48且完整44，其余三组均16/16，未通过。受阻速度误差的同向腿力矩辅助增益1/3为31/48、33/48，也未通过；留出720000尚未评估，权重不晋升。报告`wheelleg_warp/results/virtual6_capability_20260922/REPORT.md`。原半残差配置已改名`rejected_configuration.json`并标拒绝。当前没有满足用户强门的优化成果，不能再把小幅改善、更多自由度或更多训练步骤称为成功优化。

- 后续地形通过判据审计推翻了该小节原有效性：MuJoCo Warp从第一个CPU模型一次性初始化world-body静态`Data.geom_xpos/geom_xmat`，而旧`native.models.batch`只写逐世界`Model`字段，导致同一批次所有世界实际复用第一个世界的障碍/地形位置。该共享缺陷也影响此前60/64原生GPU基线的逐世界静态障碍变化；旧CPU基准不受影响。历史数值和文件保留，但60/64、terrain-v1/v2训练结论及55/64均不得再作有效场景泛化证据。
- 已在共享批量入口逐世界补写静态geom位置与姿态；六类闭环及1024世界初始化/一步物理均验证`Data.geom_xpos == Model.geom_pos`，1024初始化实测42.00秒。非legacy成功门新增左右轮地形接触掩码，输出required/touched terrain mask与`terrain_passed`；混合地形仍叠加原左右障碍接触门。平地NativeEnv完整回归保持通过。
- 原混合地形6个“未接触”均为正向场景；历史缺陷复现的轮到障碍最近纵向距离1.842～1.879m且零接触，属于错误静态几何，不是绕过、跨越或接触漏记。修正后6例均满足所需障碍接触，最近距离0.001～0.005m。轨迹证据`mixed_trace_audit.json`，代表动画为本机忽略文件`dashboard/local_data/terrain_audit/seed_250005_before_after.gif`。
- 旧检查点在同一已公开64例上的缺陷修正诊断重放为51/64：坡道13/13、横坡6/13、粗糙路13/13、台阶9/13、混合10/12；64/64均完成并满足地形接触证据，13例失败全部为世界坐标5°姿态门。该结果不是新留出验收，不用于选模或续训。下一步先做修正后同场景CPU/原生GPU配对回归，再冻结全新开发/留出集；此前不启动训练。
- 已按用户建议在同一修正后MuJoCo Warp GPU后端完成策略配对：terrain-v2既有32个开发地形上原策略30/32、当前策略31/32，Jψ 0.895332°→0.604241°；16个原场景回归均15/16，Jψ 1.008431°→0.636857°。逐场景为开发地形共同成功30、当前单独成功1、共同失败1，原场景共同成功15、共同失败1；没有原策略单独成功场景，48/48均完成并具有通过证据。因场景参与过terrain-v2选模，本结果仅说明同场景未见能力损害，不作泛化结论；证据在`wheelleg_warp/results/terrain_paired_gpu_20260922/`。下一步冻结修正后全新开发/最终留出场景，仍不直接启动训练。

- terrain-v1完成8轮、新增8192000步，第8轮末点最佳：开发地形24/32、原回归13/16，Jpsi 0.542737/0.611607。v1 OOD为46/64且存在无效组合子协议：粗糙块覆盖中央障碍区，组合0/12不作算法结论；坡道/横坡/粗糙路各13/13、台阶7/13为有效v1证据。
- 修正中央±0.35m障碍走廊后，同一12个v1组合OOD直接复测从0/12升至6/12，并新增required/touched接触掩码证据。terrain-v1结果完整保留，不原地改协议。
- terrain-v2从v1最佳恢复，使用全新开发/原回归/OOD种子，最多4轮Stage3、连续2轮停滞停止。已完成4096000新增步；第2轮末点为开发最佳，地形20/32、原回归11/16，Jpsi 0.530597/0.549816，第3/4轮无改善后正常停止。
- terrain-v2停止后首次打开新64例OOD，全部轨迹完成，55/64、Jpsi=0.586542°；坡道13/13、横坡12/13、粗糙路12/13、20～30mm全宽台阶13/13、组合5/12。新16例原回归11/16、Jpsi=0.546294°。终评未用于选模或续训。
- 组合7个失败中6个未满足冻结障碍接触门、1个姿态峰值超5°；横坡/粗糙路各1例因停车尾速超门。结果不扩大为全局最优、方法优势或实机安全。
- 动画完整保留：原基线8765、terrain-v1归档8766、terrain-v2最终页面8767，数据根互不覆盖。terrain-v2训练完成后渲染服务停止释放资源，三个前端继续提供最终快照和录像。
- 最终报告 `wheelleg_warp/results/terrain_v2_1024_20260922/REPORT.md`，曲线 `convergence.png`，机器可读结论 `final_verification.json`、`final_evaluation.json`。本轮提交后恢复Git同步并核对远程。

## 当前正式状态：唯一1024原生基线已上线（2026-09-21）

- 用户授权条件已落实：仅保留一个活动原生GPU训练基线；全部1024世界总览及任意编号3D详情上线。用户停止规则为连续3轮无实质改善或最多10轮。
- 从头训练16/50/250周期及中间检查点复核均未过稳定门，失败完整保留。最终方案迁移旧CPU32/32优胜策略，1024环境、n_steps=50、学习率3e-5、target_kl=0.005；3独立训练场景库/探索随机流各新增512000步，末次31/32、32/32、32/32，合计95/96，实际末次门通过，未拿起点成绩冒充通过。
- 新正式bootstrap采用1611流真实GPU续训末次模型：开发集32/32、Jpsi=1.0531013243°，优于旧优胜起点32/32、1.2186946633°。继承1572000步，之后新增预算单独记录；不是三次从头初始化的优势证明。
- 正式恢复预检 `native_formal_probe_20260921/verified.json` 通过：2次真实PPO更新、102400新增步，中点/末点评估与停止。仅用4个开发场景，不触碰正式64例终评集；预检权重未进入正式训练。
- 已删除旧CPU两个续训源目录、三个历史双后端运行版本及废弃双后端入口，共340个跟踪文件，含176个旧权重/归一化文件。旧本地画面/轨迹进入回收站。完整清单 `formal_native_1024_20260921/retirement.json`；公共CPU物理/控制/评估及27项审计源码、Git历史保留，新bootstrap独立，不依赖删去的旧目录。
- 正式训练已经按规则完成：共6轮、累计新增12288000步；第3轮中点为最佳开发检查点，32/32、Jpsi=0.9399237316°。第4～6轮没有实质改善，停滞计数达到3后自动停止；末次退化没有覆盖最佳，实际服务正常退出。
- 停止后首次打开固定64例新IID终评集，结果60/64、Jpsi=0.9371690106°；4个失败轨迹均完整结束，偏航峰值5.30°～5.83°超过冻结5°门。终评未用于重选、调参或续训。该结果是纯仿真泛化证据，不是全局最优、论文优势或实机安全证明。
- 唯一正式目录 `wheelleg_warp/results/formal_native_1024_20260921/`，入口 `train_native.py`。页面 http://127.0.0.1:8765/ 继续展示最终状态、1024总览快照、任意世界最后详情和录像；训练完成后渲染服务停止并禁用以释放约1GiB内存，仪表盘保持运行。
- 1024真实刚体总览、世界3状态精确核对与实际世界851详情均通过。详情400Hz只读物理帧，渲染实测约49.4FPS；页面显示实际浏览器FPS及源状态年龄，评估时如实暂停。原始轨迹和50FPS录像只归档世界0，所有世界都可看实时总览/详情。录像用真实帧，不额外仿真、不插值造帧。
- 报告与曲线：`formal_native_1024_20260921/REPORT.md`、`convergence.png`。完成状态、最佳策略和终评分别见status/selection/final_evaluation；末次退化、2048中断、原CPU/GPU全状态等价未全过等限制保留。
- 后续不要直接修改冻结的训练/控制/采集源文件或bootstrap，正式协议会校验哈希；界面与说明可独立维护。发生失败保留最佳和错误，不自动覆盖已有轮次目录或伪装精确续训。

## 当前任务：1024单基线、全部环境动画与有界循环训练（2026-09-21）

- 最新用户要求：继续优化1024策略；验证没有问题后，删除旧CPU基线及CPU/Warp两对照基线、相关策略和权重；前端只显示新的并行基线，展示算法真实训练环境及过程；启动正式多轮优化。随后用户明确要求动画展示所有环境，不仅环境0。
- 用户已选择停止规则：连续3轮没有改善则停止，最多10轮，保留最佳权重。候选实现 `train_native.py` 将开发集成功数置于Jpsi之前；0.5%相对Jpsi改善或成功数增加计为实质改善，任何严格更优检查点仍保留；终评只在停止后一次，不用于重选。
- 初始三种子对照仍在跑。已完成16步周期seed1609为31/32、Jpsi0.85969；seed1610只有28/32、Jpsi0.95923。250步seed1609为27/32、0.61852；seed1610为30/32、0.64434。不能据此说1024策略已稳定通过，旧基线尚未删除，正式循环训练尚未启动。
- 已准备50步周期补测服务 `wheelleg-native-tune-1024-v2.service`，等原对照全部结束后先运行新采集/渲染预检，再串行补测3种子同预算；不同时运行第二组1024世界。结果目录 `wheelleg_warp/results/convergence_1024_rollout50_20260921/`。前一次只等待的空任务记录移至 `_waiting_v1`，未中断GPU训练。
- 正式切换准入拟定并实现为：3种子均完整、每种子至少30/32、平均至少31/32（旧CPU已有31/32参照）；没有配置过门则不清理旧基线、不自动开始正式训练。代码可用不代表此效果门已通过。
- 新 `native/live.py` 保持原实验环境不变，附加只读GPU快照：1024世界真实刚体位置/方向与轮子转动总览；世界0完整400Hz轨迹；任意选择世界的400Hz详情。总览为真实姿态投影，点击/编号选择后显示该世界真实3D帧。详情目标50FPS，用原子图像+元数据包防止帧和标签错配，不另跑策略、不插值造帧。
- 前端已切换为单基线准备页面，地址仍为127.0.0.1:8765；旧渲染服务仍停止，当前无伪造动画。HTTP边界与0/1023选择、无效输入拒绝测试通过；浏览器无脚本错误。真实GPU采集、全环境包、选择世界状态一致性、PPO录制与渲染开销尚待预检。
- 新准入/监督/前端代码正在本轮验收，Git同步在编辑期间暂停。此前训练/吞吐失败、2048中断和源文件历史均保留；只在用户条件满足后执行精确旧目录清理，公共物理/控制/评估依赖保留。

### 本轮验证进展与关键修正

- 原16步周期末次三种子成功数31/28/0，250步为27/30/28；1611短周期失败是偏航峰值超过既定5°，轨迹仍完整，不能称稳定收敛。补测50步周期仍有末期退化，继续保留全部原末次证据。
- 在不放宽成功条件的前提下增加独立选模复核：先按原8例开发曲线成功数/Jpsi/较早步数选择中间候选，再在完整32例开发集复核；准入门仍为各至少30/32、总至少93/96。正式循环中点/末点评估，差轮次不覆盖最佳，下一轮从最佳恢复并重新抽样，累计消耗预算不因回滚抹去。
- 活跃补测服务已为 `wheelleg-native-tune-1024-v3.service`。v1/v2仅等待时停止，未杀在跑GPU；v3先完成采集预检，再补测50步三种子，最后复核选定检查点。当前不要修改 `tune_native.py` 或原冻结实验源码，以免源码哈希失效。
- `native_live_preflight_20260921` 预检通过：8环境只读采集与原生类比较，观测最大差0.0057545、奖励差8.99e-8，8个回合完成；环境3状态逐值对照通过，所有世界刚体位置与GPU状态一致。
- 同一预检真实1024 PPO完成524288步，含录制/渲染为11309策略步/秒，权重实际更新；400Hz世界0轨迹及50FPS WebP/GIF生成成功。浏览器已看到1024/1024实际世界和真实回合录像。
- 浏览器发现旧预检总览包缺少轮径字段，已兼容其实际0.05m轮径；新包从模型读取轮径。观测器metadata方法与VecEnv字段冲突在GPU预检前改名。渲染循环多等一帧间隔导致约40FPS，已改为按帧起始时刻节拍，实际新帧率待正式入口预检复测。界面显示实际浏览器FPS，采样暂停不冒充50FPS。
- 正式监督新增实际恢复短探针 `test_native_formal.py`，将以独立目录验证两个PPO更新、中点/末点及停止流程，预检权重不进入正式训练。目前效果准入未完成，旧模型尚未清理，正式训练尚未启动。

## 当前任务：暂停训练与多环境并行实测（2026-09-21）

- 用户要求先停止旧基线及当前CPU/GPU训练，再全力测试多环境并行并给出最优配置。两训练服务及其全部子进程已停止；渲染服务为独占测量暂时停止，训练不自动恢复。
- 用户补充：必须验证收敛效果，随后明确最多使用1024批量，更多会卡死。2048测试无完整结果，2048测试启动后至10:02之间发生重启；4096未运行，测试队列已不存在。不得重启大于1024的实验，测试进程需限制内存。
- 新实测目录 `wheelleg_warp/results/parallel_sweep_20260921/`；停止记录含原正式status，历史日志/检查点保留。
- 正在比较CPU多进程、现有Warp多进程及工作区已有原生GPU批量候选。保持250步轨迹、原PPO超参，区分采样与实际梯度更新；原生候选未获正式等价准入。

### 1024候选工程检查与收敛协议

- CPU扫描1/4/8/12/16/20/24/32，峰值20环境229.7策略步/秒；原Warp 1/4/8/12峰值8环境82.6。原生32/64/128/256/512/1024中1024达到11657.0策略步/秒（含PPO更新），不含评估/录像，固定场景库；只作为性能候选。
- 原生回合累计日志多扣10分，独立8回合测试复现并修复，返回给PPO的实际奖励未变；误差由10降至4.6e-14以内。正/反向、质量/延迟两场景同GPU后端控制奖励重置配对通过，未改变CPU物理等价门。
- 已在NativeEnv和扫描入口限制最多1024环境。2048中断、重启观察及4096未运行记入 `large_batch_interrupted.json`；原因未确认，不伪称通过或OOM已证实。
- 收敛实验目录 `wheelleg_warp/results/convergence_1024_20260921/`，服务 `wheelleg-native-convergence-1024.service`，24GiB内存上限且MemorySwapMax=0。三个配对种子，1024世界下n_steps=16/250、相同其余PPO配置，各2,048,000步，125/8次PPO更新，末次共同CPU开发32例。中间评估报告实际完成更新步数，不用最终保留集。
- 仅运行这个新收敛实验，不恢复旧正式训练；原生候选固定stage3训练场景库，原课程/每回合重采样仍未接入。学习曲线与效果结论尚待实验完成。

## 最新任务：GPU闭环与动画优化（2026-09-21）

- 用户指出GIF不流畅、Warp GPU较慢并要求尽可能优化。已分开处理：旧GIF25fps、直播5Hz限频/同步编码卡顿；GPU每0.5ms完整回读大量不需要字段。
- 新 `fast_physics.py` 打包已审计的控制/奖励必需字段，复用固定页内存，CUDA图中合并上传、原物理步、打包与回传，只同步当前流。原CPU和完整GPU回读类保持不变。
- 同一GPU状态回读逐值一致；非对称越障12000物理步实际接触与0/2/3接触数量变化检查通过。单环境800物理步三次中位数5.271→1.242秒，观测4.24倍。4000策略步GPU采样/更新167.882→58.331秒，观测2.88倍；CPU同预算34.890秒，GPU仍更慢，不扩大为已实现端到端GPU优势。
- CPU4000步最终权重逐值一致；GPU非逐位一致，最大参数差0.01030；已记录接触附近首次qpos数值差5.12e-9，不宣称跨运行训练轨迹逐位一致。全部证据见 `wheelleg_warp/results/packed_transfer_20260921/`。
- 动画已用完整50Hz状态生成50FPS GIF/WebP，旧25fps文件保留；编码在独立子进程，直播循环及发送上限约60Hz，实际FPS仍受单环境样本供应限制。前端明确区分真实实时与50FPS录像，无伪造插帧。WebP样例约为GIF的28%。
- 当前正式目录 `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/`；CPU从60000、GPU从20000更新检查点分段恢复策略/优化器/归一化，不用探针模型，不从零重练。两侧4000步smoke和各2000步恢复探针均通过。
- 环境及随机流重置，恢复步数不同，最终比较必须披露分段运行；当前段耗时不是无中断全程耗时。总策略预算仍是每种子200万、每后端3种子，原目录及历史成果全保留。
- 新正式服务于08:18:06实际启动，PID49625。前端已切到新目录；ETA扣除继承步数，等待新段实测再估计，不沿用旧4.4天结论。状态以run_config/progress/selection和服务为准。
- 详情及后续瓶颈见 `wheelleg_warp/OPTIMIZATION.md`；CPU控制/奖励及8个单世界GPU上下文仍限制吞吐，256世界开环704k物理步/秒不能冒充当前闭环/PPO吞吐。

## 真实训练画面第一版（原目录保留）

- 用户明确选择“训练进程原始逐帧画面”，不是另跑检查点回放。旧 `formal_cpu_warp_v1_20260921` 已停止并标为superseded，入口旧源码和全部产物保留于原目录；旧进度不混入新比较。
- 当前正式目录为 `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/`，新协议增加只读LiveRecorder，CPU和GPU均从相同种子重新初始化。双方4000步采集版真实PPO检查通过、初始权重一致；CPU最终4000步策略权重与无采集版逐值相同。
- 新正式监督服务于北京时间07:39:50实际启动，PID 41042；预算仍为两后端各3种子×200万步，各8环境，全部完成后自动共同CPU终评。物理后端差异与GPU当前较慢限制仍保留。
- 前端地址 `http://127.0.0.1:8765/`。`wheelleg-dashboard.service` 和 `wheelleg-training-render.service` 已启动、enabled且用户退出登录后持续运行。渲染器与训练进程独立。
- 画面来自每侧真实训练环境0，每50Hz策略步保存qpos/qvel/ctrl/action/obs/reward，物理仍2kHz。状态原子写入，渲染前匹配帧号/回合号；画面标明正式/预检、种子、环境、回合、仿真时间和数据年龄。评估期间采样暂停，显示最后真实帧。
- 所有完整回合NPZ在当前运行 `<backend_seed>/live/episodes/`；约每2万采样步选一个完整回合生成GIF，位于 `wheelleg_warp/dashboard/local_data/captures/`，前端可下载视频、轨迹和配置。大体积原始轨迹/GIF按用户本地保存要求保留在磁盘并被Git忽略，未远程备份；代码/协议/指标/模型检查点仍同步GitHub。
- 本轮验证：记录不改变状态/奖励/RNG；两侧4000步通过；CPU最终策略逐值一致；ETA边界和路径穿越/符号链接访问拒绝；浏览器真实画面、暂停/继续和下载链接；保存167帧轨迹可离线重放，首末图像一致，中间取样像素MAE约0.001/255，非跨GPU逐像素承诺。
- 07:44左右观测CPU约18000、GPU4000策略步，早期总体ETA约4.4天（粗区间2.2～8.9天）。此数值仅为当时吞吐外推，后续以前端/API实测为准；完整评估周期出现后纳入其开销，停机/失败/过期暂停ETA。
- 说明与重放命令见 `wheelleg_warp/dashboard/README.md`。回放可重现环境0已记录的运动状态，不等于保存全部8环境完整训练RNG和精确续训状态；强制中断时尚未结束回合可能只有最新状态落盘。

## CPU / GPU 对照第一版（无逐帧采集，已中止保留）

- 用户最新明确授权同时进行CPU/GPU两个基线的正式训练，最后比较效果；此前“本轮只做工程验收”的范围限制由本次授权更新。GPU对齐失败仍保留，不能重标为等价CPU基线。
- 冻结新协议 `wheelleg_warp/results/formal_cpu_warp_v1_20260921/protocol.json`：M3，CPU/Warp各3种子1609/1610/1611，每种子200万策略步，合计1200万新策略步；每侧8环境，双队列并行，队列内串行种子。
- 新两侧重新同种子初始化；历史CPU续训继续保留且不计入配对。PPO网络两侧均用CPU，以隔离物理后端变量；Warp物理使用GPU，控制与奖励仍在CPU，不承诺GPU端到端加速。
- 新增 `gpu_env.py` 独立环境适配，无需修改冻结CPU源码；修正GPU对外episode时钟的float32累加偏差，使用整数物理步时钟。真实非零动作/重置/结束与CPU模块隔离检查通过。
- 两侧4000步短更新均已通过：CPU采样/更新35.31秒、全程59.49秒；GPU采样/更新186.56秒、全程215.49秒。初始策略权重SHA256完全一致。短更新证据仅作管线准入，正式权重从同种子重新初始化。
- 两侧使用同一32例CPU选择集；全部训练完成后自动在提前固定的64例新IID场景比较selected和last检查点，输出 `COMPARISON.md` 与 `comparison.json`。原研究门控/最终集不用于本次比较。
- 合成汇总、真实GPU环境适配、两侧PPO短更新均通过，准入记录为实验目录 `readiness.json`。用户服务 `wheelleg-cpu-warp-formal.service` 已于北京时间2026-09-21 07:16:55实际启动（监督PID 35212），自动启动CPU/Warp队列并在双方全部完成后终评。实时状态见 `status.json` 和各run的 `progress.json`；不能把active或中途进度称为完整预算完成。

## CPU / PPO 已知状态

- GPU 分支的原始 CPU 源码基点为 `0f70b834bd8071f485bc81b1d6b1f99cf266d5de`；本轮开始时未提交变更均为在跑训练的日志/检查点。本轮未将这些训练结果归为 GPU 实验。
- `wheelleg_ppo/PAPER_PLAN.md` 第12节为 v2 研究协议；28例开发场景基线已有通过记录，困难组12/12，不能仍用不可达的“再提高10个百分点”主门。研究门沿已有v2协议，不在本轮重调。
- CPU 实验服务 `wheelleg-m3-1609-v2-resume1.service` 本轮开始时仍运行。输出目录 `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/`；当前进度以其中完整指标JSON和 selection/completed 为准，不用本文件中的静态数字替代实时状态。
- 原15分钟自动跟进任务已取消，不代表训练服务已停止。没有自动调度后续18项队列的承诺。
- 之前只把 PPO 网络置于 CUDA 的比较没有提速，见计划12.10。新 Warp 基线改变物理执行后端，不能直接继承旧实验训练成绩或协议源码哈希。

## 本轮实施与验证

- 新增 `wheelleg_warp/baseline.py`、固定版本依赖及运行文档；复用 CPU 模型构建、VMC＋六状态LQR＋零残差和最终电机限幅。GPU 使用 `cuda:0`、MuJoCo Warp 3.12.0、Warp 1.17.0、MuJoCo 3.12.0。只增加两个GPU依赖，原CPU依赖版本保持不变。
- CPU/GPU 分别运行独立控制器状态，测试站立、已见参数范围的非对称接触、跳跃；CPU模型原参数不改。固定工程校对阈值 qpos 0.02、qvel 0.5、ctrl 2.0，接触障碍集合一致。阈值不是论文成功门，也不是逐位一致承诺。
- 批量吞吐为相同初态、同一CPU控制轨迹的开环物理回放，CUDA graph 执行；CPU对照采用8线程原生 `mujoco.rollout`，不以Python循环人为压低CPU速度。
- 冒烟首轮因 Python tuple 使用 abs 报错；修正为 NumPy abs，失败记录保留于 `wheelleg_warp/results/smoke_20260921/summary.json`。
- 修正后冒烟通过，见 `wheelleg_warp/results/smoke_fix_20260921/summary.json`：0.1秒轨迹 qpos 最大误差2.51e-7；256环境×200步物理回放单次GPU/CPU速度比2.496，末态qpos误差7.09e-7。与原CPU训练同时运行，非独占机器性能结论。
- 完整校对首轮站立6秒通过；非对称接触暴露MJWarp 3.12回读未同步geom1/geom2的问题，失败保留于 `wheelleg_warp/results/paired_20260921/summary.json`。已在GPU唯一回读入口同步contact.geom到旧字段，原CPU控制器不改。针对真实接触、主动污染旧字段的 `wheelleg_warp/test_contact_bridge.py` 已通过。修正后的完整6秒×3场景运行于 `wheelleg_warp/results/paired_contact_fix_20260921/summary.json`；三项均已完整运行6秒：站立与跳跃通过，非对称接触全状态校对失败，总状态alignment_failed（入口按设计退出1）。非对称场景 qpos最大误差0.17224、qvel最大误差11.33987、ctrl最大误差1.31268；双方均接触bump_L，末态车体位置差约0.28mm，但不能据此绕过已设全状态门。GPU保留为候选基线，原CPU仍为正式效果参照。
- `python3 tools/test_git_sync.py` 已通过：自动快照/推送、中文和记忆钩子拒绝、保留既有暂存内容、远程分叉不强推。服务已安装并enable，systemd配置检查通过；登录驻留 `Linger=yes` 已开启，收尾提交时启动并核对远程。

## 未关闭项与后续边界

- CPU参考标签 `cpu-reference-pre-warp-20260921` 和 `wheelleg_warp/CPU_REFERENCE.json` 已建立；新入口运行前验证27个CPU源码/模型哈希，变化时拒绝混用原基线。
- 新基线的 Python 控制器仍在CPU，校对每物理步回读完整状态；该路径预计慢于原CPU。批量物理加速不代表完整闭环或PPO训练加速；尚未迁移 Gym/PPO 采样到 GPU。
- 只有工程对齐和受控吞吐实测可在本轮验收；新后端长期训练需另行冻结协议及验证，不能直接替换在跑 CPU 训练。
- 完整校对若失败，保留误差与失败证据，不放宽门槛，不替换原CPU。
- 同步失败用 `journalctl --user -u wheelleg-git-sync.service -n 30 --no-pager` 排查；后台同步不等于会自动发送聊天通知。身份/网络/分叉问题不能通过强推解决。

### 本轮维护检查

- 依赖一致性 `pip check` 通过；新增代码编译检查、`git diff --check`、systemd服务配置检查通过。
- 原CPU参考标签已实际推送远程：`cpu-reference-pre-warp-20260921`。
- 全状态对齐失败属于已记录的基线限制，不能写成“全校对通过”；后续只在用户要求时诊断碰撞/精度差异或迁移GPU原生闭环。

### 最终GPU验收结果（2026-09-21）

- `paired_contact_fix_20260921/summary.json`：站立最大qpos误差2.26e-6；跳跃最大qpos误差0.001499、qvel误差0.01183、ctrl误差0.00358，均通过预设门。非对称接触未通过，原CPU不替换。
- 256世界×200步回放：GPU 0.07270秒，8线程CPU 0.18369秒，单次倍率2.5266，末态qpos误差7.09e-7。只代表所述物理执行路径，存在输出量及后台负载差异，未验证PPO端到端加速。
- CPU控制器逐步回读路径约47～50秒完成6秒仿真，CPU配对路径约5.8秒；GPU原生控制/环境/PPO仍未迁移。
- 所有失败JSON保留，未调整原先门槛；原CPU27项哈希最后核对仍一致。同步脚本的回归、接触字段回归、依赖与配置检查通过。

### 同步服务上线

- `wheelleg-git-sync.service` 已实际启动且active，enabled和Linger=yes已核验；本轮提交由守护进程推送，最终同步结果以远程HEAD校验为准。

### 自动同步快照 2026-09-21T06:51:23+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_320000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_320000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_320000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_340000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_340000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_340000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_360000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_360000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_360000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_380000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_380000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_380000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_400000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_400000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_400000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_420000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_420000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_420000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_440000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_440000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_440000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_460000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_460000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_460000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_480000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_480000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_480000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_500000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_500000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_500000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_520000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_520000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_520000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_540000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_540000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_540000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_560000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_560000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_560000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_580000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_580000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_580000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_600000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_600000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_600000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_620000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_620000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_620000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_640000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_640000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_640000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_660000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_660000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_660000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_680000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_680000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_680000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_700000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_700000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_700000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_720000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_720000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_720000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_740000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_740000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_740000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_760000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_760000.zip`

### 自动同步快照 2026-09-21T06:52:36+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_760000.json`

### 自动同步快照 2026-09-21T06:54:44+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_780000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_780000.zip`

### 自动同步快照 2026-09-21T06:56:55+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_780000.json`

### 自动同步快照 2026-09-21T06:58:32+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_800000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_800000.zip`

### 自动同步快照 2026-09-21T07:00:41+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_800000.json`

### 自动同步快照 2026-09-21T07:02:19+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_820000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_820000.zip`

### 双后端正式训练交付说明

- 正式监督服务单次运行，不自动重启覆盖中断目录；用户退出登录后继续运行（Linger已启用）。系统关机/崩溃后需显式审核断点恢复，不能默认为精确续训。
- 两队列完成后自动共同终评并写比较报告，Git同步守护归档推送；本轮结束时尚无最终效果结论。
- 已有CPU历史实验继续运行，资源竞争已在协议披露。GPU接触数值差异和当前逐步回读较慢均未隐瞒，不为求提速改变预算、控制、奖励或评估。

### 自动同步快照 2026-09-21T07:18:43+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_820000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_840000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_840000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_840000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_860000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_860000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_860000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_880000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_880000.zip`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/status.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp_1609/run_config.json`

### 自动同步快照 2026-09-21T07:19:51+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp_1609/progress.json`

### 自动同步快照 2026-09-21T07:47:24+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_880000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_900000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_900000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_900000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_920000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_920000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_920000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_940000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_940000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_940000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_960000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_960000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_960000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_20000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_20000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/status.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/step_20000.json`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/step_20000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/cpu_1609/step_20000.zip`
- `wheelleg_warp/results/formal_cpu_warp_v1_20260921/warp_1609/progress.json`

### 优化后真实续训核验

- 08:24左右已确认CPU从60000继续至68000、GPU从20000继续至26000，均有新PPO更新，非仅进程active。当前段早期有效采样更新速率约CPU76/GPU63策略步每秒（8环境合计），仍待完整周期评估稳定后判断全程倍率。
- 4000步CPU权重不变、GPU非逐位一致以及初次Warp内核参数编译错误均保留证据；前者不扩大为整个GPU训练精确可复现。
- 续训ETA另剔除恢复点一次性补评估，不把继承旧步数或启动补评估误算成新采样吞吐，专项测试通过。

### 自动同步快照 2026-09-21T08:27:47+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1000000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1000000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1000000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1020000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1020000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1020000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1040000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1040000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1040000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1060000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1060000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1060000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1080000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1080000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1080000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_980000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_980000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_980000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_60000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_60000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_60000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_80000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_80000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/status.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/run_config.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_20000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_20000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_20000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_20000.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_40000.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_40000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_40000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_60000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/cpu_1609/step_60000.zip`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/step_20000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_live_v1_20260921/warp_1609/step_20000.zip`

### 自动同步快照 2026-09-21T08:28:26+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_40000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_40000.zip`

- 录像连续刷新时同步切换图像资源与来源标签，避免新录像标签仍对应旧图像；该前端修正不改训练入口或物理。

### 自动同步快照 2026-09-21T08:31:12+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1100000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1100000.zip`

### 自动同步快照 2026-09-21T08:31:47+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_80000.json`

### 自动同步快照 2026-09-21T08:32:21+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_40000.json`

### 自动同步快照 2026-09-21T09:54:13+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1.log`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/selection.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1100000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1120000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1120000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1120000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1140000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1140000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1140000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1160000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1160000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1160000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1180000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1180000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1180000.zip`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1200000.json`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1200000.pkl`
- `wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1/step_1200000.zip`
- `wheelleg_warp/check_native_episode_return.py`
- `wheelleg_warp/native/benchmark.py`
- `wheelleg_warp/native/check_controller.py`
- `wheelleg_warp/native/controller.py`
- `wheelleg_warp/native/environment.py`
- `wheelleg_warp/native/models.py`
- `wheelleg_warp/native/test_native.py`
- `wheelleg_warp/profile_gpu_path.py`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu.log`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_100000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_100000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_100000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_120000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_120000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_120000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_140000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_140000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_140000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_160000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_160000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_160000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_180000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_180000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/cpu_1609/step_180000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp.log`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/progress.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/selection.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_100000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_100000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_100000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_120000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_120000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_120000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_140000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_140000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_60000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_60000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_60000.zip`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_80000.json`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_80000.pkl`
- `wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921/warp_1609/step_80000.zip`
- `wheelleg_warp/results/gpu_saturation_20260921/profile.json`
- `wheelleg_warp/results/native_gpu_20260921/bench_128.json`
- `wheelleg_warp/results/native_gpu_20260921/bench_32.json`
- `wheelleg_warp/results/native_gpu_20260921/engineering.json`
- `wheelleg_warp/results/native_gpu_20260921/engineering_initial_short.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_1.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_1.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_12.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_12.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_16.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_16.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_20.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_20.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_24.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_24.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_32.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_32.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_4.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_4.log`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_8.json`
- `wheelleg_warp/results/parallel_sweep_20260921/cpu_8.log`
- `wheelleg_warp/results/parallel_sweep_20260921/machine.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_1024.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_1024.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_128.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_128.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_2048.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_256.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_256.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_32.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_32.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_512.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_512.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_64.json`
- `wheelleg_warp/results/parallel_sweep_20260921/native_64.log`
- `wheelleg_warp/results/parallel_sweep_20260921/refine.log`
- `wheelleg_warp/results/parallel_sweep_20260921/sweep.log`
- `wheelleg_warp/results/parallel_sweep_20260921/sweep_events.jsonl`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_1.json`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_1.log`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_12.json`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_12.log`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_4.json`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_4.log`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_8.json`
- `wheelleg_warp/results/parallel_sweep_20260921/warp_8.log`

### 自动同步快照 2026-09-21T10:22:29+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/convergence_1024_20260921/cpu_reference_check.json`
- `wheelleg_warp/results/convergence_1024_20260921/historical_cpu_context.json`
- `wheelleg_warp/results/convergence_1024_20260921/resource_snapshot.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609.log`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/completed.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/curve.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/progress.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/run_config.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_0.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_0.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_0.zip`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_1032192.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_1032192.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_1032192.zip`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_2048000.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_2048000.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_2048000.zip`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_262144.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_262144.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_262144.zip`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_524288.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_524288.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1609/step_524288.zip`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/curve.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/run_config.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_0.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_0.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_0.zip`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_256000.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_256000.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_256000.zip`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_512000.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_512000.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_512000.zip`
- `wheelleg_warp/results/convergence_1024_20260921/status.json`
- `wheelleg_warp/results/parallel_sweep_20260921/episode_return_after.log`
- `wheelleg_warp/results/parallel_sweep_20260921/episode_return_before.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_pair.log`
- `wheelleg_warp/results/parallel_sweep_20260921/native_pair_initial_failure.log`

### 自动同步快照 2026-09-21T10:23:20+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_1024000.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_1024000.zip`

### 自动同步快照 2026-09-21T10:24:02+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609.log`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/curve.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/progress.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_1024000.json`
- `wheelleg_warp/results/parallel_sweep_20260921/batch_limit_check.json`

### 自动同步快照 2026-09-21T10:25:09+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609.log`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/progress.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_2048000.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_2048000.zip`

### 自动同步快照 2026-09-21T10:27:21+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609.log`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/completed.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/curve.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/progress.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_250_seed_1609/step_2048000.json`
- `wheelleg_warp/results/convergence_1024_20260921/status.json`

### 自动同步快照 2026-09-21T10:27:55+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1610/run_config.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1610/step_0.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1610/step_0.zip`

### 自动同步快照 2026-09-21T10:28:31+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1610/curve.json`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1610/step_0.json`

### 自动同步快照 2026-09-21T10:29:05+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1610/step_262144.pkl`
- `wheelleg_warp/results/convergence_1024_20260921/rollout_16_seed_1610/step_262144.zip`

### 自动同步快照 2026-09-22T02:53:05+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_004.log`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_004/completed.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_004/run_config.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_004/step_6180000.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_004/step_6180000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_004/step_6180000.zip`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_004/step_6692000.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_004/step_6692000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_004/step_6692000.zip`
- `wheelleg_warp/results/terrain_v1_1024_20260922/selection.json`

### 自动同步快照 2026-09-22T02:53:41+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_005/run_config.json`

### 自动同步快照 2026-09-22T02:54:16+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_005/step_6692000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_005/step_6692000.zip`

### 自动同步快照 2026-09-22T02:54:52+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_005.log`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_005/step_6692000.json`

### 自动同步快照 2026-09-22T02:55:27+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_005/completed.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_005/step_7204000.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_005/step_7204000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_005/step_7204000.zip`
- `wheelleg_warp/results/terrain_v1_1024_20260922/selection.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/status.json`

### 自动同步快照 2026-09-22T02:56:33+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_006/run_config.json`

### 自动同步快照 2026-09-22T02:57:08+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_006.log`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_006/step_7716000.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_006/step_7716000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_006/step_7716000.zip`

### 自动同步快照 2026-09-22T02:57:45+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_006/step_8228000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_006/step_8228000.zip`

### 自动同步快照 2026-09-22T02:58:21+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_006/completed.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_006/step_8228000.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/selection.json`

### 自动同步快照 2026-09-22T02:58:55+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_007/run_config.json`

### 自动同步快照 2026-09-22T02:59:30+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_007.log`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_007/step_8740000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_007/step_8740000.zip`

### 自动同步快照 2026-09-22T03:00:06+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_007/step_8740000.json`

### 自动同步快照 2026-09-22T03:00:41+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_007/completed.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_007/step_9252000.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_007/step_9252000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_007/step_9252000.zip`
- `wheelleg_warp/results/terrain_v1_1024_20260922/selection.json`

### 自动同步快照 2026-09-22T03:01:19+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_008/run_config.json`

### 自动同步快照 2026-09-22T03:01:53+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_008/step_9252000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_008/step_9252000.zip`

### 自动同步快照 2026-09-22T03:02:28+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_008.log`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_008/step_9252000.json`

### 自动同步快照 2026-09-22T03:03:02+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/round_008/step_9764000.pkl`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_008/step_9764000.zip`

### 自动同步快照 2026-09-22T03:03:39+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v1_1024_20260922/final_evaluation.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_008/completed.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/round_008/step_9764000.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/selection.json`
- `wheelleg_warp/results/terrain_v1_1024_20260922/status.json`

### 自动同步快照 2026-09-22T04:44:48+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_002.log`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_002/completed.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_002/run_config.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_002/step_12324000.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_002/step_12324000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_002/step_12324000.zip`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_002/step_12836000.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_002/step_12836000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_002/step_12836000.zip`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_003/run_config.json`

### 自动同步快照 2026-09-22T04:45:27+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_003/step_12324000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_003/step_12324000.zip`

### 自动同步快照 2026-09-22T04:46:02+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_003.log`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_003/step_12324000.json`

### 自动同步快照 2026-09-22T04:46:37+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_003/completed.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_003/step_12836000.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_003/step_12836000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_003/step_12836000.zip`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004.log`
- `wheelleg_warp/results/terrain_v3_1024_20260922/selection.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/status.json`

### 自动同步快照 2026-09-22T04:47:44+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004/run_config.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004/step_12836000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004/step_12836000.zip`

### 自动同步快照 2026-09-22T04:48:19+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004.log`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004/step_12836000.json`

### 自动同步快照 2026-09-22T04:48:54+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004/step_13348000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004/step_13348000.zip`

### 自动同步快照 2026-09-22T04:49:29+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004/completed.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_004/step_13348000.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/selection.json`

### 自动同步快照 2026-09-22T04:50:05+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_005/run_config.json`

### 自动同步快照 2026-09-22T04:50:39+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_005.log`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_005/step_12836000.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_005/step_12836000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_005/step_12836000.zip`

### 自动同步快照 2026-09-22T04:51:15+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_005/step_13348000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_005/step_13348000.zip`

### 自动同步快照 2026-09-22T04:51:51+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_005/completed.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_005/step_13348000.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/selection.json`

### 自动同步快照 2026-09-22T04:52:25+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_006/run_config.json`

### 自动同步快照 2026-09-22T04:53:00+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_006.log`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_006/step_12836000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_006/step_12836000.zip`

### 自动同步快照 2026-09-22T04:53:38+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_006/step_12836000.json`

### 自动同步快照 2026-09-22T04:54:12+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/round_006/completed.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_006/step_13348000.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_006/step_13348000.pkl`
- `wheelleg_warp/results/terrain_v3_1024_20260922/round_006/step_13348000.zip`
- `wheelleg_warp/results/terrain_v3_1024_20260922/selection.json`

### 自动同步快照 2026-09-22T04:54:47+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v3_1024_20260922/final_evaluation.json`
- `wheelleg_warp/results/terrain_v3_1024_20260922/status.json`

### 自动同步快照 2026-09-22T05:44:16+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v4_diagnostics_20260922/controller_trace.json`

### 自动同步快照 2026-09-22T05:46:27+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `wheelleg_warp/results/terrain_v4_diagnostics_20260922/step_capability.json`

### 自动同步快照 2026-09-22T05:48:06+08:00

稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。

- `PROJECT_MEMORY.md`
- `wheelleg_warp/TERRAIN_PLAN.md`
- `wheelleg_warp/results/terrain_v4_diagnostics_20260922/REPORT.md`
- `wheelleg_warp/results/terrain_v4_diagnostics_20260922/readiness.json`
