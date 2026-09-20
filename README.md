# 轮腿机器人仿真与残差强化学习

基于 **MuJoCo、虚拟模型控制（VMC）、六状态 LQR 和 PPO 残差学习**的轮腿机器人研究项目。项目先建立可复验的站立、行驶、制动与跳跃控制基线，再研究左右接触不对称时的航向精度，并保留独立的 ROS2 毕设仿真基线。

本仓库是仿真与研究工作区。当前主入口为 Python，ROS2 节点、导航系统和实机部署尚未实现。历史 C++ 控制器与当前 Python 控制器也不等价。

## 导航

- [项目目的与当前状态](#项目目的与当前状态)
- [目录结构](#目录结构)
- [部署与快速运行](#部署与快速运行)
- [控制与学习实现](#控制与学习实现)
- [PPO 实验与复现](#ppo-实验与复现)
- [验证与结果管理](#验证与结果管理)
- [常见问题](#常见问题)
- [进一步阅读与许可](#进一步阅读与许可)

## 项目目的与当前状态

项目围绕两条独立目标展开：

1. **论文研究**：在固定名义 VMC＋LQR 控制器上叠加小幅学习残差，研究不对称凸台、左右摩擦差、驱动差和观测延迟下的航向控制；比较差模动作空间与经典反馈、六维残差、同维混合动作空间。
2. **毕设工程**：以已有平衡与运动控制为基础，后续接入 ROS2 通信、状态发布、可视化、定位与导航。该方向不依赖 PPO 优于基线才能推进。

| 子项目 | 用途 | 当前状态 |
| --- | --- | --- |
| `mujoco_wheelleg-main/` | 原始仿真、控制演进、硬件建模与历史证据 | 保留 Python 入口和历史 C++ 实现 |
| `wheelleg_ppo/` | PPO 论文研究主线 | 环境、残差接口、训练与恢复工具已实现；v2 预实验已启动 |
| `wheelleg_ros2/` | ROS2 毕设基线 | 可独立运行 Python 仿真；ROS2 尚未接入，开发暂缓 |

**研究状态快照：2026-09-21。** 已有 M3/1609 的 20 万策略步阶段核验，工程链路通过，但阶段最优策略尚未优于同选择集的经典基线，不能据此宣称方法优势或稳定收敛。64 例门控集与最终测试集尚未评估。后续状态以 [PAPER_PLAN.md](wheelleg_ppo/PAPER_PLAN.md) 和完整落盘的实验记录为准。

两个独立基线目录不在运行时导入对方源码。改动不会自动同步到 ROS2 目录，跨项目移植需要单独验证。

## 目录结构

下面列出部署和理解项目所需的主要文件，省略大量历史结果与素材。

```text
wheel_leg_robot_work/
├── README.md                         # 本文：项目总入口
├── requirements.txt                  # 根目录历史最小依赖；非 PPO 完整环境
├── run_wheelleg.sh                    # 原开发机启动脚本，含绝对路径
├── boot_test.py / lqr_*.py            # 早期调试与设计脚本
├── .gitignore / .gitmodules          # 忽略规则与参考子模块
├── AGENTS.md / .agents/              # 开发协作约束，不参与仿真运行
├── mujoco_wheelleg-main/
│   ├── tools/                        # Python 仿真、建模和回归
│   ├── xml/ / asset/                 # MJCF 模型、纹理
│   ├── include/ / mujoco/ / data/    # 历史 C++ 控制与窗口代码
│   ├── CMakeLists.txt / cmake/       # C++ 构建入口
│   ├── matlab/ / examples/          # 设计计算与参考实现
│   └── PPT_MATERIALS/ / *.md        # 演示素材、硬件与历史报告
├── wheelleg_ppo/
│   ├── requirements.txt             # PPO 完整 Python 依赖
│   ├── xml/ / asset/                # 论文项目独立模型与素材
│   ├── BASELINE_MANIFEST.json        # 基线来源与哈希
│   ├── PAPER_PLAN.md                # 研究协议、阶段进展与边界
│   ├── PREPPO_FIX_REPORT.md          # 训练前控制与环境修复证据
│   ├── LITERATURE_*.md              # 文献矩阵与全文核对
│   └── tools/
│       ├── wheelleg_sim.py          # 状态机、交互仿真、控制输出总入口
│       ├── hardware_profile.py      # 几何、质量、惯量与力矩包络
│       ├── model_lqr.py             # 平衡点、线性化与 LQR 设计
│       ├── rm_controller.py         # 六状态 LQR＋VMC 控制器
│       ├── state_estimation.py      # 五连杆运动学与影子估计器
│       ├── ppo_env.py               # Gymnasium 环境和残差映射
│       ├── prepare_ppo.py           # 工程准入与短更新检查
│       ├── pretrain_yaw.py          # 航向协议与来源检查
│       ├── train_yaw.py             # 冻结协议的 PPO 训练入口
│       ├── resume_yaw.py            # 检查点恢复
│       ├── review_yaw_stage.py      # 20 万步阶段审查
│       ├── score_yaw_gate.py        # 已采集门控结果的离线评分
│       ├── test_*.py                # 控制、动力学、训练与评分检查
│       └── results/                 # 协议、配置、日志、模型与证据
├── wheelleg_ros2/
│   ├── requirements.txt             # 基础仿真依赖
│   ├── tools/ / xml/ / asset/       # 独立的传统控制仿真基线
│   ├── BASELINE_MANIFEST.json
│   └── GRADUATION_PLAN.md           # 尚未实施的 ROS2 集成计划
└── ref_balance/                     # Gitee 参考工程，Git 子模块
```

`results/` 不只是临时输出：其中包含冻结的场景清单、训练配置、准入证据和历史基线，训练入口会读取这些文件。不要为了“清理日志”整体删除该目录。

## 部署与快速运行

### 1. 获取项目

```bash
git clone https://github.com/w1731331226-code/wheel_leg_robot_work.git
cd wheel_leg_robot_work
```

主仿真不依赖参考子模块。如需阅读参考工程，再执行：

```bash
git submodule update --init --recursive
```

该命令需要访问 `.gitmodules` 中的 Gitee 地址。子模块下载失败不影响三个主目录内的 Python 仿真。

### 2. 创建基础仿真环境

以下命令面向 Linux/Bash；当前开发环境使用 Python **3.10.12**。交互窗口需要可用的 OpenGL/GLFW 图形环境；无窗口测试不需要启动桌面窗口。其他系统尚未在本文中完成部署验收。

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r wheelleg_ros2/requirements.txt
python -m pip check
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

这里复用 ROS2 基线目录的依赖清单来安装基础仿真环境，**并不会安装 ROS2**。它包含 MuJoCo 3.12.0、NumPy 2.2.6、SciPy 1.15.3 和 GLFW 2.10.2，足以运行 Python 传统控制仿真。仅安装根目录的 `requirements.txt` 不足以运行 PPO。

Ubuntu/Debian 若缺少 `venv` 或窗口运行库，可按实际报错安装对应系统包：

```bash
sudo apt-get update
sudo apt-get install python3-venv libglfw3 libgl1
```

### 3. 首次无窗口验证

保持虚拟环境激活，在仓库根目录运行：

```bash
python wheelleg_ppo/tools/wheelleg_sim.py --test stand
```

此命令加载模型、设计名义控制器并执行站立场景，成功返回退出码 `0`，失败返回非零。首次求解 LQR 需要初始化计算。它验证基础控制链可运行，不代表全部鲁棒性场景通过。

### 4. 启动交互仿真

```bash
python wheelleg_ppo/tools/wheelleg_sim.py --size 1280x720
# 或全屏
python wheelleg_ppo/tools/wheelleg_sim.py --fullscreen
```

| 按键 | 功能 |
| --- | --- |
| `W` / `S` | 按住前进 / 后退，松开取消速度命令 |
| `A` / `D` | 按住转向，松开取消转向命令 |
| `↑` / `↓` | 按住调整腿长 |
| `7` 或小键盘 `7` | 提交一次跳跃请求，由状态机判断执行时机 |
| `Esc` | 关闭窗口 |

默认已启用硬件名义模型和六状态控制，无需额外添加 `--hardware --six-state`。窗口失去焦点时会清零按住状态。该入口使用传统控制器，**不会自动加载 PPO 检查点**。

其他子项目可使用相同环境单独运行：

```bash
python wheelleg_ros2/tools/wheelleg_sim.py --test stand
python mujoco_wheelleg-main/tools/wheelleg_sim.py --test stand
```

`run_wheelleg.sh` 写死了原开发机 `/home/wmt/...` 路径。新机器使用上面的 Python 命令即可。

### 5. 安装 PPO 依赖

只运行基础仿真可跳过此步。普通 PPO 环境安装：

```bash
python -m pip install -r wheelleg_ppo/requirements.txt
python -m pip check
```

当前冻结 v2 实验会精确核对依赖版本，其中 PyTorch 为 **`2.14.0+cu130`**。单纯安装 `torch==2.14.0` 可能得到不同构建，普通环境可用不等于满足冻结实验要求。按已有实验环境安装对应构建：

```bash
python -m pip install 'torch==2.14.0+cu130' \
  --extra-index-url https://download.pytorch.org/whl/cu130
python -m pip install -r wheelleg_ppo/requirements.txt
python -m pip check
```

上述版本来自本仓库配置，并非对任意平台的软件包可用性保证。如果安装源或当前平台没有该构建，应先解决环境兼容性，不能删除版本检查来冒充复现。

冻结配置目前使用 `device=cpu`，CUDA 构建同样可执行 CPU 训练。MuJoCo 物理采样仍在 CPU；已有本机设备比较没有显示 GPU 加速收益。若做新的 GPU 实验，需要单独验证驱动、PyTorch 设备和实际性能，不能直接覆盖已有冻结配置。

## 控制与学习实现

### 1. 机器人模型

每侧腿由两个主动关节构成闭环五连杆，两侧共四个髋执行器，加上两个轮执行器，共 **六路力矩输出**。MJCF 描述关节、闭环约束、接触、传感器与执行器；`hardware_profile.configure_spec()` 在编译模型前应用名义硬件参数，因此运行模型并非只由 XML 原始数值决定。

论文目录当前参数如下，历史报告中的其他轮径/轮距不是当前默认值：

| 参数 | 当前名义值 |
| --- | --- |
| 总质量 | 7 kg |
| 左右轮心距 | 300 mm |
| 轮半径 / 胎宽 | 50 mm / 55 mm |
| 中央箱体 / 上平台宽度 | 160 mm / 220 mm |
| 髋执行器模型 | DM8009，单路峰值力矩包络 ±40 N·m |
| 轮执行器模型 | MF9025，单路峰值力矩包络 ±4.5 N·m |
| 物理时间步长 | 0.0005 s，即 2 kHz |
| 积分器 | MuJoCo `implicitfast` |

当前 `IDEAL_POWER_MODEL=True`，供电、热与回生采用理想假设；部分结构质量、惯量是设计预算或估计值。仿真力矩包络不等于实机持续输出能力，尚未完成实体标定。

### 2. 传统控制器：六状态 LQR＋VMC＋状态机

`model_lqr.py` 从模型求平衡点、线性化和离散 LQR；`rm_controller.py` 在多个腿长工作点建立增益表，并按当前腿长插值。六个反馈状态为：

```text
x = [腿角误差, 腿角速度, 纵向位置误差, 纵向速度误差, 机身 pitch, pitch 角速度]
u = u_equilibrium - K(length) · x
```

LQR 处理纵向平衡的共模轮力矩与腿部虚拟力矩；VMC 支撑环处理腿长、腿速和横滚修正。五连杆运动学给出极坐标雅可比，通过虚功映射把腿部径向支撑力与角向力矩变换为主动关节力矩。代码中的 `leg_kinematics()` 已返回用于该映射的转置矩阵，不应再重复转置。

行驶和跳跃共用 `wheelleg_sim.py` 状态机，主状态包括 `DRIVE → SQUAT → JUMP → FLY → LAND → DRIVE`，另有准备、提交与恢复条件。六状态控制器主要替换 `DRIVE` 和 `LAND` 的地面共模输出，保留跳跃阶段控制。

控制器默认从固定名义模型设计，每进程缓存一次。场景随机化改变被控对象，不随每个 episode 重新设计 LQR，否则会把未知物理参数提前交给控制器。`design_controller()` 是显式离线设计接口。

### 3. 残差叠加与约束

每个物理步的控制顺序如下：

```text
状态与命令
  → 原状态机 / VMC 控制
  → 六状态 LQR 地面共模替换
  → 基控制输出可行性检查
  → PPO 残差映射与余量缩放
  → 最终力矩限幅、峰值计时
  → MuJoCo mj_step
```

策略输出为 `[-1, 1]` 内的归一化动作。`Residual` 先限制动作变化率，再映射到六路力矩；根据基础输出剩余的执行器余量，计算公共缩放系数 `λ∈[0,1]`，执行 `u_base + λ·u_residual`。最终统一限幅，每个物理步只提交一次峰值计时。基础输出不可行、映射失效或不处于 `DRIVE` 时禁用残差。

| 动作模式 | 维数 | 含义 / v2 方法 |
| --- | --- | --- |
| `diff1` | 1 | 左右轮差动力矩；M1 |
| `diff2` | 2 | 腿径向支撑差模＋轮差模；保留的环境接口 |
| `diff3` | 3 | 腿径向力、腿角向力矩、轮力矩三类差模；M3 |
| `mixed3_plus` / `mixed3_minus` | 3 | 腿通道混入正/负共模的同维对照；N3+ / N3− |
| `virtual6` | 6 | 左右腿虚拟力/力矩与左右轮力矩独立输出；B2-V |
| `torque6` | 6 | 六路执行器直接力矩残差；B2 |

差模动作对左右两侧采用相反符号，用于研究不对称接触修正。它是否优于其他动作空间需要完整对照，不能仅凭维数更低得出结论。

### 4. Gymnasium 环境与任务

`WheelLegEnv` 的策略频率为 **50 Hz**，每个动作推进 40 个 2 kHz 物理步。32 维观测包含姿态、角速度、机体坐标速度、速度命令与误差、航向、主动关节位置/速度、轮速、腿长/腿速和上一步实际残差；可按场景延迟返回观测。

场景覆盖前进/后退、左右凸台高度、摩擦差、质量、驱动增益差与观测延迟，并区分训练、验证、IID、组合保留与范围外测试。机器人起步后通过接触区域，到达目标后停车观察。奖励综合运动误差与控制代价，最终评价还单独检查任务完成、姿态、速度跟踪与停车表现。

`completed` 只表示走完流程；`success` 还要求命中规定凸台、姿态峰值不超过 5°、速度 RMSE 不超过命令速度幅值的 20%、停车位移不超过 0.60 m、末段速度不超过 0.03 m/s。跌倒、非轮接触、映射失效、非有限状态或超时会触发终止/截断。

当前控制与环境仍使用仿真真值信息。`state_estimation.py` 中的传感器影子估计器没有替代正式反馈，因此结果不是完整的实机传感闭环验证。

## PPO 实验与复现

### 冻结配置

当前使用 [v2 协议](wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21/PROTOCOL.md) 和 [训练配置](wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21/training_config.json)。

| 项目 | 值 |
| --- | --- |
| 算法与网络 | SB3 PPO，`MlpPolicy`，隐藏层 `[64, 64]` |
| 并行环境 | 8 个 `SubprocVecEnv` |
| 单环境 rollout / batch | 250 / 250 |
| 更新 epochs / 学习率 | 10 / 0.0003 |
| γ / GAE λ / clip | 0.99 / 0.95 / 0.2 |
| 归一化 | `VecNormalize` 归一化观测，不归一化奖励；评估时冻结统计量 |
| 单方法单种子预算 | 2,000,000 策略步 |
| 检查点间隔 | 20,000 策略步，固定 32 例选择集 |
| 训练种子 | 1609、1610、1611 |
| 课程阶段切换 | 0、20,000、100,000 策略步 |

策略步与物理步不同，不能把预算理解成 200 万次物理积分。v2 注册六个学习方法，合计 18 个预实验、3600 万策略步为预算，并非已完成工作量。

### 短更新验证

以下命令均从仓库根目录、已激活的完整 PPO 环境执行。输出目录必须不存在；每次复验更换目录名。

```bash
python wheelleg_ppo/tools/train_yaw.py \
  --protocol v2 --method M1 --seed 1609 --smoke \
  --output wheelleg_ppo/tools/results/local_smoke_M1
```

`--smoke` 执行 4000 策略步，核对参数更新、有限性、保存重载、归一化冻结和课程 reset，不用于证明性能或初始化正式策略。默认协议是历史 v1，当前代码应显式使用 `--protocol v2`。

### 启动新预实验

满足冻结来源、版本和工程/研究准入检查后，手动启动一个新的运行：

```bash
python wheelleg_ppo/tools/train_yaw.py \
  --protocol v2 --method M3 --seed 1609 --train \
  --output wheelleg_ppo/tools/results/local_pilot_M3_seed1609
```

该命令会真正开始长训练。示例仅说明入口；同一机器已有 8 环境训练时，应避免再并行启动竞争资源的训练或吞吐基准。脚本拒绝覆盖输出目录，也会拒绝未注册种子、方法、版本变化或失效的冻结证据。

如需自主修改奖励、控制器或依赖，应另建实验协议和独立输出，保留原结果及哈希；不要把新配置重标为旧冻结实验。

### 检查点和恢复

| 文件 | 含义 |
| --- | --- |
| `run_config.json` | 方法、种子、配置与来源哈希 |
| `step_N.zip` | PPO 策略与优化器检查点 |
| `step_N.pkl` | 配套 VecNormalize 状态 |
| `step_N.json` | 该检查点完成后的场景指标 |
| `selection.json` | 检查点汇总与按固定规则选出的最优项 |
| `completed.json` | 整次运行结束后的验证记录 |

模型文件会先于评估 JSON 写入，因此仅看到 `.zip` 不代表检查点评估完成。推理或评估必须使用配套归一化状态。

恢复示例中的 `step_20000` 必须是实际存在、匹配 v2 配置的检查点前缀，并有配套 `.zip`、`.pkl` 和运行配置：

```bash
python wheelleg_ppo/tools/resume_yaw.py \
  --checkpoint wheelleg_ppo/tools/results/local_pilot_M3_seed1609/step_20000 \
  --output wheelleg_ppo/tools/results/local_pilot_M3_seed1609_resume1 \
  --train
```

恢复保留模型、优化器、归一化与累计步数，并在原预算内继续；未保存全部仿真内部状态和随机数流，恢复时会重置仿真。这是分段训练，不能宣称逐值连续复现或把恢复段当作独立种子。

## 验证与结果管理

先跑基础站立，再按修改范围选择检查。仅编辑文档不需要重跑训练。

```bash
# 基础动作，无窗口
python wheelleg_ppo/tools/wheelleg_sim.py --test stop
python wheelleg_ppo/tools/wheelleg_sim.py --test jump
python wheelleg_ppo/tools/wheelleg_sim.py --test jumpfwd
python wheelleg_ppo/tools/wheelleg_sim.py --test jumpback

# 控制与环境接口检查，需要完整 PPO 依赖
python wheelleg_ppo/tools/test_preppo.py
python wheelleg_ppo/tools/test_yaw_training.py
python wheelleg_ppo/tools/test_yaw_gate.py

# 工程短更新检查，使用独立目录
python wheelleg_ppo/tools/prepare_ppo.py training_gate \
  --device cpu --output wheelleg_ppo/tools/results/local_training_gate
```

不同检查回答不同问题：基础动作检查控制功能；短 PPO 更新检查工程链路；固定选择集用于选检查点；门控集用于研究决策；最终保留集用于最终评价。不得用门控或最终集反复调参，也不得将工程通过等同于方法优越或实机安全。

历史基线与哈希应保留。新实验写入新目录，分享结果时同时给出配置、来源哈希、种子、完整失败记录和中断信息。`BASELINE_MANIFEST.json` 描述原始来源，不应被覆盖成当前源码哈希。

Git 已忽略本地虚拟环境、缓存、构建目录、备份和常见凭据文件；研究结果仍有部分纳入版本控制。运行中的训练会持续产生修改，提交时应明确选择文件。

## 常见问题

| 问题 | 检查与处理 |
| --- | --- |
| `ModuleNotFoundError` | 确认已激活 `.venv`，用 `python -m pip` 安装对应子项目依赖 |
| GLFW 初始化失败或无显示器 | 先用 `--test stand` 无窗口验证；交互模式再检查桌面会话与 OpenGL 运行库 |
| 新机器启动脚本路径错误 | 使用本文 Python 入口；根目录 shell 脚本绑定了原开发机路径 |
| `Frozen package version changed` | 对照冻结配置的 `versions`，尤其是 PyTorch `+cu130` 构建后缀 |
| 来源哈希不符 / `Readiness evidence is stale` | 检查代码是否改动及协议是否选对；恢复匹配版本或建立新协议，勿改旧哈希绕过检查 |
| 输出目录已存在 | 换一个新的输出目录；恢复训练使用 `resume_yaw.py` |
| GPU 利用率低 | 物理仿真在 CPU，策略网络较小；按完整采样与更新耗时判断收益 |
| 找不到 ROS2 launch / package | 当前没有 ROS2 包，目录名表示规划方向；先运行 Python 基线 |
| C++ 行为与 Python 不同 | C++ 为历史路径，未同步当前已验收 Python 控制；快速部署无需编译 C++ |

历史 C++ 工程保留 CMake、MuJoCo、GLFW 与 Eigen 依赖配置，可供后续移植参考。本文不把其构建成功作为当前控制主线部署要求，也未对其进行本轮编译验收。

## 进一步阅读与许可

- [论文研究计划与进展](wheelleg_ppo/PAPER_PLAN.md)：研究范围、对照、冻结协议、阶段核验。
- [训练前修复报告](wheelleg_ppo/PREPPO_FIX_REPORT.md)：控制链与环境实现的验证证据。
- [文献矩阵](wheelleg_ppo/LITERATURE_MATRIX.md)：相关研究与差异分析。
- [ROS2 毕设计划](wheelleg_ros2/GRADUATION_PLAN.md)：通信、TF、传感、定位与导航分阶段目标。
- [原始仿真说明](mujoco_wheelleg-main/README.md)：历史 C++ 实现和控制演进背景。
- [参考子模块来源](https://gitee.com/shuo_kai/balance-simulation)：外部参考工程。

各子项目保留原有 [MIT 许可证](wheelleg_ppo/LICENSE) 与版权声明；MuJoCo 示例等文件另有文件级声明。第三方论文、手册、图片及参考子模块遵循各自许可，不应视为自动适用本项目代码许可证。

## GPU 新基线与长期维护

[MuJoCo Warp 基线](wheelleg_warp/README.md) 独立运行GPU物理并与原CPU校对；[长期项目记忆](PROJECT_MEMORY.md) 记录跨对话决策、验收和未完成项。[项目执行约束](AGENTS.md) 要求及时中文提交，GitHub同步守护负责稳定文件快照与远程推送。原 `wheelleg_ppo/` CPU 模型、控制器与历史实验继续保留。
