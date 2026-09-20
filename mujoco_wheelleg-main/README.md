# MuJoCo 轮腿仿真代码
## 简介
当前 `xml/wheelleg.xml` 为每腿双主动关节的闭环五连杆构型；早期上交串联腿介绍不再适用于当前模型。C++ 窗口渲染和键鼠逻辑使用官方例程，打开后运行时显示为 simulate 窗口（包含各种按钮、数值显示、鼠标施加力矩、键盘快捷键等）。当前交互入口与回归命令见下文。
## 运行&调试
此代码在 MacOS Silicon 下编写，使用 CMake 4.2.0，编译器为 Clang 17.0.0 arm64-apple-darwin24.6.0。此代码未在其他平台进行过测试。<br>
在编译前，请确保你的电脑安装有 glfw 与 Eigen。<br>
编译通过后，运行 ./build/bin/simulate wheelleg.xml 即可。<br>
调试目前使用 (lldb) Launch simulate 进行调试。
需要注意的是代码中**出现 Warning 会报错**。请确保代码中无 Warning。
## 代码结构
asset/下为 xml 所需素材，目前只有天空盒图片。<br>
cmake/下为官方提供的.cmake 文件。<br>
examples/下的所有文件复制自 littleg。主要作用为写代码时方便参考。<br>
include/下为所有.h 与.mm 文件。<br>
matlab/为计算 K 矩阵的 matlab 文件与其输出。<br>
/下为所有.cc 文件以及一些杂文件。
## 文件内容介绍
.h:
| alg_lqr.h | alg_vmc.h | crouching_lqr_decl.h | data_names.h | data.h | get_data.h | wheelleg_logic.h | wheelleg.h |
|-|-|-|-|-|-|-|-|
| LQR 定义 | VMC 定义 | 暴露出 LQR 变量 | 暴露出所有数据名 | 数据类型的定义：Sensor（只读不写）Joint（只读不写）Actuator（只写不读） | 收集数据，进行欧拉角和连续角解算 | 腿与轮赋值的函数 | 腿与轮的虚类以及腿长的 PID 定义 |

.cc:
| crouching_lqr_def.cc | data_names.cc | data.cc | get_data.cc | wheelleg.cc |
|-|-|-|-|-|
| K 矩阵定义 | 所有数据名字定义 | 一些基本的数据操作，如使用 emplace_back 创建新数据、打印数据、赋值等操作 | 无 | 腿与轮类的定义、腿长 PID 与 LQR 的运算、电机扭矩赋值 |

未被提及的均为 MuJoCo 官方代码。

wheelleg.xml 为轮腿建模文件，内部只含一个 freejoint，将其注释掉即可令车辆悬空。xml 内写了将腿或车托起来的 platform/stool 便于调试。

## 运行逻辑
在 main.cc 的 main() 函数内构造所有数据，并在 mj_step(m, d); 前调用 runCrouchingControl(); 进行数据获取与赋值。

## 当前轮腿控制器与核心回归

默认入口为 `tools/wheelleg_sim.py`，直接加载主模型 `xml/wheelleg.xml`，默认启用硬件参数与六状态LQR/VMC。当前为300 mm轮距内嵌髋电机布局、160 mm中央箱体、220 mm上平台；无需额外传入 `--hardware --six-state`。C++ `wheelleg.cc` 未同步本轮Python控制，不作为当前已验收控制入口。使用项目虚拟环境运行：

```bash
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/wheelleg_sim.py
```

当前已通过的无理想基座辅助基线：

```bash
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/wheelleg_sim.py --test stand
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/wheelleg_sim.py --test stop
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/wheelleg_sim.py --test jump
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/wheelleg_sim.py --test jumpfwd
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/wheelleg_sim.py --test jumpback
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/test_robustness.py
```

以下扩展压力回归已改用物理力脉冲，当前用于暴露待修复的能力边界：

```bash
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/test_jump_height.py
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/test_jump_stress.py
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/test_jump_stop.py
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/test_step_stress.py
```

当前实施状态和后续顺序见 `TECHNICAL_ROADMAP.md`；历史理想仿真结果见 `DEBUG_REPORT_2026-08-28.md`。

### 180 mm 六状态候选入口

2026-09-08 第十四次续作：25 mm 主方案已复验空中髋阻尼 0.20，并修复 PREP 航向门边徘徊和 VMC 分腿径向阻抗。基础五项、连续跳、±1% 失配六场景、六组落地急停、腿高/取消/提交、转向后跳跃和短时横坡通过；后退跳 pitch 3.16°，后退落地立即松键停车 0.303 m、pitch/roll 3.33°/2.95°。单轮凸台与影子估计仍失败，硬件轮端需重新选型。当前结果以交接第 0 节及 `tools/results/continuation_2026-09-08/resume_split_vmc/` 为准。

2026-09-08 用户确认恢复 **25 mm 轮半径（50 mm 外径）**；180 mm 指左右轮心距。`--hardware` 当前使用该轮径和暂留的历史质量/力矩包络，原 85 mm 轮端电机直驱装配已标为不适用，需重新选型。下文 9 月 6 日及更早的成绩属于 49.2 mm 轮径历史对照，当前结果见交接第 0 节。

当前按 RoboMaster 五连杆控制思路适配的候选入口为：

```bash
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/wheelleg_sim.py \
  --hardware --track-width 0.18 --six-state --test jumpfwd
```

该入口使用暂定动力学包络（4.274 kg）和按当前 25 mm 轮半径重新求解的 **2×6 LQR**；六个状态为腿角、腿角速度、位移、速度、平台 pitch、pitch 角速度。腿长/腿速与 roll 由 VMC 支撑环处理，行驶、急停、SQUAT、JUMP、FLY、LAND 的平台参考保持水平。180 mm 只移动两侧完整闭环链，新增横梁的 CAD 质量与惯量仍待实测。

已完成的名义仿真矩阵、失败边界和恢复命令见 `WORK_HANDOFF_2026-08-31.md` 当前第 0 节；计划评估见 `PLAN_REVIEW_2026-09-05.md`。候选入口尚未替换默认控制，也不能作为实机验收结论。

2026-09-06 的 49.2 mm 候选结果：纯航向差矩与六状态共模分开合成，左右腿分别执行 VMC 映射；±1% 失配和影子估计曾通过名义暂定门。该结果属于历史轮径对照。要打开 180 mm 候选交互窗口，使用上面同一命令并移除 `--test jumpfwd`；窗口支持 `--size 1280x720` 或 `--fullscreen`。

第十二次续作把纵向速度/停车位移改为当前车头方向的投影，并修正候选航向制动力矩上限，解决大转向后的错误加速和翻倒。转向、短时横坡及单轮凸台入口见交接第 0 节；单轮凸台航向仍失败，横坡长期侧滑尚未验收，不能称复杂地形已完成。

2026-09-08 起，`--hardware` 主方案恢复 25 mm 轮半径（50 mm 外径）；49.2 mm 仅为历史对照。25 mm 下重新计算轮惯量并在编译前生成碰撞几何，基础动作结果和失败边界见交接第 0 节。原 RI85 85 mm 轮端电机直驱装配不适用，需重新选型。

## 本机 LQR 建模与独立验证

`tools/model_lqr.py` 从现用 MJCF 求平衡点、线性化和离散 LQR，并验证腿长插值与物理脉冲恢复。依赖见上级 `requirements.txt`（新增 SciPy 1.15.3）：

```bash
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/model_lqr.py --heights 0.055 0.09 0.1119 0.135 --check-midpoints
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/model_lqr.py --hardware --track-width 0.18 --heights 0.055 0.09 0.1119 0.135 --check-midpoints
```

这是保留被动关节/接触动态的 15 状态对照模型，使用仿真完整状态，尚未替换正式行驶与跳跃控制。180 mm 轮距仅用于名义结构敏感性验证。当前结果与边界见交接第 0 节。

## 硬件候选模式

RI85-PH/T81 名义选型、质量和电源核算见 `HARDWARE_SELECTION_2026-09-02.md`。该模式独立于默认基线：

```bash
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/check_hardware_sizing.py
/home/wmt/wheel_leg_robot_work/.venv/bin/python3 tools/wheelleg_sim.py --hardware --test stand
```

厂家标称值不是样机实测值；摩擦、传动效率、热限制和质心仍须到货后校准。
