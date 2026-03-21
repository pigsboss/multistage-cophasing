# Multi-stage Co-Phasing Control (MCPC) Framework

**MCPC** 是一个面向分布式航天器阵列高精度协同任务的工业级航天器动力学与控制仿真基准框架。它支持从轨道级（公里级）到波长级（纳米级）的多级嵌套控制，为空间分布式合成孔径干涉成像（如“觅音计划”）等任务提供数字底座。

当前状态：**v2.0 (L1 级已交付，支持日地 L2 点 Halo 轨道的绝对轨道维持与燃料账单分析)**。

## 🎯 项目愿景与多级控制目标

MCPC 采用渐进式模型演进策略，按工程保真度逐级逼近真实世界：

- **L1 – 基准定标**：宏观轨道质点，主导摄动（CRTBP/J2），理想推力，地基测控，绝对轨道维持。**（已完成）**
- **L2 – 协同效能**：多质点相对运动，姿态锁定假设，理想星间链路，编队构型重构。**（进行中）**
- **L3 – 原理验证**：平台-载荷双层多体模型，简化机构运动学，级间误差分配。
- **L4 – 样机鉴定**：全 6-DOF 刚体姿轨耦合，硬件非线性（死区、延迟、噪声）。
- **L5 – 数字孪生**：刚柔液多体拓扑，时变极端环境，全息数据融合。

## ✨ 核心架构特性

- **动力学分类 + 模型分级**：将任务按力学本质分为**二体类**（中心引力 + 摄动）与**三体类**（CRTBP），每类内部再按 L1-L5 逐级增加保真度，确保代码复用与可扩展性。
- **强坐标系契约**：所有跨模块数据（引力、推力、遥测、指令）均携带 `CoordinateFrame` 标签并强制校验，从源头杜绝参考系混淆。
- **物理与信息域解耦**：
  - **物理域**（`core/physics`）仿真客观宇宙，航天器仅受力积分，不直接读取算法输出。
  - **信息域**（`core/gnc`）模拟星载计算机，独立进行导航滤波与控制律计算，通过带契约的指令影响物理域。
- **真实物理量纲的最优控制**：在 LQR 控制中直接使用日地系统真实角速度（~2×10⁻⁷ rad/s）与引力梯度，避免无量纲化带来的数值问题，实现平滑收敛。
- **高性能数据流转**：基于 `h5py` 的增量式记录器（`HDF5Logger`）支持内存缓冲与压缩，防止内存溢出；可视化与计算完全分离。

## 📂 目录结构

```
mission_sim/
├── core/                          # 核心领域模型
│   ├── dynamics/                  # 基础动力学（二体/三体运动方程）
│   │   ├── twobody/               # 二体基类（占位）
│   │   └── threebody/             # CRTBP 基类、无量纲/物理转换
│   ├── physics/                   # 物理域
│   │   ├── environment.py         # 力学注册表（CelestialEnvironment）
│   │   ├── spacecraft.py          # 航天器质点模型
│   │   └── models/                # 具体力模型插件
│   │       ├── gravity_crtbp.py   # CRTBP 引力（SI 单位）
│   │       ├── j2_gravity.py      # 地球 J2 摄动
│   │       └── srp.py             # 太阳光压（Cannonball 模型）
│   ├── gnc/                       # 信息域
│   │   ├── gnc_subsystem.py       # 动态星历追踪控制器（含外推器）
│   │   ├── ground_station.py      # 测控站（可视弧段、采样率、噪声）
│   │   └── propagator.py          # 盲区外推器（简单线性、二体）
│   ├── trajectory/                # 预处理
│   │   ├── ephemeris.py           # 标称星历（三次样条插值）
│   │   └── generators/            # 多态轨道生成器工厂
│   │       ├── keplerian.py       # 开普勒解析解
│   │       ├── j2_keplerian.py    # J2 摄动数值积分
│   │       └── halo.py            # Halo 轨道（固定初值积分）
│   └── types.py                   # 全局类型（CoordinateFrame, Telecommand）
├── simulation/                    # 仿真控制器（按场景分类）
│   ├── base.py                    # 顶层抽象基类（模板方法）
│   ├── threebody/                 # 三体场景族
│   │   ├── base.py                # 三体场景基类
│   │   └── sun_earth_l2.py        # 日地 L2 点 L1 级仿真
│   └── twobody/                   # 二体场景族（占位）
├── utils/                         # 基础设施
│   ├── logger.py                  # HDF5Logger + SimulationMetadata
│   ├── math_tools.py              # LQR、LVLH 转换
│   ├── differential_correction.py # 微分修正工具（STM、多参数修正）
│   └── visualizer_*.py            # 可视化（L1/L2）
├── tests/                         # 单元测试（导入路径需按新架构修正）
├── docs/                          # 设计文档（UML 图、架构说明）
├── analysis/                      # 应用挖掘脚本（鲁棒性、燃料分析）
├── config/                        # 示例配置文件（待补充）
├── run.py                         # 通用仿真入口
└── requirements.txt
```

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/your-org/mcpc.git
cd mcpc
pip install -r requirements.txt
```

### 2. 运行日地 L2 点 L1 级仿真

使用通用入口 `run.py`，指定场景 `sun_earth_l2` 和层级 `1`：

```bash
python run.py --scene sun_earth_l2 --level 1 --simulation_days 1 --time_step 60
```

或者使用 YAML 配置文件（示例：`config/halo_example.yaml`）：

```yaml
mission_name: "Halo L1 Test"
simulation_days: 1
time_step: 60.0
Az: 0.05
```

运行：

```bash
python run.py --scene sun_earth_l2 --level 1 --config config/halo_example.yaml
```

### 3. 查看输出

仿真结束后，`data/` 目录下会生成：
- `HDF5` 数据文件（包含真值、导航状态、跟踪误差、控制力、累计 ΔV 等）
- `fuel_bill_*.csv` 燃料账单
- 可视化图表（若启用 `--enable_visualization`）

### 4. 自定义配置

所有命令行参数均可覆盖配置文件中的值，例如：

```bash
python run.py --scene sun_earth_l2 --level 1 --simulation_days 30 --time_step 10 --propagator_type kepler
```

## 📖 使用说明

### 场景与层级

- **场景标识**：当前仅支持 `sun_earth_l2`（日地 L2 点）。未来将增加 `leo`、`geo`、`cislunar` 等。
- **层级**：`1`（L1 级），更高层级将随开发进度开放。

### 关键配置项

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `mission_name` | str | 任务名称 | "MCPC Simulation" |
| `simulation_days` | float | 仿真时长（天） | 1 |
| `time_step` | float | 积分步长（秒） | 60.0 |
| `Az` | float | Halo 轨道 Z 振幅（无量纲） | 0.05 |
| `spacecraft_mass` | float | 航天器质量（kg） | 6200.0 |
| `injection_error` | list[6] | 初始状态注入误差（位置 m，速度 m/s） | [2000,2000,-1000,0.01,-0.01,0.005] |
| `propagator_type` | str | 盲区外推器类型（`simple`/`kepler`/`None`） | None |
| `enable_srp` | bool | 是否启用太阳光压 | False |
| `enable_visualization` | bool | 是否生成图表 | True |
| `data_dir` | str | 输出目录 | "data" |

### 输出文件

- **`*.h5`**：HDF5 数据集，包含 `epochs`、`nominal_states`、`true_states`、`nav_states`、`tracking_errors`、`control_forces`、`accumulated_dvs` 等。
- **`fuel_bill_*.csv`**：燃料账单，记录总 ΔV、平均每天 ΔV、最终误差等。
- **`*_trajectory.png`**：3D 轨迹对比图。
- **`*_errors.png`**：位置/速度误差随时间变化曲线。
- **`*_control.png`**：控制力与累计 ΔV 曲线。

## 🛠️ 扩展与定制

### 添加新场景

1. 在 `simulation/` 下创建新子包（如 `twobody/leo.py`）。
2. 继承 `BaseSimulation` 或对应场景基类（如 `ThreeBodyBaseSimulation`）。
3. 实现抽象方法 `_generate_nominal_orbit`、`_initialize_physical_domain`、`_initialize_information_domain`、`_design_control_law`。
4. 在 `run.py` 的 `SCENE_MODULE_MAP` 中注册场景名称与模块路径。

### 添加新层级

在对应场景子包中添加 `L2Simulation`、`L3Simulation` 等类，继承对应基类并重写相关方法（例如在 L2 中添加相对运动）。

### 添加新力模型

在 `core/physics/models/` 中创建新类，实现 `IForceModel` 接口，然后在仿真控制器的 `_initialize_physical_domain` 中通过 `self.environment.register_force()` 注册。

## 📊 应用挖掘示例

项目提供了分析脚本，用于深度挖掘 L1 级仿真能力：

- **Halo 轨道生成能力分析**：`analysis/halo_orbit_analysis.py`
- **控制鲁棒性蒙特卡洛分析**：`analysis/control_robustness_analysis.py`
- **燃料开销分析**：`analysis/fuel_analysis.py`

运行前请确保已安装 `tqdm` 等依赖：

```bash
pip install tqdm
cd analysis
python control_robustness_analysis.py
```

## 🤝 贡献指南

欢迎通过 Issue 和 Pull Request 参与贡献。请确保代码遵循 PEP 8 规范，并为新功能添加单元测试。

## 📄 许可证

Apache License 2.0（详见 LICENSE 文件）。


---

**MCPC – 从轨道到波长，逐级逼近真实。**