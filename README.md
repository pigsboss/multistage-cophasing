# Multi-stage Co-Phasing Control (MCPC)

本项目是一个专门为 **“多级共相位控制”** 任务打造的工业级航天器动力学与控制仿真基准框架。MCPC 旨在解决分布式航天器阵列在深空环境下的高精度协同问题，通过从轨道级（公里级）到波长级（纳米级）的多级嵌套控制，支撑空间分布式合成孔径干涉成像任务。

当前版本：**v1.0 (Level 1: 绝对轨道维持已交付)**

## 🎯 项目愿景与多级控制目标

本项目采用敏捷开发模式，随 Level 演进逐步覆盖多级控制维度：
* **轨道级 (Orbital Stage)**: 解决基于 CRTBP 模型的平动点绝对轨道维持（L1 已实现）。
* **编队级 (Formation Stage)**: 解决多星相对运动学协同与厘米/毫米级编队重构（L2-L3 目标）。
* **相位级 (Phasing Stage)**: 解决亚微米级光学延迟线补偿与波前控制（L4-L5 愿景）。

## ✨ 核心架构特性

* **强坐标系契约 (Strict Coordinate Contract)**: 引入全局枚举 `CoordinateFrame`。任何跨模块的数据流（引力、推力、遥测、指令）均强制进行坐标系标签核对，从底层规避参考系混淆导致的计算灾难。
* **物理与信息域解耦**: 
    * **物理域 (Physical Domain)**: `Spacecraft` 仅作为受力积分的质点/刚体容器。
    * **信息域 (Information Domain)**: `GNC_Subsystem` 模拟星载 OBC 处理逻辑，独立进行导航滤波与控制律计算。
* **高性能数据持久化**: 基于 `h5py` 实现增量式 HDF5 记录器 (`HDF5Logger`)，支持长时仿真下的防 OOM（内存溢出）落盘。
* **计算与后处理分离**: 仿真主循环仅负责生产数据，通过 `L1Visualizer` 离线读取 `.h5` 文件生成 3D 轨迹动画及控制分析图表。

## 📂 目录结构与模块映射

```text
mission_sim/
├── docs/                    # 架构演进文档 (PlantUML)
│   ├── architecture_global.puml # 全局总体静态架构
│   └── architecture_L1.puml     # Level 1 实现细节图
├── core/                    # 核心领域层 (各 Level 演进主战场)
│   ├── types.py             # 全局契约基石 (坐标系定义)
│   ├── spacecraft.py        # 物理域：航天器本体模型
│   ├── environment.py       # 物理域：CRTBP 引力环境
│   ├── ground_station.py    # 信息域：地面测控网模拟
│   └── gnc_subsystem.py     # 信息域：星载制导导航与控制
├── utils/                   # 基础设施层 (跨级复用工具)
│   ├── loggers.py           # 高性能 HDF5 增量记录器
│   └── visualizer.py        # 离线数据渲染与 3D 动画引擎
├── main.py                  # 仿真场景入口 (Level 1 场景)
└── requirements.txt         # 依赖清单

