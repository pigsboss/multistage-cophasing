# Multi-stage Co-Phasing Control (MCPC)

本项目是一个专门为 **“多级共相位控制”** 任务打造的工业级航天器动力学与控制仿真基准框架。MCPC 旨在解决分布式航天器阵列在深空环境下的高精度协同问题，通过从轨道级（公里级）到波长级（纳米级）的多级嵌套控制，支撑空间分布式合成孔径干涉成像任务（如“觅音计划”等）。

当前状态：**v1.1 (Level 1: 基于真实物理量纲的 LQR 绝对轨道维持已交付)**

## 🎯 项目愿景与多级控制目标

本项目采用敏捷开发模式，随 Level 演进逐步覆盖多级控制维度：
* **轨道级 (Orbital Stage - L1 已实现)**: 解决基于 CRTBP 模型的平动点绝对轨道维持。已实现抗科氏力耦合的 LQR 最优控制。
* **编队级 (Formation Stage - L2~L3 进行中)**: 解决多星相对运动学协同与厘米/毫米级编队重构，引入 LVLH 相对运动坐标系。
* **相位级 (Phasing Stage - L4~L5 愿景)**: 解决亚微米级光学延迟线补偿与刚体姿轨耦合波前控制。

## ✨ 核心架构特性

* **强坐标系契约 (Strict Coordinate Contract)**: 引入全局枚举 `CoordinateFrame` 和 `Telecommand` 数据类。任何跨模块的数据流（引力、推力、遥测、指令）均强制进行坐标系标签核对，从底层规避参考系混淆导致的计算灾难。
* **物理与信息域解耦 (Domain Decoupling)**: 
    * **物理域 (Physical Domain)**: 仿真真实的客观宇宙。`Spacecraft` 仅作为受力积分的容器，绝对不直接读取算法输出。
    * **信息域 (Information Domain)**: `GNC_Subsystem` 模拟星载 OBC 处理逻辑，独立进行导航滤波与控制律计算，通过发送带契约的指令影响物理域。
* **真实物理量纲的最优控制 (Real-dimension Optimal Control)**: 摒弃了传统的无量纲化 CRTBP 模型，在 LQR 最优控制中引入真实的日地角速度（$\omega \approx 1.99 \times 10^{-7} \text{ rad/s}$）与引力梯度，彻底解决了离散化仿真中的高频振荡问题，实现极低燃料消耗下的平滑收敛。
* **高性能数据流转**: 基于 `h5py` 实现增量式 HDF5 记录器 (`HDF5Logger`) 防 OOM，并将计算与可视化渲染完全分离。

## 📂 目录结构与模块映射

```text
mission_sim/
├── core/                        # 核心领域模型
│   ├── __init__.py
│   ├── physics/                 # 【物理域】
│   │   ├── __init__.py
│   │   ├── spacecraft.py        # 航天器质点模型
│   │   ├── environment.py       # 力学注册表 (Environment)
│   │   └── models/              # 具体力学实现子包
│   │       ├── __init__.py
│   │       ├── gravity_crtbp.py # 三体引力模型
│   │       └── srp_cannonball.py# 太阳光压模型
│   ├── gnc/                     # 【信息域】
│   │   ├── __init__.py
│   │   ├── gnc_subsystem.py     # 动态星历追踪控制器
│   │   └── ground_station.py    # 测控站模拟
│   ├── trajectory/              # 【预处理】
│   │   ├── __init__.py
│   │   ├── ephemeris.py         # 标称星历数据契约
│   │   ├── generators.py        # 轨道生成器基类
│   │   └── halo_corrector.py    # 微分修正算法
│   └── types.py                 # 全局类型定义 (CoordinateFrame 等)
├── utils/                       # 基础设施
│   ├── __init__.py
│   ├── loggers_hdf5.py          # 高效记录器
│   ├── math_tools.py            # 矩阵与 LQR 工具
│   └── visualizer_L1.py         # 【L1级专用】可视化工具
├── configs/                     # 配置文件
│   └── L1_halo_mission.yaml     # 【L1级】任务配置
├── data/                        # 仿真输出
├── tests/                       # 测试脚本
├── main_L1_runner.py            # 【L1级】仿真编排器入口
└── requirements.txt
```