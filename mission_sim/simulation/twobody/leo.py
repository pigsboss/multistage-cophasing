# mission_sim/simulation/twobody/leo.py
"""
低地球轨道 (LEO) L1 级仿真场景
适用于高度 200-2000km 的近地轨道任务，考虑 J2 摄动和大气阻力。
"""

import numpy as np
from typing import Optional

from mission_sim.simulation.twobody.base import TwoBodyBaseSimulation
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.trajectory.generators import KeplerianGenerator, J2KeplerianGenerator
from mission_sim.core.physics.models.atmospheric_drag import AtmosphericDrag


class LEOL1Simulation(TwoBodyBaseSimulation):
    """
    LEO 轨道 L1 级仿真
    继承自 TwoBodyBaseSimulation，实现 LEO 任务的标称轨道生成、环境初始化及控制律设计。
    """

    def __init__(self, config: dict):
        """
        初始化 LEO 仿真配置。

        默认配置（可通过 config 覆盖）:
            - mu_earth: 地球引力常数 (m³/s²) -> 3.986004418e14
            - enable_j2: 启用 J2 摄动 -> True
            - enable_atmospheric_drag: 启用大气阻力 -> True
            - area_to_mass: 面积质量比 (m²/kg) -> 0.02 (典型值)
            - Cd: 阻力系数 -> 2.2
            - rho0: 参考密度 (kg/m³) -> 1.225
            - H: 标高 (m) -> 8500.0
            - h0: 参考高度 (m) -> 0.0
            - spacecraft_mass: 航天器质量 (kg) -> 1000.0
            - injection_error: 初始状态误差 [Δx,Δy,Δz,Δvx,Δvy,Δvz] -> [0,0,0,0,0,0]
            - elements: 标称轨道根数 [a, e, i, Omega, omega, M0] -> [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0] (半径 7000km 圆轨道)
            - use_j2_generator: 是否使用 J2 轨道生成器 -> True
            - dt: 轨道生成步长 (s) -> 10.0
            - sim_time: 轨道生成时长 (s) -> simulation_days * 86400

        地面站及 GNC 参数通过基类配置传入。
        """
        # 设置 LEO 默认配置
        leo_defaults = {
            "mu_earth": 3.986004418e14,
            "enable_j2": True,
            "enable_atmospheric_drag": True,
            "area_to_mass": 0.02,          # 典型值，可根据需要调整
            "Cd": 2.2,
            "rho0": 1.225,                  # 海平面密度
            "H": 8500.0,                   # 标高
            "h0": 0.0,                     # 参考高度
            "spacecraft_mass": 1000.0,
            "injection_error": np.zeros(6),
            "elements": [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],  # 半径 7000km 圆轨道
            "use_j2_generator": True,
            "dt": 10.0,
            "sim_time": None,               # 将在 _generate_nominal_orbit 中计算
        }
        # 合并配置，用户传入的优先
        for key, value in leo_defaults.items():
            if key not in config:
                config[key] = value
        super().__init__(config)

        # 提取 LEO 特有参数
        self.enable_atmospheric_drag = self.config.get("enable_atmospheric_drag", True)
        self.area_to_mass = self.config.get("area_to_mass", 0.02)
        self.Cd = self.config.get("Cd", 2.2)
        self.rho0 = self.config.get("rho0", 1.225)
        self.H = self.config.get("H", 8500.0)
        self.h0 = self.config.get("h0", 0.0)

    def _generate_nominal_orbit(self) -> bool:
        """
        生成 LEO 标称轨道。
        根据配置选择使用 J2 摄动生成器或开普勒解析生成器。
        """
        try:
            # 计算仿真时长（秒）
            sim_seconds = self.config["simulation_days"] * 86400
            dt = self.config.get("dt", 10.0)

            # 获取轨道根数
            elements = self.config.get("elements")
            if elements is None or len(elements) != 6:
                raise ValueError("LEO 场景需要提供 6 个轨道根数 'elements'。")

            use_j2 = self.config.get("use_j2_generator", True)
            if use_j2:
                # 使用带 J2 摄动的数值积分生成器
                generator = J2KeplerianGenerator(mu=self.mu_earth)
                gen_config = {
                    "elements": elements,
                    "dt": dt,
                    "sim_time": sim_seconds,
                    "integrator": "DOP853",
                    "rtol": 1e-12
                }
                self.ephemeris = generator.generate(gen_config)
            else:
                # 使用开普勒解析解生成器（无摄动）
                generator = KeplerianGenerator(mu=self.mu_earth)
                gen_config = {
                    "elements": elements,
                    "dt": dt,
                    "sim_time": sim_seconds
                }
                self.ephemeris = generator.generate(gen_config)

            if self.verbose:
                print(f"✅ LEO 标称轨道生成完成，时长 {sim_seconds/86400:.1f} 天，点数 {len(self.ephemeris.times)}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ LEO 标称轨道生成失败: {e}")
            return False

    def _initialize_physical_domain(self):
        """
        初始化物理域，在基类基础上添加大气阻力模型。
        """
        # 调用基类初始化（注册 J2，创建航天器）
        super()._initialize_physical_domain()

        # 注册大气阻力模型（若启用）
        if self.enable_atmospheric_drag:
            drag_model = AtmosphericDrag(
                area_to_mass=self.area_to_mass,
                Cd=self.Cd,
                rho0=self.rho0,
                H=self.H,
                h0=self.h0,
                R_earth=6378137.0  # 地球半径（WGS84）
            )
            self.environment.register_force(drag_model)
            if self.verbose:
                print(f"✅ 已注册大气阻力模型 (A/m={self.area_to_mass:.4f} m²/kg, Cd={self.Cd:.2f})")

    def _design_control_law(self):
        """
        设计 LEO 轨道维持控制律。
        使用基类提供的 J2 线性化 LQR 方法，计算反馈增益矩阵。
        也可通过配置 `control_gain_scale` 调整增益强度。
        """
        # 获取轨道高度（从星历表第一个点估算）
        if self.ephemeris is not None:
            r0 = np.linalg.norm(self.ephemeris.states[0, 0:3])
            altitude = r0 - 6378137.0
        else:
            altitude = 7000e3 - 6378137.0  # 默认约 622km

        # 计算基础 LQR 增益
        K_base = self._compute_j2_lqr_gain(altitude)

        # 允许用户缩放增益（通过配置）
        gain_scale = self.config.get("control_gain_scale", 1.0)
        self.k_matrix = gain_scale * K_base

        if self.verbose:
            print(f"✅ LQR 控制律设计完成，增益缩放因子: {gain_scale:.2f}")

    def _generate_fallback_orbit(self):
        """
        备用轨道生成（当主生成失败时调用）。
        使用开普勒解析解生成圆轨道，高度 500km。
        """
        print("   使用备用轨道生成方案（开普勒圆轨道，高度 500km）...")
        from mission_sim.core.trajectory.generators import KeplerianGenerator
        a = 6878137.0  # 半径 500km 轨道
        e = 0.0
        i = 0.0
        Omega = 0.0
        omega = 0.0
        M0 = 0.0
        elements = [a, e, i, Omega, omega, M0]
        sim_seconds = self.config["simulation_days"] * 86400
        dt = self.config.get("dt", 10.0)
        config = {
            "elements": elements,
            "dt": dt,
            "sim_time": sim_seconds
        }
        gen = KeplerianGenerator(mu=self.mu_earth)
        self.ephemeris = gen.generate(config)
        print(f"✅ 备用轨道生成完成，周期: {2 * np.pi * np.sqrt(a**3 / self.mu_earth) / 86400:.2f} 天")