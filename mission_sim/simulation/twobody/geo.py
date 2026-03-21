# mission_sim/simulation/twobody/geo.py
"""
地球静止轨道 (GEO) L1 级仿真场景
适用于高度约 35786 km 的地球同步轨道任务，考虑 J2 摄动，忽略大气阻力。
"""

import numpy as np

from mission_sim.simulation.twobody.base import TwoBodyBaseSimulation
from mission_sim.core.trajectory.generators import KeplerianGenerator, J2KeplerianGenerator


class GEOL1Simulation(TwoBodyBaseSimulation):
    """
    GEO 轨道 L1 级仿真
    继承自 TwoBodyBaseSimulation，实现 GEO 任务的标称轨道生成、环境初始化及控制律设计。
    """

    # GEO 标准轨道参数
    GEO_RADIUS = 42164000.0          # 地心距 (m)，对应约 35786 km 高度
    GEO_PERIOD_DAYS = 1.0            # 周期约 1 天

    def __init__(self, config: dict):
        """
        初始化 GEO 仿真配置。

        默认配置（可通过 config 覆盖）:
            - mu_earth: 地球引力常数 (m³/s²) -> 3.986004418e14
            - enable_j2: 启用 J2 摄动 -> True
            - enable_atmospheric_drag: 启用大气阻力 -> False (GEO 可忽略)
            - spacecraft_mass: 航天器质量 (kg) -> 2000.0
            - injection_error: 初始状态误差 [Δx,Δy,Δz,Δvx,Δvy,Δvz] -> [0,0,0,0,0,0]
            - elements: 标称轨道根数 [a, e, i, Omega, omega, M0] -> [42164000, 0.0, 0.0, 0.0, 0.0, 0.0] (GEO 圆轨道)
            - use_j2_generator: 是否使用 J2 轨道生成器 -> True
            - dt: 轨道生成步长 (s) -> 60.0
            - sim_time: 轨道生成时长 (s) -> simulation_days * 86400

        地面站及 GNC 参数通过基类配置传入。
        """
        # 设置 GEO 默认配置
        geo_defaults = {
            "mu_earth": 3.986004418e14,
            "enable_j2": True,
            "enable_atmospheric_drag": False,
            "spacecraft_mass": 2000.0,
            "injection_error": np.zeros(6),
            "elements": [self.GEO_RADIUS, 0.0, 0.0, 0.0, 0.0, 0.0],
            "use_j2_generator": True,
            "dt": 60.0,
            "sim_time": None,               # 将在 _generate_nominal_orbit 中计算
        }
        # 合并配置，用户传入的优先
        for key, value in geo_defaults.items():
            if key not in config:
                config[key] = value
        super().__init__(config)

        # 确保大气阻力未启用（GEO 场景忽略）
        self.config["enable_atmospheric_drag"] = False

    def _generate_nominal_orbit(self) -> bool:
        """
        生成 GEO 标称轨道。
        根据配置选择使用 J2 摄动生成器或开普勒解析生成器。
        """
        try:
            # 计算仿真时长（秒）
            sim_seconds = self.config["simulation_days"] * 86400
            dt = self.config.get("dt", 60.0)

            # 获取轨道根数
            elements = self.config.get("elements")
            if elements is None or len(elements) != 6:
                raise ValueError("GEO 场景需要提供 6 个轨道根数 'elements'。")

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
                print(f"✅ GEO 标称轨道生成完成，时长 {sim_seconds/86400:.1f} 天，点数 {len(self.ephemeris.times)}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ GEO 标称轨道生成失败: {e}")
            return False

    def _initialize_physical_domain(self):
        """
        初始化物理域，仅注册 J2 摄动（大气阻力已禁用）。
        """
        # 调用基类初始化（注册 J2，创建航天器）
        super()._initialize_physical_domain()
        # GEO 场景无需额外力模型

    def _design_control_law(self):
        """
        设计 GEO 轨道维持控制律。
        使用基类提供的 J2 线性化 LQR 方法，计算反馈增益矩阵。
        可通过配置 `control_gain_scale` 调整增益强度。
        """
        # 获取轨道高度（从星历表第一个点估算）
        if self.ephemeris is not None:
            r0 = np.linalg.norm(self.ephemeris.states[0, 0:3])
            altitude = r0 - 6378137.0
        else:
            altitude = self.GEO_RADIUS - 6378137.0  # 标准 GEO 高度

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
        使用开普勒解析解生成标准 GEO 圆轨道。
        """
        print("   使用备用轨道生成方案（开普勒圆轨道，标准 GEO）...")
        from mission_sim.core.trajectory.generators import KeplerianGenerator
        a = self.GEO_RADIUS
        e = 0.0
        i = 0.0
        Omega = 0.0
        omega = 0.0
        M0 = 0.0
        elements = [a, e, i, Omega, omega, M0]
        sim_seconds = self.config["simulation_days"] * 86400
        dt = self.config.get("dt", 60.0)
        config = {
            "elements": elements,
            "dt": dt,
            "sim_time": sim_seconds
        }
        gen = KeplerianGenerator(mu=self.mu_earth)
        self.ephemeris = gen.generate(config)
        period = 2 * np.pi * np.sqrt(a**3 / self.mu_earth) / 86400
        print(f"✅ 备用轨道生成完成，周期: {period:.2f} 天")