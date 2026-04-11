# mission_sim/core/trajectory/generators/j2_keplerian.py
"""带 J2 摄动的开普勒轨道生成器（数值积分）"""

import numpy as np
from scipy.integrate import solve_ivp
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.spacetime.generators.base import BaseTrajectoryGenerator
from mission_sim.core.physics.models.j2_gravity import J2Gravity
from mission_sim.utils.math_tools import elements_to_cartesian


class J2KeplerianGenerator(BaseTrajectoryGenerator):
    """
    带 J2 摄动的开普勒轨道生成器。
    通过数值积分二体 + J2 摄动，生成高精度参考星历。
    适用于 LEO/GEO 任务。
    
    注意：J2KeplerianGenerator 不需要高精度星历，因为：
    1. 在地心惯性系中计算航天器相对地球的运动
    2. 地球位置始终为原点 (0,0,0)
    3. J2 模型假设地球为固定中心天体
    """

    def __init__(self, mu: float = 3.986004418e14, ephemeris=None, use_high_precision=False):
        """
        初始化生成器。

        Args:
            mu: 中心天体引力常数 (m³/s²)，默认地球。
            ephemeris: 忽略此参数（J2KeplerianGenerator 不需要星历）
            use_high_precision: 忽略此参数（J2KeplerianGenerator 不支持高精度模式）
            
        注意：J2KeplerianGenerator 不需要高精度星历，因为：
        1. 它在地心惯性系中计算相对地球的轨道
        2. 地球位置始终为原点，不需要从星历获取
        3. 这是简化摄动模型，适用于 LEO/GEO 任务规划
        """
        # 忽略ephemeris和use_high_precision参数
        super().__init__(ephemeris=None, use_high_precision=False)
        self.mu = mu
        self.j2_model = J2Gravity(mu_earth=mu)  # 使用 J2 模型

    def generate(self, config: dict) -> Ephemeris:
        """
        根据轨道根数生成 J2 摄动轨道。

        config 必须包含:
            - elements: [a, e, i, Omega, omega, M0] 轨道根数
            - dt: 输出步长 (s)
            - sim_time: 仿真时长 (s)
            - integrator: 积分器方法 (可选，默认 'DOP853')
            - rtol: 相对容差 (可选，默认 1e-12)

        Returns:
            Ephemeris 对象（J2000_ECI 坐标系）
        """
        elements = config.get("elements")
        if elements is None or len(elements) != 6:
            raise ValueError("J2KeplerianGenerator 必须在 config 中提供 6 个轨道根数 'elements'。")

        dt = config.get("dt", 1.0)
        sim_time = config.get("sim_time", 86400.0)
        integrator = config.get("integrator", 'DOP853')
        rtol = config.get("rtol", 1e-12)

        # 将轨道根数转换为笛卡尔状态
        state0 = self._elements_to_cartesian(elements)

        # 定义运动方程
        def dynamics(t, state):
            pos = state[:3]
            vel = state[3:6]
            r = np.linalg.norm(pos)
            # 中心引力
            acc_central = -self.mu * pos / r**3
            # J2 摄动
            acc_j2 = self.j2_model.compute_accel(state, t)
            return np.concatenate([vel, acc_central + acc_j2])

        # 积分
        times = np.arange(0, sim_time + dt, dt)
        sol = solve_ivp(
            dynamics,
            t_span=(0, sim_time),
            y0=state0,
            t_eval=times,
            method=integrator,
            rtol=rtol,
            atol=rtol
        )
        if not sol.success:
            raise RuntimeError(f"J2 轨道积分失败: {sol.message}")

        return Ephemeris(sol.t, sol.y.T, CoordinateFrame.J2000_ECI)

    def _elements_to_cartesian(self, elements):
        """将轨道根数转换为 J2000_ECI 笛卡尔状态"""
        a, e, i, Omega, omega, M0 = elements
        return elements_to_cartesian(self.mu, a, e, i, Omega, omega, M0)
"""
J2摄动轨道生成器

基于二体问题加上J2摄动项生成轨道。
使用数值积分来近似J2摄动。

遵循MCPC编码标准：使用UTF-8编码，英文输出和注释
"""

import numpy as np
from typing import Dict, Any
from scipy.integrate import solve_ivp

from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.generators.keplerian import KeplerianGenerator


class J2KeplerianGenerator(KeplerianGenerator):
    """
    J2-perturbed orbit generator
    
    Generates orbits with Earth's J2 perturbation using numerical integration.
    Extends the KeplerianGenerator to include J2 effects.
    """
    
    def __init__(self, ephemeris=None, use_high_precision=False):
        """Initialize generator"""
        super().__init__(ephemeris, use_high_precision)
        self.mu_earth = 3.986004418e14  # Earth gravitational parameter (m³/s²)
        self.j2_earth = 1.08262668e-3   # Earth J2 coefficient
        self.r_earth = 6378137.0        # Earth radius (m)
        
    def generate(self, config: Dict[str, Any]) -> Ephemeris:
        """
        Generate J2-perturbed orbit based on orbital elements.
        
        Args:
            config: Configuration dictionary, must contain:
                - elements: Orbital elements list [a, e, i, Ω, ω, M0]
                - dt: Output time step (s)
                - sim_time: Simulation duration (s)
                Optional:
                - epoch: Epoch time (s)
                - mu: Gravitational parameter (m³/s²)
                - j2_coefficient: J2 coefficient
                - earth_radius: Earth radius (m)
                
        Returns:
            Ephemeris: Generated orbit ephemeris with J2 perturbations
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Validate required parameters
        required_keys = ['elements', 'dt', 'sim_time']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required parameter: '{key}'")
        
        # Extract parameters
        elements = config['elements']
        dt = config['dt']
        sim_time = config['sim_time']
        epoch = config.get('epoch', 0.0)
        mu = config.get('mu', self.mu_earth)
        j2_coefficient = config.get('j2_coefficient', self.j2_earth)
        earth_radius = config.get('earth_radius', self.r_earth)
        
        # Validate orbital elements
        if len(elements) != 6:
            raise ValueError(
                f"Orbital elements must have exactly 6 values, got {len(elements)}"
            )
        
        a, e, i, Omega, omega, M0 = elements
        
        # Validate element values
        if a <= 0:
            raise ValueError(f"Semi-major axis must be positive, got a={a:.6e} m")
        
        if e < 0 or e >= 1:
            raise ValueError(
                f"Eccentricity must be 0 ≤ e < 1 for elliptical orbits, got e={e:.6f}"
            )
        
        # Validate time parameters
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got dt={dt:.6e} s")
        
        if sim_time < 0:
            raise ValueError(f"Simulation time must be non-negative, got sim_time={sim_time:.6e} s")
        
        # Generate initial state from Keplerian elements
        state0 = self.elements_to_cartesian_scalar(a, e, i, Omega, omega, M0, mu)
        
        # Define dynamics with J2 perturbation
        def j2_perturbed_dynamics(t, state):
            """Dynamics equations: two-body problem + J2 perturbation"""
            r = state[0:3]
            v = state[3:6]
            r_norm = np.linalg.norm(r)
            
            # Two-body acceleration
            a_two_body = -mu * r / (r_norm**3)
            
            # J2 perturbation acceleration
            a_j2 = self._j2_acceleration(r, mu, j2_coefficient, earth_radius)
            
            # Total acceleration
            a_total = a_two_body + a_j2
            
            return np.concatenate([v, a_total])
        
        # Generate time points
        num_points = int(sim_time / dt) + 1
        times = np.linspace(0, sim_time, num_points) + epoch
        t_eval = times - epoch  # Relative time for integration
        
        # Integrate the orbit with J2 perturbations using high-precision integrator
        try:
            sol = solve_ivp(
                j2_perturbed_dynamics,
                [0, sim_time],
                state0,
                t_eval=t_eval,
                method='DOP853',
                rtol=1e-12,
                atol=1e-12,
                max_step=dt*10  # Limit maximum step size
            )
            
            if not sol.success:
                # If integration fails, fall back to Keplerian orbit
                print(f"[J2KeplerianGenerator] Warning: J2-perturbed orbit integration failed, "
                      f"falling back to Keplerian orbit. Error: {sol.message}")
                return super().generate(config)
            
            # Ensure state array shape is correct
            states = sol.y.T  # Convert to (N, 6)
            if states.shape[0] != len(times):
                # If number of integration points doesn't match, interpolate to specified times
                from scipy.interpolate import interp1d
                states_interp = np.zeros((len(times), 6))
                for i in range(6):
                    interp_func = interp1d(sol.t, sol.y[i], kind='cubic', fill_value='extrapolate')
                    states_interp[:, i] = interp_func(t_eval)
                states = states_interp
            
            return Ephemeris(times, states, CoordinateFrame.J2000_ECI)
            
        except Exception as e:
            # If any error occurs, fall back to Keplerian orbit
            print(f"[J2KeplerianGenerator] Error during integration: {e}, "
                  f"falling back to Keplerian orbit")
            return super().generate(config)
    
    def _j2_acceleration(self, r: np.ndarray, mu: float, j2: float, r_earth: float) -> np.ndarray:
        """
        Calculate J2 perturbation acceleration.
        
        Args:
            r: Position vector (m) [x, y, z]
            mu: Gravitational parameter (m³/s²)
            j2: J2 coefficient
            r_earth: Earth radius (m)
            
        Returns:
            np.ndarray: J2 acceleration vector (m/s²) [ax, ay, az]
        """
        x, y, z = r
        r_norm = np.linalg.norm(r)
        
        if r_norm < 1e-6:
            return np.zeros(3)
        
        # Precompute common terms
        r2 = r_norm * r_norm
        r5 = r2 * r2 * r_norm
        factor = 1.5 * j2 * mu * r_earth * r_earth / r5
        
        # J2 acceleration components
        ax = factor * x * (5 * z * z / r2 - 1)
        ay = factor * y * (5 * z * z / r2 - 1)
        az = factor * z * (5 * z * z / r2 - 3)
        
        return np.array([ax, ay, az])
    
    def elements_to_cartesian_scalar(self, a, e, i, Omega, omega, M, mu):
        """
        Scalar version of orbital elements to Cartesian conversion.
        
        Args:
            a: Semi-major axis (m)
            e: Eccentricity
            i: Inclination (rad)
            Omega: Right ascension of ascending node (rad)
            omega: Argument of perigee (rad)
            M: Mean anomaly (rad)
            mu: Gravitational parameter (m³/s²)
            
        Returns:
            np.ndarray: Cartesian state vector (6,)
        """
        # Use parent class method
        M_array = np.array([M])
        states = self.elements_to_cartesian_batch(a, e, i, Omega, omega, M_array, mu)
        return states[0]


# Convenience function
def create_j2_keplerian_generator(**kwargs) -> J2KeplerianGenerator:
    """Factory function to create J2-perturbed orbit generator"""
    return J2KeplerianGenerator(**kwargs)
