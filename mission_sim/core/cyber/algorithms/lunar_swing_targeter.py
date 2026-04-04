"""
地月共振摆动轨道设计器 - 核心算法实现

实现 LunarSwingTargeter 类，提供共振轨道搜索、状态转移矩阵计算和稳定性分析功能。
"""
import numpy as np
from typing import Tuple, Dict, Callable, Union, Optional
import numpy.linalg as la


class LunarSwingTargeter:
    """地月共振摆动轨道设计器"""

    def __init__(self,
                 dynamics_model: Union[Callable, object],
                 mu: float = 0.01215,  # 地月系统质量参数
                 integrator_type: str = 'rk4',
                 options: dict = None):
        """
        初始化轨道设计器。

        Args:
            dynamics_model: 动力学模型，应支持 f(t, state) -> state_derivative
            mu: 地月系统质量参数 (m2/(m1+m2))
            integrator_type: 积分器类型 ('rk4', 'rkf78', 'dop853')
            options: 配置选项
        """
        self.dynamics = dynamics_model
        self.mu = mu
        self.integrator_type = integrator_type
        self.options = options or {}

    def find_resonant_orbit(self,
                           resonance_ratio: Tuple[int, int],
                           initial_guess: np.ndarray,
                           target_period: float = None,
                           tol: float = 1e-8,
                           max_iter: int = 50) -> Dict:
        """
        使用打靶法搜索共振周期轨道。

        Args:
            resonance_ratio: (n, m) 共振比，如 (2, 1) 表示 2:1 共振
            initial_guess: 6 维初始状态猜测 [x, y, z, vx, vy, vz]（无量纲）
            target_period: 目标周期（秒），None 则根据共振比自动计算
            tol: 收敛容差
            max_iter: 最大迭代次数

        Returns:
            字典包含：'state'（周期轨道状态）, 'period', 'convergence_history', 'success'
        """
        n, m = resonance_ratio

        # 计算目标周期（地月旋转系）
        if target_period is None:
            # 月球轨道周期约 27.321661 天
            T_moon = 27.321661 * 24 * 3600  # 秒
            target_period = (m / n) * T_moon

        print(f"搜索 {n}:{m} 共振轨道，目标周期: {target_period/86400:.3f} 天")

        # 简化的打靶法实现
        x = initial_guess.copy()
        history = []

        for i in range(max_iter):
            # 积分一个周期
            x_final = self._propagate(x, target_period)

            # 计算残差
            residual = x_final - x
            res_norm = la.norm(residual)

            # 记录收敛历史
            history.append({
                'iteration': i,
                'residual_norm': res_norm,
                'state': x.copy()
            })

            # 检查收敛
            if res_norm < tol:
                print(f"在第 {i+1} 次迭代收敛，残差范数: {res_norm:.2e}")
                return {
                    'state': x,
                    'period': target_period,
                    'convergence_history': history,
                    'success': True
                }

            # 简化的修正（实际实现需要 STM）
            # TODO: 使用状态转移矩阵进行牛顿修正
            # 临时使用梯度下降法
            x = x - 0.1 * residual

            if i % 10 == 0:
                print(f"迭代 {i+1}, 残差范数: {res_norm:.2e}")

        print(f"在 {max_iter} 次迭代后未收敛，最终残差范数: {history[-1]['residual_norm']:.2e}")
        return {
            'state': x,
            'period': target_period,
            'convergence_history': history,
            'success': False
        }

    def _propagate(self, state: np.ndarray, duration: float) -> np.ndarray:
        """简化的积分器（使用固定步长 RK4）"""
        # 使用固定步长 RK4
        num_steps = 100
        dt = duration / num_steps
        x = state.copy()

        for _ in range(num_steps):
            x = self._rk4_step(x, dt)

        return x

    def _rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """经典四阶 Runge-Kutta 积分一步"""
        k1 = self._dynamics_func(state)
        k2 = self._dynamics_func(state + 0.5 * dt * k1)
        k3 = self._dynamics_func(state + 0.5 * dt * k2)
        k4 = self._dynamics_func(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _dynamics_func(self, state: np.ndarray) -> np.ndarray:
        """调用动力学模型，返回状态导数"""
        # 如果 dynamics 是可调用函数，直接调用
        if callable(self.dynamics):
            return self.dynamics(0.0, state)  # 暂时忽略时间
        else:
            # 否则，假定它是一个有 compute_derivative 方法的对象
            return self.dynamics.compute_derivative(state)

    def _simple_crtbp_derivative(self, state: np.ndarray) -> np.ndarray:
        """简化的 CRTBP 动力学导数（无量纲）"""
        x, y, z, vx, vy, vz = state
        mu = self.mu

        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x + mu - 1)**2 + y**2 + z**2)

        # 避免除以零
        eps = 1e-15
        r1 = max(r1, eps)
        r2 = max(r2, eps)

        ax = 2*vy + x - (1-mu)*(x+mu)/(r1**3) - mu*(x+mu-1)/(r2**3)
        ay = -2*vx + y - (1-mu)*y/(r1**3) - mu*y/(r2**3)
        az = -(1-mu)*z/(r1**3) - mu*z/(r2**3)

        return np.array([vx, vy, vz, ax, ay, az])

    def compute_stm(self,
                   initial_state: np.ndarray,
                   duration: float) -> np.ndarray:
        """
        计算状态转移矩阵（简化版本，返回单位矩阵）。

        Args:
            initial_state: 初始状态
            duration: 积分时长（秒）

        Returns:
            6x6 状态转移矩阵
        """
        # 临时返回单位矩阵（待实现）
        # TODO: 实现变分方程积分
        print("警告: STM 计算尚未实现，返回单位矩阵")
        return np.eye(6)

    def analyze_stability(self,
                         orbit_state: np.ndarray,
                         period: float) -> Dict:
        """
        分析轨道稳定性（计算单值矩阵特征值）。

        Returns:
            包含特征值、稳定性指标等信息的字典
        """
        stm = self.compute_stm(orbit_state, period)
        eigenvalues = la.eigvals(stm)

        # 稳定性判据：所有特征值模长 <= 1
        max_mag = np.max(np.abs(eigenvalues))
        is_stable = max_mag <= 1.0 + 1e-6

        return {
            'eigenvalues': eigenvalues,
            'max_magnitude': max_mag,
            'stable': is_stable,
            'monodromy_matrix': stm
        }

    def __repr__(self) -> str:
        return (f"LunarSwingTargeter(mu={self.mu}, "
                f"integrator_type='{self.integrator_type}')")
