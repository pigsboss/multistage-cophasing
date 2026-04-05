"""
地月共振摆动轨道设计器 - 核心算法实现

实现 LunarSwingTargeter 类，提供共振轨道搜索、状态转移矩阵计算和稳定性分析功能。
"""
import numpy as np
from typing import Tuple, Dict, Callable, Union, Optional, List
import numpy.linalg as la

from mission_sim.utils.dynamics.stm_calculator import STMCalculator


class LunarSwingTargeter:
    """地月共振摆动轨道设计器 - 单参数打靶法实现
    
    使用微分修正（Differential Correction）算法搜索周期轨道。
    固定初始位置的 x, y, z 和速度 vx，以 vy, vz 为设计变量，
    通过牛顿迭代使一个周期后的位置残差收敛到零。
    """

    def __init__(self,
                 dynamics_model: Union[Callable, object],
                 mu: float = 0.01215,  # 地月系统质量参数
                 integrator_type: str = 'rk4',
                 num_steps: int = 200,
                 options: dict = None):
        """
        初始化轨道设计器。

        Args:
            dynamics_model: 动力学模型，应支持 f(t, state) -> state_derivative
            mu: 地月系统质量参数 (m2/(m1+m2))
            integrator_type: 积分器类型 ('rk4', 'rkf78')
            num_steps: 积分步数（影响STM计算精度）
            options: 配置选项
        """
        self.dynamics = dynamics_model
        self.mu = mu
        self.integrator_type = integrator_type
        self.num_steps = num_steps
        self.options = options or {}
        
        # 初始化 STM 计算器
        self._stm_calc = STMCalculator()

    def find_resonant_orbit(self,
                           resonance_ratio: Tuple[int, int],
                           initial_guess: np.ndarray,
                           target_period: float = None,
                           tol: float = 1e-6,
                           max_iter: int = 50,
                           damping: float = 0.5,
                           char_period: float = None,
                           adaptive_damping: bool = True) -> Dict:
        """
        使用单参数打靶法搜索共振周期轨道。
        
        增加自适应阻尼和更好的数值稳定性处理。
        
        Args:
            resonance_ratio: (n, m) 共振比，如 (2, 1) 表示 2:1 共振
            initial_guess: 6 维初始状态猜测 [x, y, z, vx, vy, vz]（无量纲）
            target_period: 目标周期（秒），None 则根据共振比自动计算
            tol: 位置残差收敛容差（无量纲单位）
            max_iter: 最大迭代次数
            damping: 初始阻尼因子 (0-1)
            char_period: CRTBP特征周期（秒），用于时间单位转换。
                        默认为4.342*86400（地月系统约4.342天）
            adaptive_damping: 是否使用自适应阻尼，当残差增加时减少阻尼
    
        Returns:
            字典包含：'state'（周期轨道状态）, 'period', 'convergence_history', 
                     'success', 'final_residual'
        """
        n, m = resonance_ratio
        
        # CRTBP特征周期（地月系统约4.342天）
        if char_period is None:
            char_period = 4.342 * 86400  # 秒

        # 计算目标周期（地月旋转系）
        if target_period is None:
            # 月球轨道周期约 27.321661 天
            T_moon = 27.321661 * 24 * 3600  # 秒
            target_period = (m / n) * T_moon

        # Convert to nondimensional time (CRTBP time units)
        target_period_nd = target_period / char_period

        print(f"Searching {n}:{m} resonant orbit, target period: {target_period/86400:.3f} days "
              f"({target_period_nd:.3f} nondim units)")
        print(f"Initial guess: {initial_guess}")

        # 初始状态
        x0 = initial_guess.copy()
        history = []
        
        # 确定设计变量和残差索引
        is_planar = (initial_guess[2] == 0.0 and initial_guess[5] == 0.0)
        if is_planar:
            design_indices = [3, 4]   # vx, vy
            residual_indices = [0, 1] # x, y
            num_design = 2
            num_residual = 2
        else:
            design_indices = [4, 5]   # vy, vz
            residual_indices = [0, 1, 2]  # x, y, z
            num_design = 2
            num_residual = 3

        current_damping = damping
        prev_residual_norm = float('inf')
        
        for i in range(max_iter):
            # 传播状态并计算STM
            x_final, stm = self._stm_calc.propagate_with_stm(
                dynamics=self._get_dynamics_func(),
                initial_state=x0,
                t0=0.0,
                tf=target_period_nd,
                method=self.integrator_type,
                num_steps=self.num_steps
            )
            
            # 数值有效性检查
            if not np.all(np.isfinite(x_final)) or not np.all(np.isfinite(stm)):
                print(f"✗ Numerical overflow at iteration {i+1}, reducing damping and restarting from previous state")
                if i > 0:
                    x0 = history[-1]['state'].copy()
                    current_damping *= 0.5
                    print(f"   Reduced damping to {current_damping:.3f}")
                    continue
                else:
                    return {
                        'state': x0,
                        'period': target_period,
                        'convergence_history': history,
                        'success': False,
                        'final_residual': np.full(6, np.nan),
                        'error': 'numerical_overflow'
                    }
            
            # 计算位置残差
            residual = x_final - x0
            pos_residual = residual[residual_indices]
            res_norm = la.norm(pos_residual)
            
            # 记录历史
            history.append({
                'iteration': i,
                'residual_norm': res_norm,
                'state': x0.copy(),
                'final_state': x_final.copy(),
                'stm': stm.copy(),
                'damping': current_damping
            })

            # 检查收敛
            if res_norm < tol:
                print(f"✓ Converged at iteration {i+1}, position residual: {res_norm:.2e}")
                return {
                    'state': x0,
                    'period': target_period,
                    'convergence_history': history,
                    'success': True,
                    'final_residual': residual
                }

            # 自适应阻尼：如果残差增加，减少阻尼
            if adaptive_damping and res_norm > prev_residual_norm * 1.1 and i > 2:
                current_damping *= 0.7
                if current_damping < 0.1:
                    current_damping = 0.1
                print(f"  Residual increased, reducing damping to {current_damping:.3f}")
            
            prev_residual_norm = res_norm
            
            # 计算敏感度矩阵
            sensitivity = stm[np.ix_(residual_indices, design_indices)]  # 3x2 or 2x2 matrix
            
            # 检查敏感度矩阵有效性
            if not np.all(np.isfinite(sensitivity)):
                print(f"  Warning: Invalid sensitivity matrix, using gradient descent")
                # 使用梯度下降法作为备选
                grad_descent_step = 0.01
                delta_v_design = -grad_descent_step * pos_residual[:num_design]
            else:
                # 奇异值分解求伪逆，提高数值稳定性
                U, s, Vt = la.svd(sensitivity, full_matrices=False)
                
                # 条件数检查
                cond_number = s[0] / (s[-1] + 1e-12)
                if cond_number > 1e12:
                    print(f"  Warning: High condition number ({cond_number:.1e}), adding regularization")
                    # 添加Tikhonov正则化
                    reg = 1e-8 * np.eye(num_design)
                    delta_v_design = la.solve(sensitivity.T @ sensitivity + reg, 
                                             sensitivity.T @ (-pos_residual))
                else:
                    # 使用SVD伪逆
                    s_inv = np.zeros_like(s)
                    s_inv[s > 1e-12] = 1.0 / s[s > 1e-12]
                    delta_v_design = Vt.T @ np.diag(s_inv) @ U.T @ (-pos_residual)
            
            # 应用阻尼
            delta_v_design *= current_damping
            
            # 限制修正量大小，防止过大跳跃
            max_correction = 0.5
            correction_norm = la.norm(delta_v_design)
            if correction_norm > max_correction:
                delta_v_design = delta_v_design / correction_norm * max_correction
                print(f"  Limiting correction from {correction_norm:.3f} to {max_correction:.3f}")
            
            # 更新设计变量
            x0[design_indices] += delta_v_design

            # 定期输出进度
            if i % 5 == 0 or res_norm < 1e-2 or res_norm > 10:
                corr_str = ', '.join([f"{dv:.3e}" for dv in delta_v_design])
                print(f"  Iter {i+1}: pos residual = {res_norm:.3e}, "
                      f"correction = [{corr_str}], damping = {current_damping:.2f}")

        print(f"✗ Failed to converge after {max_iter} iterations, final residual: {history[-1]['residual_norm']:.2e}")
        return {
            'state': x0,
            'period': target_period,
            'convergence_history': history,
            'success': False,
            'final_residual': residual
        }

    def _get_dynamics_func(self) -> Callable:
        """获取适用于 STM 计算器的动力学函数 f(t, x) -> [vx, vy, vz, ax, ay, az]"""
        if hasattr(self.dynamics, '_crtbp_acceleration_nd'):
            # 使用 UniversalCRTBP 的维度加速度方法
            # 需要包装成完整的6维状态导数 [vel, accel]
            def dynamics_wrapper(t, x):
                accel = self.dynamics._crtbp_acceleration_nd(x)
                return np.concatenate([x[3:6], accel])
            return dynamics_wrapper
        elif callable(self.dynamics):
            # 如果已经是函数形式 f(t, x)，直接使用
            def dynamics_wrapper(t, x):
                return self.dynamics(t, x)
            return dynamics_wrapper
        elif hasattr(self.dynamics, 'compute_derivative'):
            # 如果是对象，包装其 compute_derivative 方法
            def dynamics_wrapper(t, x):
                return self.dynamics.compute_derivative(x)
            return dynamics_wrapper
        else:
            # 默认使用内置 CRTBP 动力学
            def dynamics_wrapper(t, x):
                return self._simple_crtbp_derivative(x)
            return dynamics_wrapper

    def _simple_crtbp_derivative(self, state: np.ndarray) -> np.ndarray:
        """Simplified CRTBP dynamics derivative (dimensionless rotating frame)"""
        x, y, z, vx, vy, vz = state
        mu = self.mu
        
        # Clamp state components to prevent overflow in intermediate calculations
        max_pos = 1e4
        max_vel = 1e4
        x = np.clip(x, -max_pos, max_pos)
        y = np.clip(y, -max_pos, max_pos)
        z = np.clip(z, -max_pos, max_pos)
        vx = np.clip(vx, -max_vel, max_vel)
        vy = np.clip(vy, -max_vel, max_vel)
        vz = np.clip(vz, -max_vel, max_vel)

        # Compute distances with overflow protection
        dx1 = x + mu
        dx2 = x + mu - 1
        
        # Use safe squaring to prevent overflow
        r1_sq = min(dx1**2 + y**2 + z**2, 1e10)
        r2_sq = min(dx2**2 + y**2 + z**2, 1e10)
        
        # Prevent division by zero
        eps = 1e-10
        r1_sq = max(r1_sq, eps)
        r2_sq = max(r2_sq, eps)
        
        r1 = np.sqrt(r1_sq)
        r2 = np.sqrt(r2_sq)
        
        # Clamp distances
        r1 = min(r1, 1e5)
        r2 = min(r2, 1e5)
        
        r1_cubed = r1**3
        r2_cubed = r2**3

        ax = 2*vy + x - (1-mu)*dx1/r1_cubed - mu*dx2/r2_cubed
        ay = -2*vx + y - (1-mu)*y/r1_cubed - mu*y/r2_cubed
        az = -(1-mu)*z/r1_cubed - mu*z/r2_cubed

        return np.array([vx, vy, vz, ax, ay, az])

    def compute_stm(self,
                   initial_state: np.ndarray,
                   duration: float) -> np.ndarray:
        """
        计算状态转移矩阵。

        使用变分方程数值积分计算从 initial_state 出发，
        经过 duration 时间后的状态转移矩阵。

        Args:
            initial_state: 初始状态（6维）
            duration: 积分时长（秒）

        Returns:
            6x6 状态转移矩阵 Φ(duration, 0)
        """
        _, stm = self._stm_calc.propagate_with_stm(
            dynamics=self._get_dynamics_func(),
            initial_state=initial_state,
            t0=0.0,
            tf=duration,
            method=self.integrator_type,
            num_steps=self.num_steps
        )
        return stm

    def analyze_stability(self,
                         orbit_state: np.ndarray,
                         period: float) -> Dict:
        """
        Analyze orbit stability by computing eigenvalues of the monodromy matrix.

        Args:
            orbit_state: Initial state vector (6D)
            period: Orbital period (seconds)

        Returns:
            Dictionary containing eigenvalues, stability indicators, etc.
        """
        stm = self.compute_stm(orbit_state, period)
        
        # Check for invalid STM (NaN/Inf)
        if not np.all(np.isfinite(stm)):
            return {
                'eigenvalues': np.full(6, np.nan),
                'max_magnitude': np.nan,
                'stable': False,
                'monodromy_matrix': stm,
                'error': 'invalid_stm'
            }
        
        eigenvalues = la.eigvals(stm)

        # Stability criterion: all eigenvalue magnitudes <= 1
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
