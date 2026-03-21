"""CRTBP 运动方程和辅助函数"""
import numpy as np

# 尝试导入 Numba 用于 JIT 加速，若不可用则回退到普通 Python
try:
    from numba import jit as njit
    _HAS_NUMBA = True
except ImportError:
    # 定义一个空装饰器，使代码在无 numba 时仍能运行
    def njit(func):
        return func
    _HAS_NUMBA = False


@njit
def _crtbp_dynamics_nd(state: np.ndarray, mu: float) -> np.ndarray:
    """
    CRTBP 无量纲运动方程的纯函数 (可被 Numba JIT 编译)。

    输入输出均为无量纲状态，无时间依赖。

    Args:
        state: 无量纲状态向量 [x, y, z, vx, vy, vz]
        mu: 质量比

    Returns:
        状态导数 [vx, vy, vz, ax, ay, az] (无量纲)
    """
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    az = -(1-mu)*z/r1**3 - mu*z/r2**3

    return np.array([vx, vy, vz, ax, ay, az])


class CRTBP:
    """圆型限制性三体问题基类，提供运动方程和雅可比常数"""

    def __init__(self, mu, length_unit=1.495978707e11, time_unit=1.990986e-7):
        self.mu = mu
        self.L = length_unit      # 特征长度（AU 或地月距离）
        self.omega = time_unit    # 特征时间倒数（ω）
        # 物理量转换系数
        self.vel_scale = self.L * self.omega

    def dynamics(self, t, state):
        """
        无量纲运动方程（输入输出均为无量纲状态）。
        忽略 t 参数，仅用于兼容 solve_ivp 接口。
        """
        return _crtbp_dynamics_nd(state, self.mu)

    def jacobi_constant(self, state_nd):
        """雅可比常数"""
        x, y, z, vx, vy, vz = state_nd
        mu = self.mu
        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        U = (x**2 + y**2)/2 + (1-mu)/r1 + mu/r2
        v2 = vx**2 + vy**2 + vz**2
        return 2*U - v2

    def to_physical(self, state_nd, t_nd):
        """无量纲转物理"""
        state_phys = state_nd.copy()
        state_phys[0:3] *= self.L
        state_phys[3:6] *= self.vel_scale
        t_phys = t_nd / self.omega
        return state_phys, t_phys

    def to_nd(self, state_phys, t_phys):
        """物理转无量纲"""
        state_nd = state_phys.copy()
        state_nd[0:3] /= self.L
        state_nd[3:6] /= self.vel_scale
        t_nd = t_phys * self.omega
        return state_nd, t_nd