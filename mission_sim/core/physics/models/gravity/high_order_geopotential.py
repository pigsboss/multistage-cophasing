"""
高阶地球重力场模型（球谐函数展开）

实现 HighOrderGeopotential 类，作为 IForceModel 接口的具体实现。
目前提供简化的 J2 加速度计算，后续可扩展为完整的球谐函数模型。
"""
import numpy as np
from mission_sim.core.physics.environment import IForceModel


class HighOrderGeopotential(IForceModel):
    """高阶地球重力场模型（球谐函数展开）"""

    def __init__(self,
                 degree: int = 10,
                 order: int = 10,
                 coeff_file: str = None):
        """
        初始化重力场模型。

        Args:
            degree: 最大阶数
            order: 最大次数（通常 order <= degree）
            coeff_file: 球谐系数文件路径，None 则使用内置系数
        """
        self.degree = degree
        self.order = order
        self.coeff_file = coeff_file

        # 地球引力常数 (m³/s²)
        self.mu_earth = 3.986004418e14
        # 地球赤道半径 (m)
        self.R_earth = 6378137.0
        # J2 系数
        self.J2 = 1.08262668e-3

        # TODO: 从 coeff_file 加载球谐系数
        print(f"初始化高阶重力场模型: 阶数={degree}, 次数={order}")

    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        计算高阶重力场加速度（地心惯性系）。

        Args:
            state: 航天器状态 [x, y, z, vx, vy, vz] (m, m/s)
            epoch: 当前时间（秒）

        Returns:
            加速度向量 [ax, ay, az] (m/s²)
        """
        # 简化的 J2 加速度计算（临时实现）
        # TODO: 实现完整的球谐函数展开

        pos = state[0:3]
        r = np.linalg.norm(pos)

        # 避免除以零
        if r < 1e-6:
            return np.zeros(3)

        x, y, z = pos
        r_sq = r * r
        r_5 = r_sq * r_sq * r

        factor = 1.5 * self.J2 * self.mu_earth * self.R_earth * self.R_earth / r_5

        ax = factor * x * (5 * z * z / r_sq - 1)
        ay = factor * y * (5 * z * z / r_sq - 1)
        az = factor * z * (5 * z * z / r_sq - 3)

        return np.array([ax, ay, az])

    def set_max_degree(self, degree: int):
        """动态设置最大阶数（用于性能与精度权衡）"""
        self.degree = degree
        print(f"设置最大阶数为: {degree}")

    def __repr__(self) -> str:
        return (f"HighOrderGeopotential(degree={self.degree}, "
                f"order={self.order}, coeff_file={self.coeff_file})")
