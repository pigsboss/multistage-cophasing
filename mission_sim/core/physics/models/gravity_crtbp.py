# mission_sim/core/physics/models/gravity_crtbp.py
"""
日地系统圆型限制性三体引力模型 (CRTBP) - SI 单位制
包含：太阳引力、地球引力、以及旋转坐标系带来的离心力与科氏力。
作为策略模式的具体实现，可被 CelestialEnvironment 动态加载。
"""

import numpy as np
from mission_sim.core.physics.environment import IForceModel

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
def _crtbp_accel(
    pos: np.ndarray,
    vel: np.ndarray,
    GM_SUN: float,
    GM_EARTH: float,
    OMEGA: float,
    pos_sun: np.ndarray,
    pos_earth: np.ndarray
) -> np.ndarray:
    """
    CRTBP 加速度计算的纯函数 (可被 Numba JIT 编译)。

    Args:
        pos: 航天器位置向量 [x, y, z] (m)
        vel: 航天器速度向量 [vx, vy, vz] (m/s)
        GM_SUN: 太阳引力常数 (m³/s²)
        GM_EARTH: 地球引力常数 (m³/s²)
        OMEGA: 日地系统角速度 (rad/s)
        pos_sun: 太阳在旋转系中的固定位置 [x_s, y_s, z_s] (m)
        pos_earth: 地球在旋转系中的固定位置 [x_e, y_e, z_e] (m)

    Returns:
        加速度向量 [ax, ay, az] (m/s²)
    """
    # 计算航天器相对太阳和地球的位置矢量
    r_sun_vec = pos - pos_sun
    r_earth_vec = pos - pos_earth

    # 计算距离的三次方 (用于分母)
    r_sun_mag3 = np.linalg.norm(r_sun_vec) ** 3
    r_earth_mag3 = np.linalg.norm(r_earth_vec) ** 3

    # 1. 中心天体保守引力场 (牛顿万有引力)
    accel_grav_sun = -GM_SUN * r_sun_vec / r_sun_mag3
    accel_grav_earth = -GM_EARTH * r_earth_vec / r_earth_mag3

    # 2. 旋转坐标系引入的表观惯性力
    # 离心力: a_cf = ω × (ω × r) 的展开项 (由于仅绕 Z 轴旋转)
    accel_centrifugal = np.array([
        OMEGA ** 2 * pos[0],
        OMEGA ** 2 * pos[1],
        0.0
    ], dtype=np.float64)

    # 科氏力: a_cor = -2 ω × v
    accel_coriolis = np.array([
        2.0 * OMEGA * vel[1],
        -2.0 * OMEGA * vel[0],
        0.0
    ], dtype=np.float64)

    # 汇总四大力学分量
    return accel_grav_sun + accel_grav_earth + accel_centrifugal + accel_coriolis


class Gravity_CRTBP(IForceModel):
    """
    日地系统圆型限制性三体引力模型 (CRTBP) - SI 单位制
    包含：太阳引力、地球引力、以及旋转坐标系带来的离心力与科氏力。
    作为策略模式的具体实现，可被 CelestialEnvironment 动态加载。
    """

    def __init__(self):
        # --- 物理常数 (SI 单位制：m, s, kg) ---
        self.GM_SUN = 1.32712440018e20
        self.GM_EARTH = 3.986004418e14
        self.AU = 1.495978707e11
        self.OMEGA = 1.990986e-7  # 地球绕日公转平均角速度 (rad/s)

        # --- 预计算无量纲质量比与天体在旋转系中的绝对位置 ---
        # 旋转系原点位于日地系统质心，X轴指向地球
        self.mu = self.GM_EARTH / (self.GM_SUN + self.GM_EARTH)
        self.pos_sun = np.array([-self.mu * self.AU, 0.0, 0.0], dtype=np.float64)
        self.pos_earth = np.array([(1.0 - self.mu) * self.AU, 0.0, 0.0], dtype=np.float64)

    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        计算特定状态下航天器受到的 CRTBP 总加速度。

        :param state: 航天器状态向量 [x, y, z, vx, vy, vz] (基于日地旋转系)
        :param epoch: 当前历元时间 (当前模型为保守场，不显式依赖 epoch，但预留接口)
        :return: 加速度向量 [ax, ay, az]
        """
        pos = state[0:3]
        vel = state[3:6]

        # 调用纯函数计算加速度
        return _crtbp_accel(
            pos,
            vel,
            self.GM_SUN,
            self.GM_EARTH,
            self.OMEGA,
            self.pos_sun,
            self.pos_earth
        )

    def __repr__(self) -> str:
        return (f"Gravity_CRTBP(mu={self.mu:.2e}, AU={self.AU:.2e}, OMEGA={self.OMEGA:.2e})")