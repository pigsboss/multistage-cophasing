# mission_sim/core/physics/models/gravity_crtbp.py
import numpy as np
from mission_sim.core.physics.environment import IForceModel

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
        
        # 计算航天器相对太阳和地球的位置矢量
        r_sun_vec = pos - self.pos_sun
        r_earth_vec = pos - self.pos_earth
        
        # 计算距离的三次方 (用于分母)
        r_sun_mag3 = np.linalg.norm(r_sun_vec)**3
        r_earth_mag3 = np.linalg.norm(r_earth_vec)**3
        
        # 1. 中心天体保守引力场 (牛顿万有引力)
        accel_grav_sun = -self.GM_SUN * r_sun_vec / r_sun_mag3
        accel_grav_earth = -self.GM_EARTH * r_earth_vec / r_earth_mag3
        
        # 2. 旋转坐标系引入的表观惯性力
        # 离心力: a_cf = \omega \times (\omega \times r) 的展开项 (由于仅绕Z轴旋转)
        accel_centrifugal = np.array([
            self.OMEGA**2 * pos[0],
            self.OMEGA**2 * pos[1],
            0.0
        ], dtype=np.float64)
        
        # 科氏力: a_cor = -2 * \omega \times v
        accel_coriolis = np.array([
            2.0 * self.OMEGA * vel[1],
            -2.0 * self.OMEGA * vel[0],
            0.0
        ], dtype=np.float64)
        
        # 汇总四大力学分量
        return accel_grav_sun + accel_grav_earth + accel_centrifugal + accel_coriolis
