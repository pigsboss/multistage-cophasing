# mission_sim/core/trajectory/generators.py
from abc import ABC, abstractmethod
import numpy as np

from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.ephemeris import Ephemeris

class BaseTrajectoryGenerator(ABC):
    """
    标称轨道生成器抽象基类 (Strategy Pattern Interface)
    职责：规范所有轨道生成算法的输入输出契约。
    无论内部是解析公式、微分修正还是高精度数值积分，最终都必须吐出一个标准的 Ephemeris 对象。
    """
    
    @abstractmethod
    def generate(self, config: dict) -> Ephemeris:
        """
        核心生成接口。
        
        :param config: 包含生成轨道所需参数的字典 (来源于 L1_halo_mission.yaml 等配置)
                       例如包含时长、步长、目标区域、初始猜测等。
        :return: 标准化的离散星历表对象 (Ephemeris)
        """
        pass


class KeplerianGenerator(BaseTrajectoryGenerator):
    """
    开普勒轨道解析生成器 (面向 LEO/GEO 任务)
    通过经典的二体解析公式 (或附加 J2 长期项修正) 快速生成参考星历。
    """
    def __init__(self, mu: float = 3.986004418e14):
        """默认采用地球引力常数 (SI)"""
        self.mu = mu

    def generate(self, config: dict) -> Ephemeris:
        """
        根据轨道根数生成星历。
        :param config: 必须包含 'elements' (a, e, i, Omega, omega, M0), 'dt', 'sim_time'
        """
        print("[KeplerianGenerator] 正在使用解析公式生成二体参考星历...")
        
        elements = config.get("elements")
        if elements is None or len(elements) != 6:
            raise ValueError("KeplerianGenerator 必须在 config 中提供 6 个轨道根数 'elements'。")
            
        dt = config.get("dt", 1.0)
        sim_time = config.get("sim_time", 86400.0)
        times = np.arange(0, sim_time + dt, dt)
        
        a, e, i, Omega, omega, M0 = elements
        n = np.sqrt(self.mu / a**3)
        
        states = []
        for t in times:
            # 1. 解开普勒方程 M = E - e*sin(E) (此处做简化近似，假设 e 极小或圆轨道以作架构演示)
            M = M0 + n * t
            E = M # 对于纯圆轨道 e=0 的近似
            
            # 2. 计算真近点角 (简化处理)
            nu = E
            
            # 3. 轨道面内坐标
            r = a * (1 - e**2) / (1 + e * np.cos(nu))
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)
            vx_orb = -np.sqrt(self.mu / (a * (1 - e**2))) * np.sin(nu)
            vy_orb = np.sqrt(self.mu / (a * (1 - e**2))) * (e + np.cos(nu))
            
            # (省略向 J2000_ECI 转换的 3x3 旋转矩阵计算，仅为架构占位)
            # 实际工程中这里将进行完整的坐标系投影
            state_eci = np.array([x_orb, y_orb, 0.0, vx_orb, vy_orb, 0.0])
            states.append(state_eci)
            
        states = np.array(states)
        
        # 将解析结果封装进防呆保险箱，贴上 ECI 惯性系标签
        return Ephemeris(times=times, states=states, frame=CoordinateFrame.J2000_ECI)
