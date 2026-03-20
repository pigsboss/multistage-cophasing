# mission_sim/core/physics/environment.py
import numpy as np
from abc import ABC, abstractmethod
from mission_sim.core.types import CoordinateFrame

class IForceModel(ABC):
    """
    力学模型接口契约 (Strategy Pattern Interface)
    约束：所有具体的力学模型（引力、光压、大气阻力等）必须实现此接口。
    这保证了环境引擎可以在不修改自身代码的情况下，动态加载任意摄动模型。
    """
    @abstractmethod
    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        计算特定状态和时间下的物理加速度。
        :param state: 航天器状态向量 [x, y, z, vx, vy, vz]
        :param epoch: 当前仿真历元时间 (s)
        :return: 加速度向量 [ax, ay, az]
        """
        pass


class CelestialEnvironment:
    """
    天体力学环境与调度引擎 (Level 1)
    职责：作为力学注册表，管理多种摄动力模型，并向航天器提供统一的总加速度。
    """
    def __init__(self, computation_frame: CoordinateFrame, initial_epoch: float = 0.0):
        """
        :param computation_frame: 该环境引擎执行物理计算的基础坐标系 (强契约)
        :param initial_epoch: 初始历元时间 (s)
        """
        self.computation_frame = computation_frame
        self.epoch = float(initial_epoch)
        
        # 核心：力学模型注册表 (解耦具体的物理公式)
        self._force_registry: list[IForceModel] = []

    def register_force(self, force_model: IForceModel):
        """
        动态注入摄动力模型。
        """
        if not isinstance(force_model, IForceModel):
            raise TypeError("注册的力学模型必须实现 IForceModel 接口。")
        
        self._force_registry.append(force_model)
        print(f"[Environment] 成功注册力学模型: {force_model.__class__.__name__}")

    def step_time(self, dt: float):
        """推进环境历元"""
        self.epoch += dt

    def get_total_acceleration(self, sc_state: np.ndarray, sc_frame: CoordinateFrame) -> tuple[np.ndarray, CoordinateFrame]:
        """
        计算并汇总当前注册表中所有力学模型产生的总加速度。
        [航天器本体 -> 环境引擎] 接口，强制执行坐标系一致性校验。
        
        :param sc_state: 航天器状态向量 [x, y, z, vx, vy, vz]
        :param sc_frame: 航天器当前所在的坐标系
        :return: (总加速度向量 [ax, ay, az], 加速度所在的坐标系标签)
        """
        # 【防呆校验】确保航天器没有“走错片场”
        if sc_frame != self.computation_frame:
            raise ValueError(
                f"[环境引擎崩溃] 坐标系冲突！当前宇宙环境基于 {self.computation_frame.name} 运转，"
                f"但传入的航天器状态基于 {sc_frame.name}。请检查 GNC 或预处理逻辑！"
            )

        total_accel = np.zeros(3, dtype=np.float64)
        
        # 遍历所有已注册的力学模型并叠加加速度
        for force in self._force_registry:
            accel = force.compute_accel(sc_state, self.epoch)
            total_accel += accel
            
        # 返回时“盖上印章”，以便 SpacecraftPointMass 再次反向校验
        return total_accel, self.computation_frame

    def __repr__(self):
        forces = [f.__class__.__name__ for f in self._force_registry]
        return (f"CelestialEnvironment | Frame={self.computation_frame.name} | "
                f"Epoch={self.epoch:.1f}s | ActiveForces={forces}")
