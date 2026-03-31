"""
Interface Definition Specification (IDS) - Spacetime Domain
-----------------------------------------------------------
Global contracts for coordinate frames, time, and cross-domain data exchange.
"""
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional

class CoordinateFrame(Enum):
    J2000_ECI = auto()
    SUN_EARTH_ROTATING = auto()
    LVLH = auto()
    VVLH = auto()

@dataclass
class Telecommand:
    force_vector: np.ndarray
    frame: CoordinateFrame
    duration_s: float
    actuator_id: Optional[str] = None

@dataclass
class Telemetry:
    position: np.ndarray
    velocity: np.ndarray
    frame: CoordinateFrame
    timestamp: float

@dataclass
class FormationState:
    timestamp: float
    chief_position: np.ndarray
    chief_velocity: np.ndarray
    chief_frame: CoordinateFrame
    deputy_relative_positions: List[np.ndarray] = field(default_factory=list)
    deputy_relative_velocities: List[np.ndarray] = field(default_factory=list)
    deputy_frame: CoordinateFrame = CoordinateFrame.LVLH

    # 补充下面这两个方法
    def add_deputy_state(self, rel_pos: np.ndarray, rel_vel: np.ndarray):
        """添加一颗从星的相对状态"""
        self.deputy_relative_positions.append(rel_pos)
        self.deputy_relative_velocities.append(rel_vel)

    def get_num_deputies(self) -> int:
        """获取编队中的从星数量"""
        return len(self.deputy_relative_positions)
