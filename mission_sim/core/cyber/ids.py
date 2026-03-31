"""
Interface Definition Specification (IDS) - Cyber Domain
-------------------------------------------------------
Contracts for flight software state machines, control algorithms, and ISL networks.
"""
from enum import Enum, auto
from dataclasses import dataclass
import numpy as np

class PlatformGncMode(Enum):
    INITIALIZE = auto()
    STATION_KEEPING = auto()
    FORMATION_ACQUISITION = auto()
    FORMATION_MAINTAIN = auto()
    SAFE_HOLD = auto()

class DataQualityFlag(Enum):
    VALID = auto()
    DEGRADED = auto()
    STALE = auto()
    INVALID = auto()

@dataclass
class ISL_MessageFrame:
    sender_id: int
    receiver_id: int
    msg_type: str
    payload: np.ndarray
    quality_flag: DataQualityFlag
    timestamp: float
