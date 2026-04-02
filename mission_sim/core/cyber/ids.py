"""
MCPC Core Law: Cyber Domain Interface Definition Specification (IDS)
--------------------------------------------------------------------
Defines the communication and control contracts.
Maintains the boundary between universal data routing and L2 control modes.
"""

# =============================================================================
# [ABSTRACT / MCPC UNIVERSAL]
# The standard network container for all inter-satellite data exchange.
# By wrapping 'PhysicalMeasurementBase', it supports any sensor type 
# from L2 point-mass measurements to L4 multi-sensor fusion packets.
# =============================================================================

from enum import Enum, auto
import numpy as np
from dataclasses import dataclass
from mission_sim.core.physics.ids import PhysicalMeasurementBase

@dataclass
class ISLNetworkFrame:
    """
    Universal MCPC Network Frame.
    Adds 'Cyber' metadata (routing, delays, aging) to 'Physics' reality.
    """
    payload: PhysicalMeasurementBase  # Generic payload to support all MCPC Levels
    source_id: str                    # ID of the transmitting spacecraft
    dest_id: str                      # ID of the receiving spacecraft
    tx_time: float                    # Simulated transmission timestamp
    rx_time: float                    # Expected reception timestamp (including latency)
    
    def get_age(self, current_time: float) -> float:
        """[UNIVERSAL] Calculate the latency since physical generation."""
        return current_time - self.payload.phys_timestamp
        
    def is_stale(self, current_time: float, max_delay: float) -> bool:
        """[UNIVERSAL] Check if the packet exceeds the staleness threshold."""
        return self.get_age(current_time) > max_delay

class PlatformGNCMode(Enum):
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
class ISLMessage:
    sender_id: int
    receiver_id: int
    msg_type: str
    payload: np.ndarray
    quality_flag: DataQualityFlag
    timestamp: float

# =============================================================================
# [L2-SPECIFIC / FORMATION CONTROL MODES]
# Defines the discrete states for the L2 formation autonomous state machine.
# =============================================================================
class FormationMode(Enum):
    """
    L2-Specific: Modes for the Three-Stage Formation State Machine.
    Governs the high-level behavior of the Deputy spacecraft.
    """
    GENERATION = auto()       # Phase 1: Initial acquisition and rendezvous
    KEEPING = auto()          # Phase 2: Steady-state baseline maintenance
    RECONFIGURATION = auto()  # Phase 3: Transitioning to a new relative geometry
