"""
MCPC Core Law: Spacetime Domain Interface Definition Specification (IDS)
------------------------------------------------------------------------
This file defines universally accepted coordinate frames, command formats, 
and the Level 2 (L2) multi-satellite FormationState container. 
All cross-domain interactions must be based on the structures defined here.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Tuple


class CoordinateFrame(Enum):
    """Global unified coordinate frame contracts."""
    J2000_ECI = auto()           # J2000 Earth-Centered Inertial / Heliocentric Inertial
    SUN_EARTH_ROTATING = auto()  # Sun-Earth Rotating Frame (CRTBP)
    LVLH = auto()                # Local Vertical Local Horizontal (Relative Frame)
    VVLH = auto()                # Velocity-Velocity Local Horizontal


@dataclass
class Telecommand:
    """
    Commands issued by the Cyber brain to the Physics actuators.
    Strictly constrained by coordinate frame and duration.
    Follows Law 3: Impulse Equivalence Principle.
    """
    force_vector: np.ndarray     # 3x1 Desired force vector (N)
    frame: CoordinateFrame       # Frame in which the force is defined (typically LVLH)
    duration_s: float            # Expected duration of the force (s)
    actuator_id: str             # Target actuator ID (e.g., "THR_MAIN_1")

@dataclass
class Telemetry:
    """
    [MCPC UNIVERSAL]
    Standard telemetry report from the Physics domain to the Cyber domain.
    Used for single-satellite absolute state reporting (legacy L1 & universal base).
    """
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    frame: CoordinateFrame

@dataclass
class FormationState:
    """
    L2 Core Data Bus: Multi-satellite Formation State Container.
    Carries the absolute state of the Chief and the relative states of N Deputies.
    """
    timestamp: float
    chief_position: np.ndarray       # Absolute position of Chief (3x1)
    chief_velocity: np.ndarray       # Absolute velocity of Chief (3x1)
    chief_frame: CoordinateFrame     # Chief's absolute frame (J2000 or ROTATING)
    
    # Explicitly maintain Deputy IDs to prevent index-mismatch bugs in multi-star topologies
    deputy_ids: List[str] = field(default_factory=list)
    deputy_relative_positions: List[np.ndarray] = field(default_factory=list)
    deputy_relative_velocities: List[np.ndarray] = field(default_factory=list)
    deputy_frame: CoordinateFrame = CoordinateFrame.LVLH  # Default frame for relative states

    def get_num_deputies(self) -> int:
        """Returns the number of deputies currently in the formation."""
        return len(self.deputy_ids)

    def add_deputy_state(self, deputy_id: str, rel_pos: np.ndarray, rel_vel: np.ndarray) -> None:
        """Mounts the relative state of a deputy spacecraft."""
        self.deputy_ids.append(deputy_id)
        self.deputy_relative_positions.append(np.array(rel_pos, dtype=np.float64))
        self.deputy_relative_velocities.append(np.array(rel_vel, dtype=np.float64))

    def get_deputy_index(self, deputy_id: str) -> int:
        """Safely retrieves the data index for a deputy based on its ID."""
        try:
            return self.deputy_ids.index(deputy_id)
        except ValueError:
            raise KeyError(f"Deputy ID '{deputy_id}' not found in current formation.")

    # --- Serialization methods for HDF5 logging and configuration support ---

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the FormationState to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "chief_position": self.chief_position.tolist(),
            "chief_velocity": self.chief_velocity.tolist(),
            "chief_frame": self.chief_frame.name,
            "deputy_ids": self.deputy_ids,
            "deputy_relative_positions": [p.tolist() for p in self.deputy_relative_positions],
            "deputy_relative_velocities": [v.tolist() for v in self.deputy_relative_velocities],
            "deputy_frame": self.deputy_frame.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormationState':
        """Deserializes a dictionary back into a FormationState instance."""
        return cls(
            timestamp=data["timestamp"],
            chief_position=np.array(data["chief_position"]),
            chief_velocity=np.array(data["chief_velocity"]),
            chief_frame=CoordinateFrame[data["chief_frame"]],
            deputy_ids=data.get("deputy_ids", []),
            deputy_relative_positions=[np.array(p) for p in data.get("deputy_relative_positions", [])],
            deputy_relative_velocities=[np.array(v) for v in data.get("deputy_relative_velocities", [])],
            deputy_frame=CoordinateFrame[data.get("deputy_frame", "LVLH")]
        )
