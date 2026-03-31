"""
Interface Definition Specification (IDS) - Physics Domain
---------------------------------------------------------
Contracts for physical constants, environmental states, and hardware faults.
"""
from enum import Enum, auto

class PhysicalConstants:
    G = 6.67430e-11          # Gravitational constant (m^3 kg^-1 s^-2)
    C = 299792458.0          # Speed of light (m/s)
    AU = 149597870700.0      # Astronomical Unit (m)

class ComponentHealthStatus(Enum):
    NOMINAL = auto()
    DEGRADED = auto()
    FAILED = auto()
