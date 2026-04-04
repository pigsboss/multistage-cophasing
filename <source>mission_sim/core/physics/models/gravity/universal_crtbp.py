"""
Universal Circular Restricted Three-Body Problem (CRTBP) Model

This module provides a generic CRTBP implementation that supports any two-body system
(Earth-Moon, Sun-Earth, etc.) and follows the IForceModel interface.
"""
import numpy as np
from abc import ABC, abstractmethod
from mission_sim.core.physics.models.base import IForceModel


class IUniversalCRTBP(IForceModel, ABC):
    """Interface for Universal CRTBP models"""
    
    @classmethod
    @abstractmethod
    def earth_moon_system(cls) -> 'IUniversalCRTBP':
        """Create an Earth-Moon system CRTBP instance"""
        pass
    
    @classmethod
    @abstractmethod
    def sun_earth_system(cls) -> 'IUniversalCRTBP':
        """Create a Sun-Earth system CRTBP instance (for compatibility)"""
        pass
    
    @property
    @abstractmethod
    def mu(self) -> float:
        """Mass ratio μ = m2/(m1+m2)"""
        pass
    
    @property
    @abstractmethod
    def distance(self) -> float:
        """Distance between primary bodies (m)"""
        pass
    
    @property
    @abstractmethod
    def omega(self) -> float:
        """System angular velocity (rad/s)"""
        pass
    
    @abstractmethod
    def compute_acceleration(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        Compute CRTBP acceleration
        
        Args:
            state: [x, y, z, vx, vy, vz] (SI units, rotating frame)
            epoch: Time (seconds)
            
        Returns:
            np.ndarray: Acceleration [ax, ay, az] (m/s²)
        """
        pass
    
    @abstractmethod
    def jacobi_constant(self, state: np.ndarray) -> float:
        """Compute Jacobi constant (dimensionless)"""
        pass
    
    @abstractmethod
    def to_nd(self, state_physical: np.ndarray) -> np.ndarray:
        """Convert physical state to dimensionless state"""
        pass
    
    @abstractmethod
    def to_physical(self, state_nd: np.ndarray) -> np.ndarray:
        """Convert dimensionless state to physical state"""
        pass


# Stub implementation for testing
class UniversalCRTBP(IUniversalCRTBP):
    """Stub implementation of Universal CRTBP for interface testing"""
    
    def __init__(self, primary_mass: float, secondary_mass: float, 
                 distance: float, system_name: str = 'custom'):
        self._primary_mass = primary_mass
        self._secondary_mass = secondary_mass
        self._distance = distance
        self._system_name = system_name
        
        # Compute derived parameters
        self._mu = secondary_mass / (primary_mass + secondary_mass)
        self._omega = np.sqrt((primary_mass + secondary_mass) / distance**3)
    
    @classmethod
    def earth_moon_system(cls) -> 'UniversalCRTBP':
        """Create Earth-Moon system CRTBP"""
        earth_mass = 5.972e24  # kg
        moon_mass = 7.342e22   # kg
        distance = 3.844e8     # m
        return cls(earth_mass, moon_mass, distance, 'earth_moon')
    
    @classmethod
    def sun_earth_system(cls) -> 'UniversalCRTBP':
        """Create Sun-Earth system CRTBP"""
        sun_mass = 1.989e30    # kg
        earth_mass = 5.972e24  # kg
        distance = 1.496e11    # m
        return cls(sun_mass, earth_mass, distance, 'sun_earth')
    
    @property
    def mu(self) -> float:
        return self._mu
    
    @property
    def distance(self) -> float:
        return self._distance
    
    @property
    def omega(self) -> float:
        return self._omega
    
    def compute_acceleration(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """Stub implementation - returns zeros"""
        return np.zeros(3)
    
    def jacobi_constant(self, state: np.ndarray) -> float:
        """Stub implementation - returns constant value"""
        return 3.0
    
    def to_nd(self, state_physical: np.ndarray) -> np.ndarray:
        """Stub implementation - returns same state"""
        return state_physical.copy()
    
    def to_physical(self, state_nd: np.ndarray) -> np.ndarray:
        """Stub implementation - returns same state"""
        return state_nd.copy()
    
    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """Alias for compute_acceleration (IForceModel compatibility)"""
        return self.compute_acceleration(state, epoch)
