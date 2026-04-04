import numpy as np
from typing import Tuple, Dict, Any
from mission_sim.core.physics.environment import IForceModel
from mission_sim.utils.math_tools import inertial_to_rotating, rotating_to_inertial


class UniversalCRTBP(IForceModel):
    """
    Universal Circular Restricted Three-Body Problem force model.
    """
    
    def __init__(self, primary_mass: float, secondary_mass: float, distance: float, system_name: str = 'custom'):
        """
        Initialize a CRTBP system with given masses and distance.
        
        Args:
            primary_mass: Mass of primary body (kg)
            secondary_mass: Mass of secondary body (kg)
            distance: Distance between primary and secondary (m)
            system_name: Name of the system
        """
        self.primary_mass = primary_mass
        self.secondary_mass = secondary_mass
        self.distance = distance
        self.system_name = system_name
        
        # Mass ratio μ = m2 / (m1 + m2)
        self.mu = secondary_mass / (primary_mass + secondary_mass)
        
        # Angular velocity ω = sqrt(G(m1 + m2) / d^3)
        # Using gravitational constant G = 6.67430e-11
        G = 6.67430e-11
        self.omega = np.sqrt(G * (primary_mass + secondary_mass) / distance**3)
        
        # Positions of primary and secondary in rotating frame
        self._x1 = -self.mu * distance  # Primary (larger mass)
        self._x2 = (1 - self.mu) * distance  # Secondary (smaller mass)
    
    @classmethod
    def earth_moon_system(cls):
        """Create an Earth-Moon CRTBP system."""
        earth_mass = 5.972e24
        moon_mass = 7.342e22
        distance = 3.844e8
        instance = cls(earth_mass, moon_mass, distance, 'earth_moon')
        return instance
    
    @classmethod
    def sun_earth_system(cls):
        """Create a Sun-Earth CRTBP system."""
        sun_mass = 1.989e30
        earth_mass = 5.972e24
        distance = 1.496e11
        instance = cls(sun_mass, earth_mass, distance, 'sun_earth')
        return instance
    
    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        Compute acceleration in the rotating frame.
        
        Args:
            state: State vector [x, y, z, vx, vy, vz] in rotating frame
            epoch: Time (s)
            
        Returns:
            Acceleration vector [ax, ay, az]
        """
        x, y, z = state[0:3]
        
        # Distances to primary and secondary
        r1 = np.sqrt((x - self._x1)**2 + y**2 + z**2)
        r2 = np.sqrt((x - self._x2)**2 + y**2 + z**2)
        
        # Avoid division by zero
        r1_3 = r1**3
        r2_3 = r2**3
        if r1_3 < 1e-30:
            r1_3 = 1e-30
        if r2_3 < 1e-30:
            r2_3 = 1e-30
        
        # CRTBP acceleration formula
        ax = (x - (1 - self.mu) * (x - self._x1) / r1_3 - 
              self.mu * (x - self._x2) / r2_3 + 2 * self.omega * state[4] + 
              self.omega**2 * x)
        ay = (y - (1 - self.mu) * y / r1_3 - 
              self.mu * y / r2_3 - 2 * self.omega * state[3] + 
              self.omega**2 * y)
        az = (-(1 - self.mu) * z / r1_3 - self.mu * z / r2_3)
        
        return np.array([ax, ay, az], dtype=np.float64)
    
    def compute_vectorized_acc(self, state_matrix: np.ndarray, epoch: float) -> np.ndarray:
        """
        Vectorized acceleration computation.
        
        Args:
            state_matrix: State matrix of shape (N, 6)
            epoch: Time (s)
            
        Returns:
            Acceleration matrix of shape (N, 3)
        """
        N = state_matrix.shape[0]
        acc_matrix = np.zeros((N, 3), dtype=np.float64)
        
        for i in range(N):
            acc_matrix[i] = self.compute_accel(state_matrix[i], epoch)
            
        return acc_matrix
    
    def jacobi_constant(self, state: np.ndarray) -> float:
        """
        Compute Jacobi constant for a given state.
        
        Args:
            state: State vector [x, y, z, vx, vy, vz]
            
        Returns:
            Jacobi constant value
        """
        x, y, z, vx, vy, vz = state
        
        # Distances to primary and secondary
        r1 = np.sqrt((x - self._x1)**2 + y**2 + z**2)
        r2 = np.sqrt((x - self._x2)**2 + y**2 + z**2)
        
        # Effective potential
        U = (x**2 + y**2) / 2 + (1 - self.mu) / r1 + self.mu / r2
        
        # Velocity squared
        v2 = vx**2 + vy**2 + vz**2
        
        # Jacobi constant
        C = 2 * U - v2
        
        return float(C)
    
    def to_rotating_frame(self, inertial_state: np.ndarray, epoch: float) -> np.ndarray:
        """
        Convert from inertial frame to rotating frame.
        
        Args:
            inertial_state: State in inertial frame
            epoch: Time (s)
            
        Returns:
            State in rotating frame
        """
        return inertial_to_rotating(inertial_state, epoch, self.omega)
    
    def to_inertial_frame(self, rotating_state: np.ndarray, epoch: float) -> np.ndarray:
        """
        Convert from rotating frame to inertial frame.
        
        Args:
            rotating_state: State in rotating frame
            epoch: Time (s)
            
        Returns:
            State in inertial frame
        """
        return rotating_to_inertial(rotating_state, epoch, self.omega)
    
    def get_system_parameters(self) -> Dict[str, Any]:
        """Get system parameters as a dictionary."""
        return {
            'system_name': self.system_name,
            'mu': self.mu,
            'omega': self.omega,
            'distance': self.distance,
            'primary_mass': self.primary_mass,
            'secondary_mass': self.secondary_mass,
            'x1': self._x1,
            'x2': self._x2
        }
    
    def get_lagrange_points_nd(self) -> Dict[str, np.ndarray]:
        """
        Get Lagrange points in non-dimensional coordinates.
        
        Returns:
            Dictionary with keys 'L1' through 'L5'
        """
        # For simplicity, return approximate positions
        # In practice, these would be computed by solving the equations
        mu = self.mu
        
        # Approximate L1, L2, L3 positions (collinear points)
        # These are rough approximations
        gamma1 = (mu / 3)**(1/3)
        gamma2 = (mu / 3)**(1/3)
        gamma3 = 1 - (7/12) * mu
        
        L1 = np.array([1 - mu - gamma1, 0.0, 0.0]) * self.distance
        L2 = np.array([1 - mu + gamma2, 0.0, 0.0]) * self.distance
        L3 = np.array([-1 - mu + gamma3, 0.0, 0.0]) * self.distance
        L4 = np.array([0.5 - mu, np.sqrt(3)/2, 0.0]) * self.distance
        L5 = np.array([0.5 - mu, -np.sqrt(3)/2, 0.0]) * self.distance
        
        return {
            'L1': L1,
            'L2': L2,
            'L3': L3,
            'L4': L4,
            'L5': L5
        }
    
    def _to_nd(self, physical_state: np.ndarray) -> np.ndarray:
        """Convert physical state to non-dimensional units (for internal use)."""
        # Simple scaling for now
        return physical_state / self.distance
    
    def _to_physical(self, nd_state: np.ndarray) -> np.ndarray:
        """Convert non-dimensional state to physical units (for internal use)."""
        return nd_state * self.distance
