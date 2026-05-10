"""Attitude model abstract base class.

All concrete attitude models must implement `rotation_at` and `quaternion_at`.
Optionally `angular_velocity_at` and `derivative_at` may be provided.
"""

from abc import ABC, abstractmethod
import numpy as np


class AttitudeModel(ABC):
    """Abstract base class for all attitude models.

    Subclasses must implement `rotation_at` and `quaternion_at`.
    """

    @abstractmethod
    def rotation_at(self, t: float) -> np.ndarray:
        """Return 3×3 rotation matrix from J2000 ECI to the body‑fixed frame at time *t*.

        Args:
            t: Time in seconds since J2000.0.

        Returns:
            np.ndarray: 3×3 rotation matrix (float64).
        """
        ...

    @abstractmethod
    def quaternion_at(self, t: float) -> np.ndarray:
        """Return unit quaternion [w, x, y, z] at time *t*.

        Args:
            t: Time in seconds since J2000.0.

        Returns:
            np.ndarray: shape (4,) unit quaternion.
        """
        ...
