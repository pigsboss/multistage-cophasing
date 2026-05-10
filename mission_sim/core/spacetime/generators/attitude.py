"""
Numerical attitude generator.

Integrates Euler’s rotational equations of motion (or other attitude dynamics)
to produce a time‑series of quaternions, which can then be used to construct
an InterpolatedAttitude instance.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from scipy.integrate import solve_ivp

from mission_sim.core.spacetime.attitude.base import AttitudeModel
from mission_sim.core.spacetime.attitude.interpolated import InterpolatedAttitude


class NumericalAttitudeGenerator:
    """Generate attitude quaternion series by numerical integration.

    The integrator solves the standard torque‑free or torque‑controlled
    rotational dynamics.  The output is an `InterpolatedAttitude` model
    that can be queried at arbitrary times.

    Args:
        inertia_matrix: 3×3 inertia tensor (kg·m²).
        initial_quat: Initial quaternion [w, x, y, z].
        initial_angvel: Initial angular velocity [wx, wy, wz] (rad/s).
        torque_function: Callable(t) returning 3‑vector of external torque (Nm).
                         If None, torque‑free motion is assumed.
    """

    def __init__(
        self,
        inertia_matrix: np.ndarray,
        initial_quat: np.ndarray,
        initial_angvel: np.ndarray,
        torque_function: Optional = None,
    ):
        self._I = np.asarray(inertia_matrix, dtype=np.float64)
        self._I_inv = np.linalg.inv(self._I)
        self._q0 = np.asarray(initial_quat, dtype=np.float64)
        self._w0 = np.asarray(initial_angvel, dtype=np.float64)
        self._torque = torque_function

    def generate(
        self,
        t_span: tuple,
        dt: float,
        method: str = "RK45",
        **integrator_kwargs,
    ) -> InterpolatedAttitude:
        """Integrate attitude dynamics and return an InterpolatedAttitude.

        Args:
            t_span: (t_start, t_end) in seconds.
            dt: Requested time step (output spacing).
            method: Integration method passed to solve_ivp.
            **integrator_kwargs: Additional keyword arguments for solve_ivp.

        Returns:
            InterpolatedAttitude instance.
        """
        # State vector: [q (4), omega (3)]
        def rhs(t, state):
            q = state[:4]
            w = state[4:]
            # Normalise quaternion (maintain unit)
            q_norm = q / np.linalg.norm(q)
            # Quaternion kinematics: dq/dt = 0.5 * Omega(w) * q
            wx, wy, wz = w
            Omega = 0.5 * np.array([
                [ 0, -wx, -wy, -wz],
                [ wx,  0,  wz, -wy],
                [ wy, -wz,  0,  wx],
                [ wz,  wy, -wx,  0],
            ])
            dq = Omega @ q_norm
            # Euler’s rotational equation: I·dw/dt = Torque - cross(w, I·w)
            Iw = self._I @ w
            if self._torque is None:
                torque = np.zeros(3)
            else:
                torque = self._torque(t)
            dw = self._I_inv @ (torque - np.cross(w, Iw))
            return np.concatenate([dq, dw])

        # Initial state
        y0 = np.concatenate([self._q0, self._w0])

        # Evaluate
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            **integrator_kwargs,
        )

        times = sol.t
        quats = sol.y[:4].T   # shape (N, 4)

        # Normalise quaternions
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        return InterpolatedAttitude(times=times, quaternions=quats)


# Convenience factory function
def create_numerical_attitude_generator(
    config_dict: Dict[str, Any]
) -> NumericalAttitudeGenerator:
    """Create a NumericalAttitudeGenerator from a configuration dictionary.

    Expected keys:
        - inertia: 3×3 inertia matrix (list of lists)
        - initial_quat: 4‑element list [w, x, y, z]
        - initial_angvel: 3‑element list [wx, wy, wz]
        - torque_function: optional callable (if not provided, torque‑free)
    """
    inertia = np.array(config_dict["inertia"], dtype=np.float64)
    q0 = np.array(config_dict["initial_quat"], dtype=np.float64)
    w0 = np.array(config_dict["initial_angvel"], dtype=np.float64)
    torque = config_dict.get("torque_function", None)
    return NumericalAttitudeGenerator(inertia, q0, w0, torque)
