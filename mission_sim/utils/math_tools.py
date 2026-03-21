# mission_sim/utils/math_tools.py
"""
Mathematical utilities for MCPC framework.
Includes LQR gain calculation, LVLH transformation, orbital mechanics conversions,
and coordinate frame transformations (inertial ↔ rotating).
"""

import numpy as np
import scipy.linalg


def get_lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute the optimal feedback gain matrix K for a continuous-time Linear Quadratic Regulator (LQR).
    Solves the Continuous-time Algebraic Riccati Equation (CARE):
        A^T P + P A - P B R^-1 B^T P + Q = 0

    Args:
        A: State matrix (n x n)
        B: Control input matrix (n x m)
        Q: State weighting matrix (n x n), semi-positive definite
        R: Control weighting matrix (m x m), positive definite

    Returns:
        K: Optimal feedback gain matrix (m x n), such that u(t) = -K * e(t)
    """
    try:
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    except scipy.linalg.LinAlgError as e:
        raise RuntimeError(f"LQR Riccati equation solving failed. Check if A, B are stabilizable! Error: {e}")

    # K = R^-1 B^T P
    K = np.linalg.inv(R) @ B.T @ P
    return K


def absolute_to_lvlh(state_chief: np.ndarray, state_deputy: np.ndarray) -> np.ndarray:
    """
    Transform the absolute state of a deputy spacecraft to the LVLH (Local Vertical, Local Horizontal)
    frame centered on the chief spacecraft.

    LVLH frame definition (Radial-In-Track-Cross-Track):
        - X axis (Radial): along the chief position vector (from central body to chief)
        - Z axis (Cross-track): along the chief angular momentum vector (r x v)
        - Y axis (Along-track): completes the right-handed system (Z x X)

    Args:
        state_chief: Chief absolute state [x, y, z, vx, vy, vz] (inertial frame)
        state_deputy: Deputy absolute state [x, y, z, vx, vy, vz] (same inertial frame)

    Returns:
        Relative state in LVLH frame [x_rel, y_rel, z_rel, vx_rel, vy_rel, vz_rel] (m, m/s)
    """
    r_c = state_chief[0:3]
    v_c = state_chief[3:6]
    r_d = state_deputy[0:3]
    v_d = state_deputy[3:6]

    # --- 1. Compute LVLH basis vectors ---
    norm_r = np.linalg.norm(r_c)
    i_hat = r_c / norm_r

    h_vec = np.cross(r_c, v_c)
    norm_h = np.linalg.norm(h_vec)
    k_hat = h_vec / norm_h

    j_hat = np.cross(k_hat, i_hat)

    # Rotation matrix from inertial to LVLH
    C_I2L = np.vstack([i_hat, j_hat, k_hat])

    # --- 2. Relative position mapping ---
    delta_r_I = r_d - r_c
    rel_pos = C_I2L @ delta_r_I

    # --- 3. Relative velocity mapping (remove Coriolis term due to rotating frame) ---
    # Chief orbital angular velocity magnitude omega = |h| / |r|^2
    omega_mag = norm_h / (norm_r ** 2)
    omega_vec_L = np.array([0.0, 0.0, omega_mag])

    delta_v_I = v_d - v_c
    # Relative velocity = transformed absolute relative velocity - omega x r_rel
    rel_vel = C_I2L @ delta_v_I - np.cross(omega_vec_L, rel_pos)

    return np.concatenate([rel_pos, rel_vel])


def elements_to_cartesian(
    mu: float,
    a: float,
    e: float,
    i: float,
    Omega: float,
    omega: float,
    M: float
) -> np.ndarray:
    """
    Convert classical orbital elements to Cartesian state vector (J2000_ECI or similar inertial frame).

    The conversion follows the standard procedure:
        1. Solve Kepler's equation for eccentric anomaly E.
        2. Compute true anomaly nu.
        3. Compute position and velocity in the orbital plane.
        4. Rotate to the inertial frame using the rotation sequence: Omega (Z), i (X), omega (Z).

    Args:
        mu: Gravitational parameter of the central body (m³/s²)
        a: Semi-major axis (m)
        e: Eccentricity (0 <= e < 1)
        i: Inclination (rad)
        Omega: Right ascension of ascending node (rad)
        omega: Argument of periapsis (rad)
        M: Mean anomaly (rad)

    Returns:
        Cartesian state vector [x, y, z, vx, vy, vz] (m, m/s)
    """
    # 1. Solve Kepler's equation: M = E - e sin(E) using Newton-Raphson
    E = M
    for _ in range(10):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E -= delta
        if abs(delta) < 1e-12:
            break

    # 2. True anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

    # 3. Position and velocity in orbital plane
    r = a * (1 - e * np.cos(E))
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    p = a * (1 - e ** 2)
    vx_orb = -np.sqrt(mu / p) * np.sin(nu)
    vy_orb = np.sqrt(mu / p) * (e + np.cos(nu))

    # 4. Rotation matrix from orbital plane to inertial frame
    # Sequence: rotate by Omega around Z, then i around X, then omega around Z
    cos_Omega = np.cos(Omega)
    sin_Omega = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    R = np.array([
        [cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i,
         -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i,
         sin_Omega * sin_i],
        [sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i,
         -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i,
         -cos_Omega * sin_i],
        [sin_omega * sin_i,
         cos_omega * sin_i,
         cos_i]
    ])

    pos = R @ [x_orb, y_orb, 0.0]
    vel = R @ [vx_orb, vy_orb, 0.0]

    return np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]], dtype=np.float64)


def inertial_to_rotating(
    state_inertial: np.ndarray,
    t: float,
    omega: float
) -> np.ndarray:
    """
    Convert a state vector from the inertial frame (J2000_SSB or similar)
    to the rotating frame that rotates about the Z-axis with constant angular velocity omega.

    The transformation is a simple rotation about the Z-axis:
        x_rot = x_inertial * cos(omega*t) + y_inertial * sin(omega*t)
        y_rot = -x_inertial * sin(omega*t) + y_inertial * cos(omega*t)
        z_rot = z_inertial

    Velocities transform accordingly:
        vx_rot = vx_inertial * cos(omega*t) + vy_inertial * sin(omega*t) + omega * y_rot
        vy_rot = -vx_inertial * sin(omega*t) + vy_inertial * cos(omega*t) - omega * x_rot
        vz_rot = vz_inertial

    Args:
        state_inertial: State vector in inertial frame [x, y, z, vx, vy, vz] (m, m/s)
        t: Time (s)
        omega: Angular velocity of rotating frame (rad/s)

    Returns:
        State vector in rotating frame [x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot] (m, m/s)
    """
    x, y, z, vx, vy, vz = state_inertial

    cos_theta = np.cos(omega * t)
    sin_theta = np.sin(omega * t)

    # Position transformation
    x_rot = x * cos_theta + y * sin_theta
    y_rot = -x * sin_theta + y * cos_theta
    z_rot = z

    # Velocity transformation (includes coriolis term)
    vx_rot = vx * cos_theta + vy * sin_theta + omega * y_rot
    vy_rot = -vx * sin_theta + vy * cos_theta - omega * x_rot
    vz_rot = vz

    return np.array([x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot], dtype=np.float64)


def rotating_to_inertial(
    state_rotating: np.ndarray,
    t: float,
    omega: float
) -> np.ndarray:
    """
    Convert a state vector from the rotating frame (which rotates about Z-axis with constant omega)
    back to the inertial frame.

    The inverse transformation of inertial_to_rotating.

    Args:
        state_rotating: State vector in rotating frame [x, y, z, vx, vy, vz] (m, m/s)
        t: Time (s)
        omega: Angular velocity of rotating frame (rad/s)

    Returns:
        State vector in inertial frame [x_inertial, y_inertial, z_inertial, vx_inertial, vy_inertial, vz_inertial] (m, m/s)
    """
    x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot = state_rotating

    cos_theta = np.cos(omega * t)
    sin_theta = np.sin(omega * t)

    # Position transformation (inverse rotation)
    x = x_rot * cos_theta - y_rot * sin_theta
    y = x_rot * sin_theta + y_rot * cos_theta
    z = z_rot

    # Velocity transformation (inverse of the rotating frame transformation)
    # Derivation: from rotating to inertial, we need to add back the coriolis term.
    # The known relation: v_inertial = v_rot + ω × r
    # Here ω = [0,0,omega], r = [x_rot, y_rot, z_rot] in rotating coordinates.
    # But careful: the velocity transformation can be derived directly from the forward formulas.
    # Alternatively, we can use:
    vx = vx_rot * cos_theta - vy_rot * sin_theta - omega * (x_rot * sin_theta + y_rot * cos_theta)
    vy = vx_rot * sin_theta + vy_rot * cos_theta + omega * (x_rot * cos_theta - y_rot * sin_theta)
    vz = vz_rot

    # Simplified:
    # vx = vx_rot * cos_theta - vy_rot * sin_theta - omega * y
    # vy = vx_rot * sin_theta + vy_rot * cos_theta + omega * x
    # where x,y are the inertial positions computed above.
    # Let's compute using the derived positions for clarity.
    # Using the computed x,y:
    # vx = vx_rot * cos_theta - vy_rot * sin_theta - omega * y
    # vy = vx_rot * sin_theta + vy_rot * cos_theta + omega * x
    # This matches the formula above.

    # We'll use the simpler expressions:
    vx = vx_rot * cos_theta - vy_rot * sin_theta - omega * y
    vy = vx_rot * sin_theta + vy_rot * cos_theta + omega * x
    vz = vz_rot

    return np.array([x, y, z, vx, vy, vz], dtype=np.float64)