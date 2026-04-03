# mission_sim/core/cyber/models/crtbp_relative_dynamics.py
"""
CRTBP-based relative dynamics for formation flying around libration points.
Provides discrete-time state transition matrix (STM) computed via matrix
exponential of the linearized dynamics matrix, assuming it is constant over
the time step (frozen-time approximation). This is numerically stable and
efficient for small time steps (dt <= 100 s) typical for L2 simulations.

For higher accuracy with time-varying A, one can integrate the variational
equations, but the exponential method works well for dt up to a few minutes.
"""

import numpy as np
from scipy.linalg import expm
from mission_sim.core.cyber.models.relative_dynamics import RelativeDynamics
from mission_sim.core.cyber.models.threebody.base import CRTBP


class CRTBPRelativeDynamics(RelativeDynamics):
    """
    Relative dynamics for a deputy around a chief in a CRTBP orbit (e.g., Halo).
    The STM is computed using the matrix exponential of the linearized dynamics
    matrix A(t), which is assumed constant over the time step (frozen-time).
    This is stable and efficient for typical simulation step sizes (<= 100 s).
    """

    def __init__(self, crtbp: CRTBP, chief_trajectory=None):
        """
        Initialize CRTBP relative dynamics.

        Args:
            crtbp: CRTBP instance (provides dynamics and constants)
            chief_trajectory: Optional function chief_state(t) -> np.ndarray (6,)
                              If provided, A(t) is evaluated at the midpoint
                              of the time step. If None, the chief state is
                              assumed constant (frozen orbit).
        """
        self.crtbp = crtbp
        self.chief_trajectory = chief_trajectory

    def compute_discrete_stm(self, dt: float, current_time: float = 0.0,
                             chief_state: np.ndarray = None) -> np.ndarray:
        """
        Compute the discrete STM Φ(dt) = exp(A * dt), where A is the
        linearized CRTBP dynamics matrix evaluated at the chief state.

        If chief_trajectory is provided, A is evaluated at the midpoint
        time (current_time + dt/2) to improve accuracy. Otherwise, the
        given chief_state is used (or the current time's state if None).

        Args:
            dt: Time step (s)
            current_time: Current time (s) – used only if chief_trajectory is given
            chief_state: Chief state (6,) at current_time – if both chief_trajectory
                         and chief_state are None, the chief state is assumed constant
                         and must have been set externally (e.g., during initialization)

        Returns:
            STM (6x6) matrix
        """
        # Determine the chief state at which to linearize
        if self.chief_trajectory is not None:
            # Use midpoint to get better average
            t_mid = current_time + dt / 2.0
            xc = self.chief_trajectory(t_mid)
        elif chief_state is not None:
            xc = chief_state
        else:
            raise ValueError("Either chief_trajectory or chief_state must be provided")

        # Compute linearized dynamics matrix A at xc
        A = self._linearized_matrix(xc)

        # Compute matrix exponential
        Phi = expm(A * dt)
        return Phi.astype(np.float64)

    def _linearized_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        [SI 物理域适配] 
        在真实的米/秒尺度下计算连续时间相对动力学雅可比矩阵 (Jacobian Matrix A)
        """
        r = state[:3]
        
        # 提取物理常数
        mu = self.crtbp.mu
        L = self.crtbp.L
        omega = self.crtbp.omega
        
        # 还原真实的引力常数参数 (由开普勒第三定律 GM = omega^2 * L^3 导出)
        GM_total = (omega**2) * (L**3)
        GM1 = (1.0 - mu) * GM_total
        GM2 = mu * GM_total
        x1 = -mu * L
        x2 = (1.0 - mu) * L

        x = r[0]; y = r[1]; z = r[2]
        rx1 = x - x1
        rx2 = x - x2
        
        r1_sq = rx1**2 + y**2 + z**2
        r2_sq = rx2**2 + y**2 + z**2
        
        r1_5 = r1_sq**2.5 + 1e-30 # 防止除零
        r2_5 = r2_sq**2.5 + 1e-30
        r1_3 = r1_sq**1.5 + 1e-30
        r2_3 = r2_sq**1.5 + 1e-30

        # 计算真实的引力梯度张量 (Gravity Gradient Tensor, SI 单位)
        # 注意: omega**2 替代了 1.0; GM1 和 GM2 替代了 (1-mu) 和 mu
        Uxx = omega**2 - GM1/r1_3 + 3*GM1*rx1**2/r1_5 - GM2/r2_3 + 3*GM2*rx2**2/r2_5
        Uyy = omega**2 - GM1/r1_3 + 3*GM1*y**2/r1_5   - GM2/r2_3 + 3*GM2*y**2/r2_5
        Uzz =          - GM1/r1_3 + 3*GM1*z**2/r1_5   - GM2/r2_3 + 3*GM2*z**2/r2_5
        
        Uxy = 3*GM1*rx1*y/r1_5 + 3*GM2*rx2*y/r2_5
        Uxz = 3*GM1*rx1*z/r1_5 + 3*GM2*rx2*z/r2_5
        Uyz = 3*GM1*y*z/r1_5   + 3*GM2*y*z/r2_5

        # 组装 6x6 状态矩阵 A
        A = np.zeros((6, 6), dtype=np.float64)
        
        # 1. 速度积分项
        A[0:3, 3:6] = np.eye(3)
        
        # 2. 引力梯度与离心力项
        A[3, 0], A[3, 1], A[3, 2] = Uxx, Uxy, Uxz
        A[4, 0], A[4, 1], A[4, 2] = Uxy, Uyy, Uyz
        A[5, 0], A[5, 1], A[5, 2] = Uxz, Uyz, Uzz
        
        # 3. 科里奥利力项
        A[3, 4] =  2.0 * omega
        A[4, 3] = -2.0 * omega
        
        return A

    def predict_state(self, current_state: np.ndarray, stm: np.ndarray) -> np.ndarray:
        """Apply STM to current state."""
        return stm @ current_state
