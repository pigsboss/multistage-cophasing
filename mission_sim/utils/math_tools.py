import numpy as np
from scipy.linalg import solve_continuous_are

def get_lqr_gain(A, B, Q, R):
    """求解连续时间 LQR 增益矩阵 K"""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K
