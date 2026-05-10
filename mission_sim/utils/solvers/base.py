"""
Hey! This module solves Kepler's equation and converts orbital elements 
to Cartesian states in batch, using Numba for just-in-time compilation 
and parallel acceleration. Great for large-scale orbit calculations.

Note: All functions only work for elliptical orbits (0 ≤ e < 1).
"""

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _kepler_elliptic(M, e, tol=1e-12, max_iter=50):
    """
    求解椭圆开普勒方程 M = E - e sin(E)，返回偏近点角 E。
    支持向量化：M 和 e 为同长度数组，内部使用 prange 并行。

    :param M: 平近点角数组 (N,) 弧度
    :param e: 偏心率数组 (N,)
    :param tol: 牛顿法收敛容差
    :param max_iter: 最大迭代次数
    :return: 偏近点角数组 E (N,)
    """
    n = M.shape[0]
    E = np.empty_like(M)
    for i in prange(n):
        Ei = M[i]
        for _ in range(max_iter):
            sin_Ei = np.sin(Ei)
            cos_Ei = np.cos(Ei)
            dE = (Ei - e[i] * sin_Ei - M[i]) / (1.0 - e[i] * cos_Ei)
            Ei -= dE
            if abs(dE) < tol:
                break
        E[i] = Ei
    return E

@njit(parallel=True)
def kepler_elements_to_cartesian_batch(a_arr, e_arr, i_arr, Omega_arr, omega_arr, M_arr, mu):
    """
    批量将轨道根数转换为笛卡尔状态向量 [x, y, z, vx, vy, vz]。
    所有输入为一维数组，长度相同。

    :param a_arr: 半长轴 (m) 数组 (N,)
    :param e_arr: 偏心率数组 (N,)
    :param i_arr: 倾角 (rad) 数组 (N,)
    :param Omega_arr: 升交点赤经 (rad) 数组 (N,)
    :param omega_arr: 近地点幅角 (rad) 数组 (N,)
    :param M_arr: 平近点角 (rad) 数组 (N,)
    :param mu: 引力常数 (m³/s²) 标量
    :return: 状态向量数组 (N, 6)，每行为 [x, y, z, vx, vy, vz]
    """
    n = a_arr.shape[0]
    states = np.empty((n, 6))

    # 1. 求解偏近点角
    E_arr = _kepler_elliptic(M_arr, e_arr)

    cos_E = np.cos(E_arr)
    sin_E = np.sin(E_arr)

    # 2. 真近点角
    sqrt_1pe = np.sqrt(1.0 + e_arr)
    sqrt_1me = np.sqrt(1.0 - e_arr)
    # tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    nu = 2.0 * np.arctan2(sqrt_1pe * np.sin(0.5 * E_arr),
                          sqrt_1me * np.cos(0.5 * E_arr))

    # 3. 向径
    r = a_arr * (1.0 - e_arr * cos_E)

    # 4. 轨道面内速度
    sqrt_1me2 = np.sqrt(1.0 - e_arr * e_arr)
    n_motion = np.sqrt(mu * a_arr) / r
    vx_orb = -n_motion * sin_E
    vy_orb = n_motion * sqrt_1me2 * cos_E

    # 5. 旋转矩阵 (Omega, i, omega)
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)
    x_orb = r * cos_nu
    y_orb = r * sin_nu

    for idx in prange(n):
        cos_Om = np.cos(Omega_arr[idx])
        sin_Om = np.sin(Omega_arr[idx])
        cos_i  = np.cos(i_arr[idx])
        sin_i  = np.sin(i_arr[idx])
        cos_om = np.cos(omega_arr[idx])
        sin_om = np.sin(omega_arr[idx])

        # 旋转位置向量
        x1 = x_orb[idx] * cos_om - y_orb[idx] * sin_om
        y1 = x_orb[idx] * sin_om + y_orb[idx] * cos_om
        y2 = y1 * cos_i
        z1 = y1 * sin_i
        x3 = x1 * cos_Om - y2 * sin_Om
        y3 = x1 * sin_Om + y2 * cos_Om
        z3 = z1

        # 旋转速度向量
        vx1 = vx_orb[idx] * cos_om - vy_orb[idx] * sin_om
        vy1 = vx_orb[idx] * sin_om + vy_orb[idx] * cos_om
        vy2 = vy1 * cos_i
        vz1 = vy1 * sin_i
        vx3 = vx1 * cos_Om - vy2 * sin_Om
        vy3 = vx1 * sin_Om + vy2 * cos_Om
        vz3 = vz1

        states[idx, 0] = x3
        states[idx, 1] = y3
        states[idx, 2] = z3
        states[idx, 3] = vx3
        states[idx, 4] = vy3
        states[idx, 5] = vz3

    return states
