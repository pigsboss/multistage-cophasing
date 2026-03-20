# mission_sim/utils/math_tools.py
import numpy as np
import scipy.linalg

def get_lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    计算连续时间线性二次型调节器 (LQR) 的最优反馈增益矩阵 K。
    核心数学：求解连续代数黎卡提方程 (CARE)
    A^T P + P A - P B R^-1 B^T P + Q = 0
    
    :param A: 状态矩阵 (nxn)
    :param B: 控制输入矩阵 (nxm)
    :param Q: 状态惩罚权重矩阵 (nxn)，半正定
    :param R: 控制惩罚权重矩阵 (mxm)，正定
    :return: 最优反馈增益矩阵 K (mxn)，使得 u(t) = -K * e(t)
    """
    # 使用 scipy 底层高度优化的 LAPACK 库求解黎卡提方程
    try:
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    except scipy.linalg.LinAlgError as e:
        raise RuntimeError(f"LQR 黎卡提方程求解失败，请检查 A, B 矩阵是否可镇定！底层错误: {e}")
        
    # 计算增益矩阵 K = R^-1 B^T P
    K = np.linalg.inv(R) @ B.T @ P
    return K


def absolute_to_lvlh(state_chief: np.ndarray, state_deputy: np.ndarray) -> np.ndarray:
    """
    【L2 级预留】将从星的绝对状态转换到以主星为原点的 LVLH (局部垂直/局部水平) 相对坐标系中。
    这是后续进行星间编队共相控制的纯数学基石。
    
    LVLH 坐标系定义 (标准径向-沿迹-法向体系):
      - X轴 (径向 Radial) : 沿主星位置矢量方向 (从中心天体指向主星)
      - Z轴 (法向 Cross-track) : 沿主星轨道角动量方向 (右手定则 r x v)
      - Y轴 (沿迹 Along-track) : 补全右手坐标系 (Z x X)
    
    :param state_chief: 主星绝对状态 [x, y, z, vx, vy, vz]
    :param state_deputy: 从星绝对状态 [x, y, z, vx, vy, vz]
    :return: 相对状态 [x_rel, y_rel, z_rel, vx_rel, vy_rel, vz_rel]
    """
    r_c = state_chief[0:3]
    v_c = state_chief[3:6]
    r_d = state_deputy[0:3]
    v_d = state_deputy[3:6]

    # --- 1. 计算 LVLH 旋转矩阵的基向量 ---
    norm_r = np.linalg.norm(r_c)
    i_hat = r_c / norm_r
    
    h_vec = np.cross(r_c, v_c)
    norm_h = np.linalg.norm(h_vec)
    k_hat = h_vec / norm_h
    
    j_hat = np.cross(k_hat, i_hat)

    # 旋转矩阵 C_I2L (从惯性/绝对系到 LVLH 系)
    C_I2L = np.vstack([i_hat, j_hat, k_hat])
    
    # --- 2. 相对位置映射 ---
    delta_r_I = r_d - r_c
    rel_pos = C_I2L @ delta_r_I
    
    # --- 3. 相对速度映射 (剔除坐标系旋转引起的科氏项) ---
    # 主星轨道角速度标量 omega = |h| / |r|^2
    omega_mag = norm_h / (norm_r**2)
    # LVLH 系下的角速度向量始终在 Z 轴上
    omega_vec_L = np.array([0.0, 0.0, omega_mag])
    
    delta_v_I = v_d - v_c
    # 物理学中的牵连速度剔除：v_rel = v_abs - omega x r_rel
    rel_vel = C_I2L @ delta_v_I - np.cross(omega_vec_L, rel_pos)
    
    return np.concatenate([rel_pos, rel_vel])
