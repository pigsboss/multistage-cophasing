#!/usr/bin/env python3
"""
MCPC L2 级多星编队仿真启动脚本 (SI 单位制修正版)
"""

import numpy as np
import os
from mission_sim.simulation.formation_simulation import FormationSimulation
from mission_sim.core.spacetime.ids import CoordinateFrame
# 关键修正：lvl -> lvlh
from mission_sim.utils.math_tools import lvlh_to_absolute, get_lqr_gain
from mission_sim.core.cyber.models.threebody.base import CRTBP
from mission_sim.core.cyber.models.crtbp_relative_dynamics import CRTBPRelativeDynamics

# 1. 定义物理环境参数
mu = 3.00348e-6
L = 1.495978707e11
omega = 1.990986e-7
crtbp = CRTBP(mu, L, omega)

# 2. 定义主星初始状态
chief0 = np.array([
    1.50613280e11,
    0.0,
    1.20000000e8,
    0.0,
    1.51320000e2,
    0.0
])

# 3. 定义从星目标构型 (LVLH 坐标, m)
rel_targets = {
    "DEP1": np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
    "DEP2": np.array([-50.0, 0.0, 50.0, 0.0, 0.0, 0.0]),
}

# 4. 计算从星初始绝对状态
deputy_initial_configs = []
for dep_id, rel_state in rel_targets.items():
    # 修正后的调用
    r_abs, v_abs = lvlh_to_absolute(chief0[:3], chief0[3:], rel_state[:3], rel_state[3:])
    deputy_initial_configs.append((dep_id, np.concatenate([r_abs, v_abs])))

# 5. 预计算 LQR 控制增益 K
temp_dynamics = CRTBPRelativeDynamics(crtbp)
A_cont = temp_dynamics._linearized_matrix(chief0)
B = np.zeros((6, 3))
B[3:6, 0:3] = np.eye(3)

Q = np.diag([1e-6, 1e-6, 1e-6, 1.0, 1.0, 1.0]) 
R = np.diag([1e8, 1e8, 1e8])                    
K = get_lqr_gain(A_cont, B, Q, R)

# 6. 构建仿真配置
config = {
    "mission_name": "Halo_Formation_CRTBP",
    "simulation_days": 1,
    "time_step": 10.0,
    "data_dir": "data",
    "verbose": True,
    "chief_initial_state": chief0.tolist(),
    "deputy_initial_states": deputy_initial_configs,
    "formation_targets": rel_targets, 
    "mu": mu,
    "L": L,
    "omega": omega,
    "lqr_gain": K,
    "thruster_max_thrust_n": 5.0,
}

if __name__ == "__main__":
    if not os.path.exists(config["data_dir"]):
        os.makedirs(config["data_dir"])
    sim = FormationSimulation(config)
    sim.run()
