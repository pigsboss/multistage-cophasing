import pytest
import os
import numpy as np
from mission_sim.simulation.threebody.sun_earth_l2 import SunEarthL2L1Simulation
from mission_sim.simulation.twobody.leo import LEOL1Simulation
from mission_sim.simulation.twobody.geo import GEOL1Simulation
from mission_sim.utils.logger import HDF5Logger


def test_short_simulation_sun_earth_l2(temp_dir, default_config):
    """运行简短仿真（1 天）验证集成（日地 L2 点）"""
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 60.0
    config["enable_visualization"] = False

    sim = SunEarthL2L1Simulation(config)
    success = sim.run()
    assert success

    # 检查输出文件
    h5_file = sim.h5_file
    assert os.path.exists(h5_file)
    # 检查燃料账单
    fuel_bill = os.path.join(temp_dir, f"fuel_bill_{sim.mission_id}.csv")
    assert os.path.exists(fuel_bill)


def test_leo_simulation(temp_dir, default_config):
    """运行 LEO 仿真（1 天）验证集成"""
    # 修改配置为 LEO 所需参数
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 10.0
    config["enable_visualization"] = False
    # LEO 特定配置（使用默认的 7000km 圆轨道）
    config["elements"] = [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0]
    config["spacecraft_mass"] = 1000.0
    config["area_to_mass"] = 0.02
    config["Cd"] = 2.2
    config["enable_atmospheric_drag"] = True
    config["enable_j2"] = True

    sim = LEOL1Simulation(config)
    success = sim.run()
    assert success

    # 检查输出文件
    h5_file = sim.h5_file
    assert os.path.exists(h5_file)

    # 检查燃料账单
    fuel_bill = os.path.join(temp_dir, f"fuel_bill_{sim.mission_id}.csv")
    assert os.path.exists(fuel_bill)

    # 验证燃料消耗在合理范围（LEO 约 0.1~0.5 m/s/天）
    with open(fuel_bill, 'r') as f:
        lines = f.readlines()
        # 跳过表头
        data = lines[1].strip().split(',')
        total_dv = float(data[1])   # total_dv_mps
        avg_dv_per_day = float(data[2])  # avg_dv_per_day_mps
        assert 0.05 < avg_dv_per_day < 0.8, f"LEO 燃料消耗异常: {avg_dv_per_day:.4f} m/s/天"
        # 可选：更严格的量级检查
        assert total_dv > 0.0

    # 可选：检查 HDF5 中的控制力数据非空
    with HDF5Logger(h5_file) as logger:
        forces = logger.load_data('control_forces')
        assert len(forces) > 0


def test_geo_simulation(temp_dir, default_config):
    """运行 GEO 仿真（1 天）验证集成"""
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 60.0
    config["enable_visualization"] = False
    # GEO 特定配置（使用标准 GEO 半径）
    config["elements"] = [42164000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config["spacecraft_mass"] = 2000.0
    config["enable_atmospheric_drag"] = False
    config["enable_j2"] = True

    sim = GEOL1Simulation(config)
    success = sim.run()
    assert success

    # 检查输出文件
    h5_file = sim.h5_file
    assert os.path.exists(h5_file)

    fuel_bill = os.path.join(temp_dir, f"fuel_bill_{sim.mission_id}.csv")
    assert os.path.exists(fuel_bill)

    # 验证燃料消耗在合理范围（GEO 约 0.01~0.05 m/s/天）
    with open(fuel_bill, 'r') as f:
        lines = f.readlines()
        data = lines[1].strip().split(',')
        avg_dv_per_day = float(data[2])
        assert 0.005 < avg_dv_per_day < 0.1, f"GEO 燃料消耗异常: {avg_dv_per_day:.4f} m/s/天"


def test_rk45_integrator_sun_earth_l2(temp_dir, default_config):
    """测试 RK45 变步长积分器与 RK4 结果的一致性（日地 L2 点）"""
    # 先运行 RK4 仿真
    config_rk4 = default_config.copy()
    config_rk4["data_dir"] = str(temp_dir)
    config_rk4["simulation_days"] = 0.1  # 2.4 小时，快速测试
    config_rk4["time_step"] = 60.0
    config_rk4["enable_visualization"] = False
    config_rk4["integrator"] = "rk4"
    sim_rk4 = SunEarthL2L1Simulation(config_rk4)
    success = sim_rk4.run()
    assert success

    # 运行 RK45 仿真
    config_rk45 = config_rk4.copy()
    config_rk45["integrator"] = "rk45"
    config_rk45["integrator_rtol"] = 1e-9
    config_rk45["integrator_atol"] = 1e-12
    # 修改 mission_name 避免文件名冲突
    config_rk45["mission_name"] = "RK45_Test"
    sim_rk45 = SunEarthL2L1Simulation(config_rk45)
    success = sim_rk45.run()
    assert success

    # 加载两个仿真的最终状态
    with HDF5Logger(sim_rk4.h5_file) as logger:
        states_rk4 = logger.load_data('true_states')
        final_rk4 = states_rk4[-1]
    with HDF5Logger(sim_rk45.h5_file) as logger:
        states_rk45 = logger.load_data('true_states')
        final_rk45 = states_rk45[-1]

    # 验证最终状态差异较小（允许的绝对误差取决于仿真时长）
    pos_diff = np.linalg.norm(final_rk4[:3] - final_rk45[:3])
    vel_diff = np.linalg.norm(final_rk4[3:] - final_rk45[3:])
    # 对于 2.4 小时仿真，位置差异应小于 1km，速度差异小于 0.1 m/s
    assert pos_diff < 1000.0, f"位置差异过大: {pos_diff:.2f} m"
    assert vel_diff < 0.1, f"速度差异过大: {vel_diff:.2f} m/s"

    # 也可验证整个轨迹的 RMS 差异
    # 由于时间点可能不同，简单对比终点已足够