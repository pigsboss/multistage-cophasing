# tests/test_integration.py
import pytest
import os
import numpy as np
import h5py
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

    h5_file = sim.h5_file
    assert os.path.exists(h5_file)
    fuel_bill = os.path.join(temp_dir, f"fuel_bill_{sim.mission_id}.csv")
    assert os.path.exists(fuel_bill)


def test_leo_simulation_no_control(temp_dir, default_config):
    """无控模式下验证 LEO 标称轨道：航天器应精确跟随标称轨道（误差很小）"""
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 10.0
    config["enable_visualization"] = False
    config["elements"] = [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0]
    config["spacecraft_mass"] = 1000.0
    config["enable_atmospheric_drag"] = False
    config["enable_j2"] = False
    config["use_j2_generator"] = False          # 使用开普勒生成器
    config["control_gain_scale"] = 0.0          # 禁用控制

    sim = LEOL1Simulation(config)
    success = sim.run()
    assert success

    h5_file = sim.h5_file
    with h5py.File(h5_file, 'r') as f:
        errors = f['tracking_errors'][:]
        pos_errors = np.linalg.norm(errors[:, 0:3], axis=1)
        vel_errors = np.linalg.norm(errors[:, 3:6], axis=1)

    max_pos_error = np.max(pos_errors)
    max_vel_error = np.max(vel_errors)
    assert max_pos_error < 1000.0, f"无控模式位置误差过大: {max_pos_error:.2f} m"
    assert max_vel_error < 10.0, f"无控模式速度误差过大: {max_vel_error:.4f} m/s"


def test_leo_simulation(temp_dir, default_config):
    """运行 LEO 仿真（1 天）验证集成（有控）"""
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 10.0
    config["enable_visualization"] = False
    config["elements"] = [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0]
    config["spacecraft_mass"] = 1000.0
    config["area_to_mass"] = 0.02
    config["Cd"] = 2.2
    config["enable_atmospheric_drag"] = True
    config["enable_j2"] = True
    config["use_j2_generator"] = False          # 临时禁用 J2 生成器
    config["control_gain_scale"] = 5e-9

    sim = LEOL1Simulation(config)
    success = sim.run()
    assert success

    h5_file = sim.h5_file
    assert os.path.exists(h5_file)

    fuel_bill = os.path.join(temp_dir, f"fuel_bill_{sim.mission_id}.csv")
    assert os.path.exists(fuel_bill)

    with open(fuel_bill, 'r') as f:
        lines = f.readlines()
        data = lines[1].strip().split(',')
        avg_dv_per_day = float(data[2])
        # 严格燃料消耗范围（后续需通过调整 LQR 权重或增益缩放实现）
        assert 0.05 < avg_dv_per_day < 0.8, f"LEO 燃料消耗异常: {avg_dv_per_day:.4f} m/s/天"


def test_geo_simulation(temp_dir, default_config):
    """运行 GEO 仿真（1 天）验证集成"""
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 60.0
    config["enable_visualization"] = False
    config["elements"] = [42164000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config["spacecraft_mass"] = 2000.0
    config["enable_atmospheric_drag"] = False
    config["enable_j2"] = True
    config["use_j2_generator"] = False
    config["control_gain_scale"] = 5e-9

    sim = GEOL1Simulation(config)
    success = sim.run()
    assert success

    h5_file = sim.h5_file
    assert os.path.exists(h5_file)

    fuel_bill = os.path.join(temp_dir, f"fuel_bill_{sim.mission_id}.csv")
    assert os.path.exists(fuel_bill)

    with open(fuel_bill, 'r') as f:
        lines = f.readlines()
        data = lines[1].strip().split(',')
        avg_dv_per_day = float(data[2])
        assert 0.005 < avg_dv_per_day < 0.1, f"GEO 燃料消耗异常: {avg_dv_per_day:.4f} m/s/天"


def test_rk45_integrator_sun_earth_l2(temp_dir, default_config):
    """测试 RK45 变步长积分器与 RK4 结果的一致性（日地 L2 点）"""
    config_rk4 = default_config.copy()
    config_rk4["data_dir"] = str(temp_dir)
    config_rk4["simulation_days"] = 0.1
    config_rk4["time_step"] = 60.0
    config_rk4["enable_visualization"] = False
    config_rk4["integrator"] = "rk4"
    sim_rk4 = SunEarthL2L1Simulation(config_rk4)
    success = sim_rk4.run()
    assert success

    config_rk45 = config_rk4.copy()
    config_rk45["integrator"] = "rk45"
    config_rk45["integrator_rtol"] = 1e-9
    config_rk45["integrator_atol"] = 1e-12
    config_rk45["mission_name"] = "RK45_Test"
    sim_rk45 = SunEarthL2L1Simulation(config_rk45)
    success = sim_rk45.run()
    assert success

    with HDF5Logger(sim_rk4.h5_file) as logger:
        states_rk4 = logger.load_data('true_states')
        final_rk4 = states_rk4[-1]
    with HDF5Logger(sim_rk45.h5_file) as logger:
        states_rk45 = logger.load_data('true_states')
        final_rk45 = states_rk45[-1]

    pos_diff = np.linalg.norm(final_rk4[:3] - final_rk45[:3])
    vel_diff = np.linalg.norm(final_rk4[3:] - final_rk45[3:])
    assert pos_diff < 1000.0, f"位置差异过大: {pos_diff:.2f} m"
    assert vel_diff < 0.1, f"速度差异过大: {vel_diff:.2f} m/s"
