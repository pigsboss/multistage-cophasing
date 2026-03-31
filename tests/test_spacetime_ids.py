import pytest
import numpy as np
from mission_sim.core.spacetime.ids import CoordinateFrame, Telecommand, Telemetry, FormationState

def test_coordinate_frame_enum():
    """测试坐标系枚举定义 (L1与L2级基准)"""
    assert isinstance(CoordinateFrame.J2000_ECI, CoordinateFrame)
    assert isinstance(CoordinateFrame.LVLH, CoordinateFrame)
    assert CoordinateFrame.LVLH in CoordinateFrame

def test_telecommand_creation():
    """测试 Telecommand (控制指令包) 的创建与属性"""
    force = np.array([0.5, 0.0, 0.0])  # 0.5 N 的推力
    cmd = Telecommand(
        force_vector=force, 
        frame=CoordinateFrame.LVLH, 
        duration_s=1.5, 
        actuator_id="THR_MAIN"
    )
    
    assert np.array_equal(cmd.force_vector, force)
    assert cmd.frame == CoordinateFrame.LVLH
    assert cmd.duration_s == 1.5
    assert cmd.actuator_id == "THR_MAIN"

def test_telemetry_creation():
    """测试 Telemetry (遥测/观测数据包) 的创建"""
    pos = np.array([7000e3, 0, 0])
    vel = np.array([0, 7.5e3, 0])
    telem = Telemetry(
        position=pos, 
        velocity=vel, 
        frame=CoordinateFrame.J2000_ECI, 
        timestamp=3600.0
    )
    
    assert np.array_equal(telem.position, pos)
    assert np.array_equal(telem.velocity, vel)
    assert telem.timestamp == 3600.0

def test_formation_state_container():
    """测试 L2 级核心数据总线：多星编队状态容器"""
    chief_pos = np.array([1.5e11, 0, 0])
    chief_vel = np.array([0, 30e3, 0])
    
    # 1. 初始化主星状态
    state = FormationState(
        timestamp=0.0,
        chief_position=chief_pos,
        chief_velocity=chief_vel,
        chief_frame=CoordinateFrame.SUN_EARTH_ROTATING
    )
    
    assert state.get_num_deputies() == 0
    assert state.chief_frame == CoordinateFrame.SUN_EARTH_ROTATING
    
    # 2. 挂载从星的相对状态 (例如 LVLH 系下距离主星 100 米)
    dep_pos = np.array([100.0, 0.0, 0.0])
    dep_vel = np.array([0.0, 0.1, 0.0])
    state.add_deputy_state(dep_pos, dep_vel)
    
    assert state.get_num_deputies() == 1
    assert np.array_equal(state.deputy_relative_positions[0], dep_pos)
    assert state.deputy_frame == CoordinateFrame.LVLH  # 默认必须是相对系
