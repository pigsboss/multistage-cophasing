import numpy as np
import pytest
from mission_sim.utils.math_tools import (
    get_lqr_gain,
    absolute_to_lvlh,
    elements_to_cartesian,
    inertial_to_rotating,
    rotating_to_inertial,
)
from mission_sim.utils.differential_correction import jacobi_constant


def test_get_lqr_gain():
    """测试 LQR 增益计算（简单双积分器）"""
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)
    K = get_lqr_gain(A, B, Q, R)
    # 双积分器 LQR 增益应为 [1, sqrt(3)]? 这里只验证不崩溃
    assert K.shape == (1, 2)

def test_absolute_to_lvlh():
    """Test absolute to LVLH transformation with L2 rigorous math tools."""
    import numpy as np
    from mission_sim.utils.math_tools import absolute_to_lvlh
    
    # Chief in circular orbit
    r = 7000e3
    v = np.sqrt(3.986004418e14 / r)
    
    # Split into explicit 3D position and velocity vectors for Chief
    r_chief = np.array([r, 0.0, 0.0])
    v_chief = np.array([0.0, v, 0.0])
    
    # Deputy is 100m ahead of the Chief (in the absolute Y direction)
    r_deputy = np.array([r, 100.0, 0.0])
    v_deputy = np.array([0.0, v, 0.0])
    
    # Execute the new L2 contract (4 separate arguments)
    rho_lvlh, rho_dot_lvlh = absolute_to_lvlh(r_chief, v_chief, r_deputy, v_deputy)
    
    # Verify the relative position
    # Under our LVLH convention (Z=Radial, Y=Cross-track, X=Along-track):
    # Absolute Y aligns with Chief's velocity, so it becomes the LVLH X-axis.
    # Therefore, the +100m offset should perfectly map to [100.0, 0.0, 0.0] in LVLH.
    from numpy.testing import assert_allclose
    assert_allclose(rho_lvlh, [100.0, 0.0, 0.0], atol=1e-7)

def test_elements_to_cartesian():
    """测试轨道根数转笛卡尔坐标（开普勒圆轨道）"""
    mu = 3.986004418e14
    a = 7000e3
    e = 0.0
    i = 0.0
    Omega = 0.0
    omega = 0.0
    M = 0.0
    state = elements_to_cartesian(mu, a, e, i, Omega, omega, M)

    # 验证位置大小接近 a
    r = np.linalg.norm(state[:3])
    assert abs(r - a) < 1.0
    # 验证速度大小接近 sqrt(mu/a)
    v = np.linalg.norm(state[3:])
    expected_v = np.sqrt(mu / a)
    assert abs(v - expected_v) < 1e-6
    # 验证轨道面法向与倾角一致（倾角为0，角动量应沿Z轴）
    h = np.cross(state[:3], state[3:])
    assert np.allclose(h / np.linalg.norm(h), [0, 0, 1], atol=1e-6)


def test_inertial_to_rotating_consistency():
    """测试惯性系与旋转系转换的一致性"""
    omega = 1.990986e-7  # 日地系统角速度
    t = 86400.0  # 一天后

    # 初始惯性系状态（例如日地旋转系中某点转换到惯性系）
    # 为了方便，我们先在旋转系定义一个简单状态，然后转到惯性系，再转回，验证往返误差
    state_rot = np.array([1.0e11, 0.0, 0.0, 0.0, 1.0e4, 0.0])
    # 转换到惯性系
    state_inertial = rotating_to_inertial(state_rot, t, omega)
    # 再转回旋转系
    state_rot2 = inertial_to_rotating(state_inertial, t, omega)

    # 往返误差应很小
    assert np.allclose(state_rot, state_rot2, rtol=1e-10, atol=1e-6)

    # 测试恒速旋转系中的静止物体
    # 在旋转系中静止的物体，在惯性系中应做匀速圆周运动
    pos_rot = np.array([1.0e11, 0.0, 0.0])
    vel_rot = np.array([0.0, 0.0, 0.0])
    state_rot_static = np.concatenate([pos_rot, vel_rot])
    state_inertial = rotating_to_inertial(state_rot_static, t, omega)

    # 惯性系中的位置应旋转了角度 omega*t
    theta = omega * t
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_expected = pos_rot[0] * cos_theta - pos_rot[1] * sin_theta
    y_expected = pos_rot[0] * sin_theta + pos_rot[1] * cos_theta
    assert np.allclose(state_inertial[0], x_expected, rtol=1e-10)
    assert np.allclose(state_inertial[1], y_expected, rtol=1e-10)
    # 速度应满足圆周运动
    vx_expected = -omega * y_expected
    vy_expected = omega * x_expected
    assert np.allclose(state_inertial[3], vx_expected, rtol=1e-10)
    assert np.allclose(state_inertial[4], vy_expected, rtol=1e-10)
    assert state_inertial[2] == 0.0
    assert state_inertial[5] == 0.0


def test_rotating_to_inertial_consistency():
    """测试旋转系到惯性系转换的逆变换一致性"""
    omega = 1.990986e-7
    t = 3600.0
    state_inertial = np.array([1.0e11, 0.0, 0.0, 0.0, 1.0e4, 0.0])
    state_rot = inertial_to_rotating(state_inertial, t, omega)
    state_inertial2 = rotating_to_inertial(state_rot, t, omega)
    assert np.allclose(state_inertial, state_inertial2, rtol=1e-10, atol=1e-6)


def test_jacobi_constant():
    """测试雅可比常数计算"""
    state_nd = np.array([1.01106, 0.0, 0.05, 0.0, 0.0105, 0.0])
    mu = 3.00348e-6
    C = jacobi_constant(state_nd, mu)
    assert isinstance(C, float)
    assert not np.isnan(C)
