"""
UniversalCRTBP 严格单元测试

验证 CRTBP 模型的数值精度和物理正确性。
Sprint 2 核心验证：雅可比常数守恒精度 < 1e-10
"""
import pytest
import numpy as np
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


class TestCRTBPConservation:
    """测试 CRTBP 守恒量"""
    
    def test_jacobi_constant_conservation_100_days(self):
        """验证100天积分内雅可比常数守恒"""
        crtbp = UniversalCRTBP.earth_moon_system()
        
        # 初始状态：靠近 L1 的晕轨道近似
        x0 = np.array([0.85 * crtbp.distance, 0.0, 0.05 * crtbp.distance,
                       0.0, 0.15 * 1e3, 0.0])
        
        # 计算初始雅可比常数
        C0 = crtbp.jacobi_constant(x0)
        
        # 简化的 RK4 积分（1000步）
        dt = 100 * 86400 / 1000  # 100天分1000步
        x = x0.copy()
        
        for _ in range(1000):
            # RK4 积分（使用 CRTBP 加速度）
            k1 = self._crtbp_derivative(crtbp, x)
            k2 = self._crtbp_derivative(crtbp, x + 0.5*dt*k1)
            k3 = self._crtbp_derivative(crtbp, x + 0.5*dt*k2)
            k4 = self._crtbp_derivative(crtbp, x + dt*k3)
            x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 计算最终雅可比常数
        C_final = crtbp.jacobi_constant(x)
        
        # 验证守恒精度 < 1e-10（相对）
        relative_error = abs(C_final - C0) / abs(C0)
        assert relative_error < 1e-10, f"雅可比常数漂移: {relative_error:.2e}"
    
    def _crtbp_derivative(self, crtbp: UniversalCRTBP, state: np.ndarray) -> np.ndarray:
        """计算 CRTBP 状态导数（用于积分测试）"""
        pos = state[0:3]
        vel = state[3:6]
        
        # 获取加速度
        accel = crtbp.compute_accel(state, epoch=0.0)
        
        return np.concatenate([vel, accel])
    
    def test_lagrange_points_accuracy(self):
        """验证拉格朗日点计算精度"""
        crtbp = UniversalCRTBP.earth_moon_system()
        
        # 理论值（地月系统）
        mu = crtbp.mu
        # L1 近似位置（无量纲）
        gamma_L1 = (mu/3)**(1/3)
        L1_theory = 1 - mu - gamma_L1
        
        # 获取计算的 L1
        lagrange_points = crtbp.get_lagrange_points_nd()
        L1_computed = lagrange_points['L1'][0]  # x坐标
        
        # 验证相对误差 < 1%
        relative_error = abs(L1_computed - L1_theory) / abs(L1_theory)
        assert relative_error < 0.01, f"L1位置误差: {relative_error:.2e}"


class TestCRTBPReferenceSolutions:
    """与参考解对比"""
    
    def test_halo_orbit_approximation(self):
        """验证与 Richardson 三阶 Halo 轨道近似解的一致性"""
        # TODO: 实现与解析近似解的对比
        pytest.skip("Halo轨道近似解对比待实现")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
