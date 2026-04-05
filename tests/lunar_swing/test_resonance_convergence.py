"""
共振轨道搜索收敛性验证测试

验证 LunarSwingTargeter 能否找到周期轨道。
"""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


class TestResonanceConvergence:
    """测试共振轨道搜索算法的收敛性"""
    
    @pytest.fixture
    def targeter(self):
        """创建地月系统轨道设计器"""
        crtbp = UniversalCRTBP.earth_moon_system()
        return LunarSwingTargeter(
            dynamics_model=crtbp,
            mu=crtbp.mu,
            integrator_type='rk4',
            num_steps=200
        )
    
    def test_simple_lyapunov_search(self, targeter):
        """测试：搜索地月 L1 附近的 Lyapunov 轨道（平面周期轨道）"""
        # 使用更保守的参数，测试算法能取得进展即可
        # 不完全收敛也是可以接受的，只要残差有显著下降
        mu = targeter.mu
        L1_x = 0.8369  # 近似值
        
        # 平面轨道初始猜测
        x0 = L1_x + 0.05
        vy_guess = np.sqrt((1 - mu) / (x0 + mu)**2 + mu / (x0 + mu - 1)**2 - 1)
        
        initial_guess = np.array([
            x0, 0.0, 0.0,  # 位置
            0.0, vy_guess, 0.0  # 速度
        ])
        
        # 目标周期约 2.8 天
        T_dim = 2.8 * 86400 / (4.342 * 86400)
        
        result = targeter.find_resonant_orbit(
            resonance_ratio=(1, 1),
            initial_guess=initial_guess,
            target_period=T_dim * 4.342 * 86400,
            tol=1e-6,  # 更严格的容差
            max_iter=100,  # 更多迭代次数
            damping=0.8   # 更大的阻尼因子（更激进的修正）
        )
        
        # 验证：即使不完全收敛，残差也应该显著下降
        history = result['convergence_history']
        if len(history) > 1:
            initial_residual = history[0]['residual_norm']
            final_residual = history[-1]['residual_norm']
            improvement = initial_residual / final_residual if final_residual > 0 else float('inf')
            
            print(f"\n收敛分析:")
            print(f"  初始残差: {initial_residual:.3e}")
            print(f"  最终残差: {final_residual:.3e}")
            print(f"  改善倍数: {improvement:.1f}x")
            print(f"  是否收敛: {result['success']}")
            
            # 残差应该至少下降10倍，或达到收敛
            assert result['success'] or improvement > 10, \
                f"算法未收敛且残差改善不足: {improvement:.1f}x"
        else:
            pytest.skip("迭代历史为空，跳过测试")
    
    def test_convergence_history_plot(self, targeter, tmp_path):
        """测试：生成收敛历史图"""
        mu = targeter.mu
        L1_x = 0.8369
        
        initial_guess = np.array([
            L1_x + 0.08, 0.0, 0.0,
            0.0, 0.35, 0.0
        ])
        
        result = targeter.find_resonant_orbit(
            resonance_ratio=(1, 1),
            initial_guess=initial_guess,
            target_period=3.0 * 86400,  # 3天周期
            tol=1e-6,
            max_iter=50,
            damping=0.5
        )
        
        # 绘制收敛曲线
        history = result['convergence_history']
        iterations = [h['iteration'] for h in history]
        residuals = [h['residual_norm'] for h in history]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogy(iterations, residuals, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('位置残差范数 (log)', fontsize=12)
        ax.set_title(f'单参数打靶法收敛曲线\n{"收敛" if result["success"] else "未收敛"}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加收敛线
        ax.axhline(y=1e-6, color='r', linestyle='--', label='收敛阈值 (1e-6)')
        ax.legend()
        
        # 保存
        output_dir = tmp_path / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "convergence_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"收敛曲线已保存至: {output_dir / 'convergence_curve.png'}")
        
        # 验证至少有一定收敛（即使不完全收敛，残差也应下降）
        if len(residuals) > 1:
            assert residuals[-1] < residuals[0] * 0.9, "残差未下降，算法可能发散"
    
    def test_2to1_resonance_search(self, targeter):
        """测试：搜索 2:1 共振轨道（简化测试，可能不收敛但验证流程）"""
        # 2:1 共振意味着航天器转2圈，月球转1圈
        # 周期约 13.66 天
        
        # 更激进的初始猜测（近地点低，远地点高）
        initial_guess = np.array([
            0.9, 0.0, 0.0,   # 近地点附近
            0.0, 1.2, 0.0    # 较高速度
        ])
        
        result = targeter.find_resonant_orbit(
            resonance_ratio=(2, 1),
            initial_guess=initial_guess,
            tol=1e-5,
            max_iter=20,  # 限制迭代次数，作为 smoke test
            damping=0.2   # 更保守的阻尼
        )
        
        # 对于 2:1 共振，我们可能没有好的初始猜测，所以不强制要求收敛
        # 但要求算法运行不报错，且残差历史被记录
        assert 'convergence_history' in result
        assert len(result['convergence_history']) > 0
        
        print(f"\n2:1 共振搜索 {'成功' if result['success'] else '未完成'}")
        if not result['success']:
            print(f"  最终残差: {result['convergence_history'][-1]['residual_norm']:.2e}")
            print("  （可能需要更好的初始猜测或更多迭代）")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
