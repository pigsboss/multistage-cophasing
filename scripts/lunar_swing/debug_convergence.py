"""
调试收敛问题的工具脚本
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP

def analyze_sensitivity_matrix(targeter, state, period_nd):
    """分析敏感度矩阵的特性"""
    _, stm = targeter._stm_calc.propagate_with_stm(
        dynamics=targeter._get_dynamics_func(),
        initial_state=state,
        t0=0.0,
        tf=period_nd,
        method=targeter.integrator_type,
        num_steps=targeter.num_steps
    )
    
    # 对于平面轨道，分析位置对速度的敏感度
    sensitivity = stm[0:2, 3:5]  # dx/dvx, dx/dvy; dy/dvx, dy/dvy
    
    print("\nSensitivity Matrix Analysis:")
    print(f"Matrix:\n{sensitivity}")
    
    # 计算行列式和条件数
    det = np.linalg.det(sensitivity)
    cond = np.linalg.cond(sensitivity)
    
    print(f"Determinant: {det:.6e}")
    print(f"Condition number: {cond:.6e}")
    
    # 特征值分析
    eigenvalues = np.linalg.eigvals(sensitivity)
    print(f"Eigenvalues: {eigenvalues}")
    
    return sensitivity, det, cond

def test_different_periods():
    """测试不同周期下的收敛性"""
    crtbp = UniversalCRTBP.earth_moon_system()
    mu = crtbp.mu
    
    # 月球周期
    T_moon = 27.321661 * 24 * 3600
    char_period = 4.342 * 86400
    
    periods_nd = []
    residuals = []
    
    for n, m in [(2, 1), (3, 2), (1, 1)]:
        target_period = (m / n) * T_moon
        period_nd = target_period / char_period
        periods_nd.append(period_nd)
        
        # 简单测试
        initial_guess = np.array([0.9, 0.0, 0.0, 0.0, 1.0, 0.0])
        
        targeter = LunarSwingTargeter(
            dynamics_model=crtbp,
            mu=mu,
            integrator_type='rk4',
            num_steps=1000
        )
        
        result = targeter.find_resonant_orbit(
            resonance_ratio=(n, m),
            initial_guess=initial_guess,
            target_period=target_period,
            max_iter=10,
            damping=0.5
        )
        
        if result['convergence_history']:
            final_res = result['convergence_history'][-1]['residual_norm']
        else:
            final_res = np.nan
            
        residuals.append(final_res)
        print(f"Resonance {n}:{m}, period_nd={period_nd:.3f}, residual={final_res:.2e}")
    
    return periods_nd, residuals

def main():
    print("Debugging Convergence Issues")
    print("=" * 60)
    
    # 1. 初始化
    crtbp = UniversalCRTBP.earth_moon_system()
    mu = crtbp.mu
    
    # 2. 测试敏感度矩阵
    print("\n1. Testing Sensitivity Matrix at Initial Guess")
    targeter = LunarSwingTargeter(
        dynamics_model=crtbp,
        mu=mu,
        integrator_type='rk4',
        num_steps=2000
    )
    
    # 2:1共振周期
    T_moon = 27.321661 * 24 * 3600
    period = (1 / 2) * T_moon
    period_nd = period / (4.342 * 86400)
    
    # 测试不同初始状态的敏感度
    test_states = [
        np.array([0.95, 0.0, 0.0, 0.0, 1.15, 0.0]),  # 原始猜测
        np.array([0.85, 0.0, 0.0, 0.05, 0.95, 0.0]),  # 改进猜测1
        np.array([0.90, 0.0, 0.0, 0.02, 0.85, 0.0]),  # 改进猜测2
    ]
    
    for i, state in enumerate(test_states):
        print(f"\nTest State {i+1}: {state}")
        analyze_sensitivity_matrix(targeter, state, period_nd)
    
    # 3. 测试不同周期的收敛性
    print("\n2. Testing Convergence at Different Periods")
    periods_nd, residuals = test_different_periods()
    
    # 4. 绘制结果
    plt.figure(figsize=(10, 6))
    plt.semilogy(periods_nd, residuals, 'bo-', linewidth=2)
    plt.xlabel('Period (nondimensional units)')
    plt.ylabel('Final Residual')
    plt.title('Convergence vs Period')
    plt.grid(True, alpha=0.3)
    
    # 标注共振比
    labels = ['2:1', '3:2', '1:1']
    for i, (x, y, label) in enumerate(zip(periods_nd, residuals, labels)):
        plt.annotate(label, (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('debug_convergence.png', dpi=150)
    print(f"\nPlot saved to debug_convergence.png")
    
    print("\n" + "=" * 60)
    print("Debug complete. Check the sensitivity matrix analysis above.")
    print("If condition numbers are very high (>1e12), the problem is ill-conditioned.")
    print("Try different initial guesses or regularization methods.")

if __name__ == "__main__":
    main()
