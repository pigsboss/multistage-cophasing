# mission_sim/utils/visualizer_L1.py
import numpy as np
import matplotlib.pyplot as plt
from mission_sim.utils.visualizer import BaseVisualizer

class L1Visualizer(BaseVisualizer):
    """
    MCPC L1 级专用可视化工具 (继承自 BaseVisualizer)
    职责：从 HDF5 中读取时序数据，绘制 3D 绝对轨迹、GNC 追踪误差以及控制打火/ΔV 消耗曲线。
    """
    def __init__(self, filepath: str, mission_name: str = "L1 Baseline Mission"):
        super().__init__(filepath)
        self.mission_name = mission_name

    def plot_3d_trajectory(self, save_path: str = "data/L1_3d_trajectory.png"):
        """绘制 3D 标称轨道与实际物理轨迹的对比图并保存"""
        nominal_states = self.load_dataset('nominal_states')
        true_states = self.load_dataset('true_states')

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        M2KM = 1e-3
        nom_pos = nominal_states[:, 0:3] * M2KM
        true_pos = true_states[:, 0:3] * M2KM
        
        ax.plot(nom_pos[:, 0], nom_pos[:, 1], nom_pos[:, 2], 
                color='gray', linestyle='--', linewidth=1.5, label='Nominal Orbit')
        ax.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], 
                color='dodgerblue', linewidth=2, label='True Trajectory')
        
        ax.scatter(true_pos[0, 0], true_pos[0, 1], true_pos[0, 2], 
                   color='green', marker='o', s=50, label='Start')
        ax.scatter(true_pos[-1, 0], true_pos[-1, 1], true_pos[-1, 2], 
                   color='red', marker='X', s=50, label='End')

        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')
        ax.set_title(f"{self.mission_name} - 3D Absolute Trajectory", fontsize=14, fontweight='bold')
        ax.legend()
        
        self.save_plot(fig, save_path)

    def plot_tracking_error(self, save_path: str = "data/L1_tracking_error.png"):
        """绘制 GNC 系统的位置与速度追踪偏差并保存"""
        times = self.load_dataset('epochs')
        errors = self.load_dataset('tracking_errors')
        days = times / 86400.0
        
        fig, axes = self.create_figure(2, 1, f"{self.mission_name} - Tracking Error")
        ax1, ax2 = axes
        
        ax1.plot(days, errors[:, 0], label='Error X', alpha=0.8)
        ax1.plot(days, errors[:, 1], label='Error Y', alpha=0.8)
        ax1.plot(days, errors[:, 2], label='Error Z', alpha=0.8)
        ax1.set_ylabel('Position Error [m]', fontweight='bold')
        ax1.legend()
        
        ax2.plot(days, errors[:, 3] * 1000, label='Error Vx', alpha=0.8)
        ax2.plot(days, errors[:, 4] * 1000, label='Error Vy', alpha=0.8)
        ax2.plot(days, errors[:, 5] * 1000, label='Error Vz', alpha=0.8)
        ax2.set_xlabel('Time [Days]', fontweight='bold')
        ax2.set_ylabel('Velocity Error [mm/s]', fontweight='bold')
        ax2.legend()
        
        self.save_plot(fig, save_path)

    def plot_control_effort(self, save_path: str = "data/L1_control_effort.png"):
        """绘制控制打火推力时序与累计 ΔV 消耗并保存"""
        times = self.load_dataset('epochs')
        forces = self.load_dataset('control_forces')
        accumulated_dvs = self.load_dataset('accumulated_dvs')
        days = times / 86400.0
        
        fig, axes = self.create_figure(2, 1, f"{self.mission_name} - Control Effort & Fuel")
        ax1, ax2 = axes
        
        ax1.plot(days, forces[:, 0], label='Force X', alpha=0.7)
        ax1.plot(days, forces[:, 1], label='Force Y', alpha=0.7)
        ax1.plot(days, forces[:, 2], label='Force Z', alpha=0.7)
        ax1.set_ylabel('Control Force [N]', fontweight='bold')
        ax1.legend()
        
        ax2.plot(days, accumulated_dvs, color='firebrick', linewidth=2)
        ax2.set_xlabel('Time [Days]', fontweight='bold')
        ax2.set_ylabel(r'Accumulated $\Delta V$ [m/s]', fontweight='bold')
        
        final_dv = accumulated_dvs[-1]
        ax2.annotate(f'Total: {final_dv:.4f} m/s', 
                     xy=(days[-1], final_dv), xytext=(days[-1]*0.8, final_dv*0.8),
                     arrowprops=dict(facecolor='black', arrowstyle='->'), 
                     fontsize=10, fontweight='bold')
        
        self.save_plot(fig, save_path)
