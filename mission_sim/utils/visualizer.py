import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class L1Visualizer:
    """
    Level 1 仿真数据后处理与可视化工具
    职责：读取 HDF5 仿真数据，生成静态时序图表与动态过程动画。
    """
    def __init__(self, h5_filepath: str, sc_id: str):
        """
        :param h5_filepath: HDF5 数据文件路径
        :param sc_id: 需要可视化的航天器 ID (作为 HDF5 的 Group 名)
        """
        if not os.path.exists(h5_filepath):
            raise FileNotFoundError(f"找不到数据文件: {h5_filepath}")
            
        self.filepath = h5_filepath
        self.sc_id = sc_id
        
        # 预加载核心数据到内存 (适用于后处理阶段)
        with h5py.File(self.filepath, 'r') as f:
            self.time = f['time'][:]
            
            # 读取航天器数据组
            sc_group = f[self.sc_id]
            self.true_state = sc_group['true_state'][:]
            self.target_state = sc_group['target_state'][:]
            self.thrust = sc_group['thrust'][:]
            
            # 尝试读取绑定的坐标系元数据
            self.frame_name = sc_group.attrs.get('frame', 'Unknown_Frame')
            
        # 转换为更易读的时间单位 (小时)
        self.time_hours = self.time / 3600.0

    def plot_state_history(self, save_path: str = None):
        """
        功能 1：绘制航天器在特定坐标系下的位置、速度随时间变化曲线。
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        fig.suptitle(f"[{self.sc_id}] State History ({self.frame_name})", fontsize=16)
        
        labels_pos = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
        labels_vel = ['X Velocity (m/s)', 'Y Velocity (m/s)', 'Z Velocity (m/s)']
        
        for i in range(3):
            # 绘制位置 (左列)
            axes[i, 0].plot(self.time_hours, self.true_state[:, i], label='True', color='b')
            axes[i, 0].plot(self.time_hours, self.target_state[:, i], label='Target', color='r', linestyle='--')
            axes[i, 0].set_ylabel(labels_pos[i])
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0: axes[i, 0].legend()
            
            # 绘制速度 (右列)
            axes[i, 1].plot(self.time_hours, self.true_state[:, i+3], label='True', color='g')
            axes[i, 1].plot(self.time_hours, self.target_state[:, i+3], label='Target', color='r', linestyle='--')
            axes[i, 1].set_ylabel(labels_vel[i])
            axes[i, 1].grid(True, alpha=0.3)
            
        axes[2, 0].set_xlabel('Time (Hours)')
        axes[2, 1].set_xlabel('Time (Hours)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[*] 状态时序图已保存至 {save_path}")
        plt.show()

    def plot_gnc_activity(self, save_path: str = None):
        """
        功能 2：绘制 GNC 分系统活动时序 (推力指令大小与分量)。
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"[{self.sc_id}] GNC Activity Timeline", fontsize=16)
        
        # 计算推力幅值
        thrust_mag = np.linalg.norm(self.thrust, axis=1)
        
        # 子图 1：推力三分量
        ax1.plot(self.time_hours, self.thrust[:, 0], label='Fx', alpha=0.8)
        ax1.plot(self.time_hours, self.thrust[:, 1], label='Fy', alpha=0.8)
        ax1.plot(self.time_hours, self.thrust[:, 2], label='Fz', alpha=0.8)
        ax1.set_ylabel('Thrust Components (N)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图 2：推力总幅值 (打火时序)
        ax2.plot(self.time_hours, thrust_mag, color='purple', label='|F| Total')
        ax2.fill_between(self.time_hours, 0, thrust_mag, color='purple', alpha=0.2)
        ax2.set_ylabel('Total Thrust Magnitude (N)')
        ax2.set_xlabel('Time (Hours)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[*] GNC 活动时序图已保存至 {save_path}")
        plt.show()

    def create_animation(self, save_path: str = "simulation_anim.mp4", downsample: int = 10, thrust_scale: float = 50.0):
        """
        功能 3：生成 3D 轨迹动画，并用矢量箭头 (Quiver) 动态展示 GNC 推力活动。
        
        :param save_path: 动画保存路径 (.mp4 或 .gif)
        :param downsample: 降采样率 (例如 10 表示每 10 帧数据渲染 1 帧画面，加快渲染速度)
        :param thrust_scale: 推力箭头的视觉放大系数，防止推力在天文尺度下看不见
        """
        print(f"[*] 正在生成 GNC 动态可视化动画 (降采样={downsample})... 请耐心等待。")
        
        # 降采样数据以提高渲染效率
        t_data = self.time[::downsample]
        pos_data = self.true_state[::downsample, 0:3]
        target_pos = self.target_state[0, 0:3] # 假设 L1 阶段目标点静止
        thrust_data = self.thrust[::downsample]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置视角和边界 (动态计算边界)
        margin = np.max(np.abs(pos_data - target_pos)) * 1.5
        if margin < 1.0: margin = 1000.0 # 防止完全无偏差时缩放崩溃
        
        ax.set_xlim(target_pos[0] - margin, target_pos[0] + margin)
        ax.set_ylim(target_pos[1] - margin, target_pos[1] + margin)
        ax.set_zlim(target_pos[2] - margin, target_pos[2] + margin)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"Spacecraft Trajectory & GNC Thrust ({self.frame_name})")
        
        # 初始化图元
        target_scatter = ax.scatter(*target_pos, color='r', marker='x', s=100, label='Target (L2)')
        traj_line, = ax.plot([], [], [], color='b', alpha=0.5, label='Trajectory')
        sc_scatter, = ax.plot([], [], [], marker='o', color='blue', markersize=8, label=self.sc_id)
        
        # 推力箭头容器
        quiver = None
        
        ax.legend()
        
        def update(frame):
            nonlocal quiver
            # 更新轨迹线 (从起点画到当前帧)
            traj_line.set_data(pos_data[:frame, 0], pos_data[:frame, 1])
            traj_line.set_3d_properties(pos_data[:frame, 2])
            
            # 更新航天器当前位置点
            current_pos = pos_data[frame]
            sc_scatter.set_data([current_pos[0]], [current_pos[1]])
            sc_scatter.set_3d_properties([current_pos[2]])
            
            # 更新推力箭头 (如果存在前一帧的箭头，先移除)
            if quiver:
                quiver.remove()
                
            current_thrust = thrust_data[frame]
            if np.linalg.norm(current_thrust) > 1e-6: # 只有推力大于阈值才绘制箭头
                quiver = ax.quiver(
                    current_pos[0], current_pos[1], current_pos[2],
                    current_thrust[0], current_thrust[1], current_thrust[2],
                    color='red', length=thrust_scale, normalize=False, 
                    arrow_length_ratio=0.3, linewidth=2
                )
            else:
                quiver = None
                
            ax.set_title(f"Time: {t_data[frame]/3600.0:.2f} Hours | {self.frame_name}")
            return traj_line, sc_scatter
            
        frames_count = len(pos_data)
        anim = FuncAnimation(fig, update, frames=frames_count, interval=50, blit=False)
        
        # 尝试保存动画 (需要系统中安装了 ffmpeg)
        try:
            anim.save(save_path, writer='ffmpeg', fps=30)
            print(f"[*] 动画已成功保存至 {save_path}")
        except Exception as e:
            print(f"[*] 动画保存失败 (可能缺少 ffmpeg)。错误信息: {e}")
            print("[*] 提示: 尝试将后缀改为 .gif，使用 pillow 引擎保存。")
