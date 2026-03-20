# mission_sim/core/gnc/ground_station.py
import numpy as np
from mission_sim.core.types import CoordinateFrame, Telecommand

class GroundStation:
    """
    受限真实地面测控站类 (Level 1)
    职责：模拟地面站定轨系统，基于可视弧段和采样率，向航天器发送带噪遥测数据和上行遥控指令。
    """
    def __init__(self, 
                 name: str, 
                 operating_frame: CoordinateFrame, 
                 pos_noise_std: float = 5.0, 
                 vel_noise_std: float = 0.005,
                 sampling_rate_hz: float = 0.1,
                 visibility_windows: list[tuple[float, float]] = None):
        """
        :param name: 测控站或深空网名称 (如 "DSN_Goldstone")
        :param operating_frame: 地面站解算星历的工作基准坐标系 (强契约)
        :param pos_noise_std: 定轨位置误差标准差 (1 sigma, m)
        :param vel_noise_std: 定轨速度误差标准差 (1 sigma, m/s)
        :param sampling_rate_hz: 测控数据下发频率 (Hz)，例如 0.1Hz 表示每 10 秒一个点
        :param visibility_windows: 可视弧段列表，格式为 [(start_time1, end_time1), ...] (秒)。若为 None 则全天候可见。
        """
        self.name = name
        self.operating_frame = operating_frame
        self.pos_noise_std = float(pos_noise_std)
        self.vel_noise_std = float(vel_noise_std)
        
        # 离散采样率控制
        self.sampling_rate_hz = sampling_rate_hz
        self.min_interval = 1.0 / sampling_rate_hz if sampling_rate_hz > 0 else 0.0
        self.last_track_time = -np.inf
        
        # 盲区与可视弧段模拟
        self.visibility_windows = visibility_windows or []

    def is_visible(self, epoch: float) -> bool:
        """判断当前仿真历元是否处于测控站的可视弧段内"""
        if not self.visibility_windows:
            return True  # 兜底：如果没有定义窗口，假设全弧段覆盖 (如由3个中继星组网)
            
        for (start_t, end_t) in self.visibility_windows:
            if start_t <= epoch <= end_t:
                return True
        return False

    def track_spacecraft(self, true_state: np.ndarray, sc_frame: CoordinateFrame, epoch: float) -> tuple[np.ndarray | None, CoordinateFrame]:
        """
        模拟跟踪测量：判断可视性与采样率，叠加噪声后返回。
        
        :param true_state: 航天器绝对真实状态 [x, y, z, vx, vy, vz]
        :param sc_frame: 该真实状态所在的坐标系
        :param epoch: 当前仿真时间
        :return: (带噪声的观测状态 | 若不可见则为 None, 地面站的工作坐标系)
        """
        # 【防呆校验】确保物理域的状态没给错坐标系
        if sc_frame != self.operating_frame:
            raise ValueError(
                f"[{self.name} 测控站崩溃] 坐标系不匹配！地面站工作在 {self.operating_frame.name}，"
                f"无法直接解算基于 {sc_frame.name} 的物理真值。"
            )

        # 1. 检查可视弧段：如果在盲区，则失去信号
        if not self.is_visible(epoch):
            return None, self.operating_frame
            
        # 2. 检查离散采样率：如果还没到下一个采样点，保持静默
        if (epoch - self.last_track_time) < self.min_interval - 1e-5:
            return None, self.operating_frame

        # 3. 生成高斯白噪声并叠加
        pos_noise = np.random.normal(0.0, self.pos_noise_std, 3)
        vel_noise = np.random.normal(0.0, self.vel_noise_std, 3)
        noise_vector = np.concatenate([pos_noise, vel_noise])
        
        observed_state = true_state + noise_vector
        self.last_track_time = epoch
        
        # 返回观测数据并盖上坐标系印章
        return observed_state, self.operating_frame

    def generate_telecommand(self, cmd_type: str, target_state: np.ndarray, target_frame: CoordinateFrame, execution_epoch: float = 0.0) -> Telecommand:
        """生成发送给 GNC 的标准遥控指令包"""
        return Telecommand(
            cmd_type=cmd_type, 
            target_state=target_state, 
            frame=target_frame,
            execution_epoch=execution_epoch
        )    

    def __repr__(self):
        return (f"GroundStation[{self.name}] | Frame: {self.operating_frame.name} | "
                f"Noise: {self.pos_noise_std}m, {self.vel_noise_std}m/s | "
                f"SampleRate: {self.sampling_rate_hz}Hz")
