# mission_sim/core/gnc/gnc_subsystem.py
"""
制导、导航与控制 (GNC) 子系统 - 重构版
修复了控制力数组形状问题，增强错误处理和调试信息
"""

import numpy as np
from typing import Tuple, Optional

from mission_sim.core.types import CoordinateFrame, Telecommand
from mission_sim.core.trajectory.ephemeris import Ephemeris


class GNC_Subsystem:
    """
    制导、导航与控制 (GNC) 子系统 (Level 1) - 重构版
    职责：接收带有坐标系契约的导航状态，读取动态星历标称轨迹，计算追踪误差，并基于控制律输出补偿推力。
    
    重构重点：
    1. 修复控制力数组形状问题
    2. 增强错误处理和调试信息
    3. 添加控制力标准化方法
    4. 完善坐标系统一校验
    """
    
    def __init__(self, sc_id: str, operating_frame: CoordinateFrame):
        """
        初始化 GNC 子系统
        
        Args:
            sc_id: 航天器标识符 (如 "JWST_Alpha")
            operating_frame: GNC 算法的工作基准坐标系 (强契约)
        """
        self.sc_id = sc_id
        self.operating_frame = operating_frame
        
        # 导航滤波器输出状态 (初始为空)
        self.current_nav_state = np.zeros(6, dtype=np.float64)
        
        # 标称轨道星历表
        self.ref_ephemeris: Optional[Ephemeris] = None
        
        # 遥测记录暂存区
        self.last_control_force = np.zeros(3, dtype=np.float64)  # 确保是3维数组
        self.last_tracking_error = np.zeros(6, dtype=np.float64)
        self.last_target_state = np.zeros(6, dtype=np.float64)
        
        # 调试和统计信息
        self.total_control_calls = 0
        self.force_shape_warnings = 0
        
        print(f"✅ [{self.sc_id} GNC] 初始化完成，工作坐标系: {self.operating_frame.name}")
    
    def load_reference_trajectory(self, eph: Ephemeris) -> None:
        """
        加载由预处理阶段生成的标称星历
        
        Args:
            eph: 标称轨道星历对象
            
        Raises:
            ValueError: 如果星历坐标系与GNC工作坐标系不匹配
        """
        # 强校验：确保预处理工厂生成的星历坐标系，与 GNC 当前的工作坐标系一致
        if eph.frame != self.operating_frame:
            error_msg = (
                f"[{self.sc_id} GNC 崩溃] 标称星历坐标系不匹配！\n"
                f"  GNC 运行在: {self.operating_frame.name}\n"
                f"  星历基准为: {eph.frame.name}\n"
                f"  请检查轨道生成器的输出配置。"
            )
            raise ValueError(error_msg)
        
        self.ref_ephemeris = eph
        duration_hours = (eph.times[-1] - eph.times[0]) / 3600.0
        
        print(f"✅ [{self.sc_id} GNC] 成功锁定动态基准星历")
        print(f"   星历时长: {duration_hours:.1f} 小时")
        print(f"   星历点数: {len(eph.times)}")
        print(f"   时间范围: {eph.times[0]:.1f} 到 {eph.times[-1]:.1f} 秒")
    
    def update_navigation(self, obs_state: np.ndarray, frame: CoordinateFrame) -> None:
        """
        更新来自 GroundStation 的导航状态
        
        Args:
            obs_state: 观测状态向量 [x, y, z, vx, vy, vz]
            frame: 观测状态所在的坐标系
            
        Raises:
            ValueError: 如果观测状态坐标系与GNC工作坐标系不匹配
            TypeError: 如果观测状态不是numpy数组
            ValueError: 如果观测状态形状不正确
        """
        # 类型校验
        if not isinstance(obs_state, np.ndarray):
            raise TypeError(f"[{self.sc_id} GNC] 观测状态必须是 numpy 数组，当前类型: {type(obs_state)}")
        
        # 形状校验
        if obs_state.shape != (6,):
            raise ValueError(f"[{self.sc_id} GNC] 观测状态必须是形状为 (6,) 的向量，当前形状: {obs_state.shape}")
        
        # 坐标系校验
        if frame != self.operating_frame:
            error_msg = (
                f"[{self.sc_id} GNC 拒收] 导航状态坐标系不匹配！\n"
                f"  期望坐标系: {self.operating_frame.name}\n"
                f"  实际坐标系: {frame.name}\n"
                f"  请检查地面站配置或坐标系转换逻辑。"
            )
            raise ValueError(error_msg)
        
        # 更新导航状态
        self.current_nav_state = np.copy(obs_state.astype(np.float64))
        
        # 调试信息
        pos_norm = np.linalg.norm(obs_state[0:3])
        vel_norm = np.linalg.norm(obs_state[3:6])
        
        if self.total_control_calls % 1000 == 0:  # 减少输出频率
            print(f"  [{self.sc_id} GNC] 导航更新: 位置={pos_norm:.1f}m, 速度={vel_norm:.4f}m/s")
    
    def compute_control_force(self, epoch: float, K_matrix: np.ndarray) -> Tuple[np.ndarray, CoordinateFrame]:
        """
        计算站位维持的补偿推力 - 重构版
        修复了控制力数组形状问题，确保输出总是3维数组
        
        核心逻辑: e(t) = X_nav(t) - X_nominal(t)
                  u(t) = -K * e(t)
        
        Args:
            epoch: 当前仿真历元时间 (s)
            K_matrix: 最优控制反馈增益矩阵，期望形状 (3, 6)
            
        Returns:
            Tuple[np.ndarray, CoordinateFrame]: 
                - 推力向量 [Fx, Fy, Fz] (形状始终为 (3,))
                - 推力所在的坐标系标签
                
        Raises:
            RuntimeError: 如果未加载标称星历
            ValueError: 如果K矩阵形状不正确
        """
        self.total_control_calls += 1
        
        # 检查标称星历
        if self.ref_ephemeris is None:
            raise RuntimeError(f"[{self.sc_id} GNC] 未加载标称星历，无法计算动态追踪偏差！")
        
        # 1. 动态获取当前时刻的绝对标称目标状态
        try:
            target_state = self.ref_ephemeris.get_interpolated_state(epoch)
            self.last_target_state = np.copy(target_state)
        except Exception as e:
            print(f"⚠️ [{self.sc_id} GNC] 星历插值失败: {e}")
            # 使用最近的有效状态
            target_state = self.last_target_state
        
        # 2. 计算追踪误差
        error = self.current_nav_state - target_state
        self.last_tracking_error = np.copy(error)
        
        # 3. 验证K矩阵形状
        K_matrix = self._validate_and_fix_K_matrix(K_matrix)
        
        # 4. 线性反馈控制律 (LQR)
        try:
            # 计算原始控制力
            raw_force = -K_matrix @ error
            
            # 标准化控制力，确保输出是3维数组
            control_force = self._standardize_control_force(raw_force)
            self.last_control_force = np.copy(control_force)
            
        except Exception as e:
            print(f"⚠️ [{self.sc_id} GNC] 控制力计算失败: {e}")
            # 使用零控制力作为安全回退
            control_force = np.zeros(3, dtype=np.float64)
            self.last_control_force = control_force
        
        # 5. 调试输出
        if self.total_control_calls % 1000 == 0:  # 减少输出频率
            err_pos = np.linalg.norm(error[0:3])
            err_vel = np.linalg.norm(error[3:6]) * 1000
            force_norm = np.linalg.norm(control_force)
            
            print(f"  [{self.sc_id} GNC] 控制计算: "
                  f"位置误差={err_pos:.2f}m, "
                  f"速度误差={err_vel:.2f}mm/s, "
                  f"控制力={force_norm:.4f}N")
        
        return control_force, self.operating_frame
    
    def _validate_and_fix_K_matrix(self, K_matrix: np.ndarray) -> np.ndarray:
        """
        验证和修复K矩阵形状
        
        Args:
            K_matrix: 输入的增益矩阵
            
        Returns:
            np.ndarray: 修正后的增益矩阵，确保形状为 (3, 6)
        """
        # 确保是numpy数组
        if not isinstance(K_matrix, np.ndarray):
            K_matrix = np.array(K_matrix, dtype=np.float64)
        
        # 检查形状
        expected_shape = (3, 6)
        
        if K_matrix.shape == expected_shape:
            return K_matrix
        
        # 形状修正逻辑
        print(f"⚠️ [{self.sc_id} GNC] K矩阵形状异常: {K_matrix.shape}，期望 {expected_shape}")
        
        if K_matrix.shape == (1, 6) or K_matrix.shape == (6,):
            # 如果K是1x6或6x1，则扩展为3x6
            if K_matrix.shape == (6,):
                K_matrix = K_matrix.reshape(1, 6)
            K_matrix = np.tile(K_matrix, (3, 1))
            print(f"  [{self.sc_id} GNC] 已扩展为3x6矩阵")
            
        elif K_matrix.shape[0] > 3:
            # 如果行数>3，只取前3行
            K_matrix = K_matrix[:3, :]
            print(f"  [{self.sc_id} GNC] 已截取为3x6矩阵")
            
        elif K_matrix.shape[1] > 6:
            # 如果列数>6，只取前6列
            K_matrix = K_matrix[:, :6]
            print(f"  [{self.sc_id} GNC] 已截取为3x6矩阵")
            
        else:
            # 其他情况，尝试重塑
            try:
                K_matrix = K_matrix.reshape(expected_shape)
                print(f"  [{self.sc_id} GNC] 已重塑为3x6矩阵")
            except:
                # 无法修复，使用单位矩阵作为备用
                print(f"⚠️ [{self.sc_id} GNC] 无法修复K矩阵形状，使用备用增益")
                K_matrix = np.eye(3, 6, dtype=np.float64) * 1e-3
        
        return K_matrix
    
    def _standardize_control_force(self, raw_force) -> np.ndarray:
        """
        标准化控制力输入，确保输出总是形状为 (3,) 的数组
        
        Args:
            raw_force: 原始控制力，可以是标量或数组
            
        Returns:
            np.ndarray: 标准化后的控制力数组 (3,)
        """
        # 处理各种输入类型
        if isinstance(raw_force, (int, float, np.number)):
            # 标量 -> 转换为数组 [force, 0, 0]
            return np.array([float(raw_force), 0.0, 0.0], dtype=np.float64)
        
        elif isinstance(raw_force, np.ndarray):
            if raw_force.shape == ():
                # 0维数组 -> 转换为标量数组
                return np.array([float(raw_force), 0.0, 0.0], dtype=np.float64)
            elif raw_force.shape == (1,):
                # 1维标量数组 -> 转换为3维数组
                return np.array([float(raw_force[0]), 0.0, 0.0], dtype=np.float64)
            elif raw_force.shape == (3,):
                # 已经是3维数组
                return raw_force.astype(np.float64)
            elif len(raw_force) >= 3:
                # 长度≥3的数组 -> 取前3个元素
                return raw_force[:3].astype(np.float64)
            else:
                # 其他形状 -> 尝试重塑
                try:
                    if raw_force.size == 3:
                        return raw_force.reshape(3).astype(np.float64)
                except:
                    pass
        
        # 无法处理的情况，返回零向量
        if self.force_shape_warnings < 5:  # 限制警告次数
            print(f"⚠️ [{self.sc_id} GNC] 控制力格式异常: {type(raw_force)}, 形状: {getattr(raw_force, 'shape', 'N/A')}")
            self.force_shape_warnings += 1
        
        return np.zeros(3, dtype=np.float64)
    
    def get_tracking_error(self, epoch: float) -> np.ndarray:
        """
        获取当前时刻的追踪误差（不计算控制力）
        
        Args:
            epoch: 当前仿真历元时间 (s)
            
        Returns:
            np.ndarray: 追踪误差向量 (6,)
        """
        if self.ref_ephemeris is None:
            return np.zeros(6, dtype=np.float64)
        
        try:
            target_state = self.ref_ephemeris.get_interpolated_state(epoch)
            error = self.current_nav_state - target_state
            return error
        except:
            return self.last_tracking_error
    
    def get_performance_metrics(self) -> dict:
        """
        获取GNC性能指标
        
        Returns:
            dict: 包含性能指标的字典
        """
        err_pos = np.linalg.norm(self.last_tracking_error[0:3])
        err_vel = np.linalg.norm(self.last_tracking_error[3:6])
        force_norm = np.linalg.norm(self.last_control_force)
        
        return {
            "position_error_m": float(err_pos),
            "velocity_error_mps": float(err_vel),
            "control_force_norm": float(force_norm),
            "total_control_calls": self.total_control_calls,
            "force_shape_warnings": self.force_shape_warnings
        }
    
    def reset(self) -> None:
        """重置GNC状态（用于测试或重新初始化）"""
        self.current_nav_state = np.zeros(6, dtype=np.float64)
        self.last_control_force = np.zeros(3, dtype=np.float64)
        self.last_tracking_error = np.zeros(6, dtype=np.float64)
        self.total_control_calls = 0
        self.force_shape_warnings = 0
        print(f"[{self.sc_id} GNC] 已重置")
    
    def __repr__(self) -> str:
        """字符串表示"""
        metrics = self.get_performance_metrics()
        return (f"GNC[{self.sc_id}] | Frame: {self.operating_frame.name} | "
                f"PosErr: {metrics['position_error_m']:.2f}m | "
                f"VelErr: {metrics['velocity_error_mps']*1000:.2f}mm/s | "
                f"Calls: {metrics['total_control_calls']}")


# 兼容性别名
GNCSubsystem = GNC_Subsystem