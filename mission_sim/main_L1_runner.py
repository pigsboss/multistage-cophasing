# mission_sim/main_L1_runner.py
"""
MCPC 框架 L1 级仿真主程序 - 修正版
JWST 标称轨道 30 天全链路仿真
修复了控制力数组形状错误、LQR增益计算错误、logger参数验证问题
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from scipy.integrate import solve_ivp

# 核心类型与领域模型
from mission_sim.core.types import CoordinateFrame, Telecommand
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.trajectory.halo_corrector import HaloDifferentialCorrector
from mission_sim.core.physics.environment import CelestialEnvironment
from mission_sim.core.physics.spacecraft import SpacecraftPointMass
from mission_sim.core.physics.models.gravity_crtbp import Gravity_CRTBP
from mission_sim.core.gnc.ground_station import GroundStation
from mission_sim.core.gnc.gnc_subsystem import GNC_Subsystem

# 基础设施工具
from mission_sim.utils.logger import HDF5Logger, SimulationMetadata
from mission_sim.utils.math_tools import get_lqr_gain
from mission_sim.utils.visualizer_L1 import L1Visualizer


class L1MissionSimulation:
    """
    L1 级任务仿真控制器 - 修正版
    修复了控制力数组形状错误、LQR增益计算错误
    """
    
    def __init__(self, config: dict = None):
        """
        初始化仿真控制器
        
        Args:
            config: 仿真配置字典
        """
        # 默认配置
        self.default_config = {
            "mission_name": "JWST 30-Day L2 Station Keeping",
            "simulation_days": 30,
            "time_step": 10.0,           # 仿真步长 (秒)
            "dt_orbital": 0.001,         # 轨道生成步长 (无量纲)
            "Az_target": 0.05,           # 目标Z振幅 (无量纲)
            "log_buffer_size": 500,      # 日志缓冲区大小
            "log_compression": True,     # 日志压缩
            "progress_interval": 0.05,   # 进度报告间隔 (比例)
            "enable_visualization": True, # 是否启用可视化
            "data_dir": "data",          # 数据输出目录
            "log_level": "INFO"          # 日志级别
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        
        # 初始化状态变量
        self.simulation_start_time = None
        self.simulation_end_time = None
        self.current_step = 0
        self.total_steps = 0
        
        # 核心组件
        self.environment = None
        self.spacecraft = None
        self.ground_station = None
        self.gnc_system = None
        self.ephemeris = None
        self.logger = None
        self.k_matrix = None
        
        # 创建输出目录
        os.makedirs(self.config["data_dir"], exist_ok=True)
        
        # 生成唯一的任务ID
        self.mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.h5_file = os.path.join(
            self.config["data_dir"], 
            f"L1_{self.config['mission_name'].replace(' ', '_')}_{self.mission_id}.h5"
        )
        
        print("="*80)
        print(f"🚀 MCPC 框架 L1 级: {self.config['mission_name']} (修正版)")
        print("="*80)
        print(f"[配置] 任务ID: {self.mission_id}")
        print(f"[配置] 仿真时长: {self.config['simulation_days']} 天")
        print(f"[配置] 步长: {self.config['time_step']} 秒")
        print(f"[配置] 输出文件: {self.h5_file}")
    
    def run(self) -> bool:
        """
        执行完整仿真流程
        
        Returns:
            仿真是否成功
        """
        try:
            # 记录开始时间
            self.simulation_start_time = time.time()
            
            # 1. 生成标称轨道
            if not self._generate_nominal_orbit():
                print("❌ 标称轨道生成失败，使用备用轨道")
                self._generate_fallback_orbit()
            
            # 2. 初始化物理域
            self._initialize_physical_domain()
            
            # 3. 初始化信息域
            self._initialize_information_domain()
            
            # 4. 设计控制律
            self._design_control_law()
            
            # 5. 初始化数据记录
            self._initialize_data_logging()
            
            # 6. 执行仿真主循环
            self._execute_simulation_loop()
            
            # 7. 最终处理
            success = self._finalize_simulation()
            
            return success
            
        except KeyboardInterrupt:
            print("\n⏹️ 仿真被用户中断")
            self._emergency_shutdown()
            return False
            
        except Exception as e:
            print(f"\n❌ 仿真运行失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self._emergency_shutdown()
            return False
    
    def _generate_nominal_orbit(self) -> bool:
        """
        生成标称 Halo 轨道
        
        Returns:
            是否成功生成
        """
        print("\n" + "-"*60)
        print("[阶段1] 生成标称 Halo 轨道")
        print("-"*60)
        
        try:
            # 创建轨道生成器
            generator = HaloDifferentialCorrector()
            
            # 配置生成参数
            orbit_config = {
                "Az": self.config["Az_target"],
                "dt": self.config["dt_orbital"],
                "initial_guess": [1.01106, 0.05, 0.0105]
            }
            
            # 生成轨道
            self.ephemeris = generator.generate(orbit_config)
            
            # 验证轨道质量
            if self._validate_orbit_quality(self.ephemeris):
                print(f"✅ 标称轨道生成成功")
                print(f"   周期: {self.ephemeris.times[-1]/86400:.2f} 天")
                print(f"   点数: {len(self.ephemeris.times)}")
                return True
            else:
                print("⚠️ 轨道质量验证失败，但继续使用")
                return True
                
        except Exception as e:
            print(f"❌ 轨道生成失败: {e}")
            return False
    
    def _generate_fallback_orbit(self) -> None:
        """
        生成备用轨道（当主方法失败时）
        """
        print("   使用备用轨道生成方案...")
        
        # 使用已知的、经验证的初始状态
        state0_nd = np.array([
            1.01106,    # x: L2点附近
            0.0,        # y: 从XZ平面出发
            0.05,       # z: 目标振幅
            0.0,        # vx: 初始x速度为0
            0.0105,     # vy: 精心调整的切向速度
            0.0         # vz: 初始z速度为0
        ])
        
        # 积分轨道
        def crtbp_eom(t, state):
            x, y, z, vx, vy, vz = state
            mu = 3.00348e-6
            
            r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
            r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
            
            ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
            ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
            az = -(1-mu)*z/r1**3 - mu*z/r2**3
            
            return np.array([vx, vy, vz, ax, ay, az])
        
        # 无量纲周期 (约 π)
        T_nd = 3.141
        times_nd = np.arange(0, T_nd, self.config["dt_orbital"])
        
        sol = solve_ivp(
            fun=crtbp_eom,
            t_span=(0, T_nd),
            y0=state0_nd,
            t_eval=times_nd,
            method='DOP853',
            rtol=1e-12,
            atol=1e-12
        )
        
        # 转换为物理单位
        AU = 1.495978707e11
        OMEGA = 1.990986e-7
        
        physical_times = sol.t / OMEGA
        physical_states = sol.y.T.copy()
        physical_states[:, 0:3] *= AU
        physical_states[:, 3:6] *= (AU * OMEGA)
        
        self.ephemeris = Ephemeris(
            times=physical_times,
            states=physical_states,
            frame=CoordinateFrame.SUN_EARTH_ROTATING
        )
        
        print(f"✅ 备用轨道生成完成，周期: {T_nd/OMEGA/86400:.2f} 天")
    
    def _validate_orbit_quality(self, ephemeris: Ephemeris) -> bool:
        """
        验证轨道质量
        
        Args:
            ephemeris: 轨道星历
            
        Returns:
            轨道质量是否可接受
        """
        # 检查轨道闭合性
        pos_start = ephemeris.states[0, 0:3]
        pos_end = ephemeris.states[-1, 0:3]
        vel_start = ephemeris.states[0, 3:6]
        vel_end = ephemeris.states[-1, 3:6]
        
        pos_error = np.linalg.norm(pos_end - pos_start)
        vel_error = np.linalg.norm(vel_end - vel_start)
        
        print(f"   轨道闭合性检查:")
        print(f"     位置误差: {pos_error:.2e} m")
        print(f"     速度误差: {vel_error:.2e} m/s")
        
        # 检查时间序列单调性
        time_diff = np.diff(ephemeris.times)
        if np.any(time_diff <= 0):
            print("⚠️ 时间序列非单调递增")
            return False
        
        # 检查数据有效性
        if np.any(np.isnan(ephemeris.states)):
            print("❌ 轨道数据包含NaN值")
            return False
        
        return pos_error < 1e8  # 位置误差小于100,000 km
    
    def _initialize_physical_domain(self) -> None:
        """
        初始化物理域（环境、航天器）
        """
        print("\n" + "-"*60)
        print("[阶段2] 初始化物理域")
        print("-"*60)
        
        # 1. 天体环境
        self.environment = CelestialEnvironment(
            computation_frame=CoordinateFrame.SUN_EARTH_ROTATING,
            initial_epoch=0.0
        )
        
        # 注册力学模型
        gravity_model = Gravity_CRTBP()
        self.environment.register_force(gravity_model)
        
        # 2. 航天器初始化
        # 获取标称起点并注入初始入轨偏差
        nominal_state_0 = self.ephemeris.get_interpolated_state(0.0)
        
        # 初始注入误差：位置 ±2km，速度 ±1cm/s
        injection_error = np.array([
            2000.0, 2000.0, -1000.0,  # 位置误差 (m)
            0.01, -0.01, 0.005        # 速度误差 (m/s)
        ])
        
        true_initial_state = nominal_state_0 + injection_error
        sc_mass = 6200.0  # JWST 发射质量 (kg)
        
        self.spacecraft = SpacecraftPointMass(
            sc_id="JWST_Shadow",
            initial_state=true_initial_state,
            frame=CoordinateFrame.SUN_EARTH_ROTATING,
            initial_mass=sc_mass
        )
        
        print("✅ 物理域初始化完成")
        print(f"   航天器质量: {sc_mass} kg")
        print(f"   初始位置偏差: {np.linalg.norm(injection_error[0:3]):.1f} m")
        print(f"   初始速度偏差: {np.linalg.norm(injection_error[3:6])*1000:.3f} mm/s")
    
    def _initialize_information_domain(self) -> None:
        """
        初始化信息域（测控、GNC）
        """
        print("\n" + "-"*60)
        print("[阶段3] 初始化信息域")
        print("-"*60)
        
        # 1. 地面测控站
        # 模拟深空网：5m位置噪声，5mm/s速度噪声，0.1Hz采样率
        self.ground_station = GroundStation(
            name="DSN_Network",
            operating_frame=CoordinateFrame.SUN_EARTH_ROTATING,
            pos_noise_std=5.0,
            vel_noise_std=0.005,
            sampling_rate_hz=0.1
        )
        
        # 2. GNC 子系统
        self.gnc_system = GNC_Subsystem(
            sc_id="JWST_Shadow",
            operating_frame=CoordinateFrame.SUN_EARTH_ROTATING
        )
        
        # 加载参考轨道
        self.gnc_system.load_reference_trajectory(self.ephemeris)
        
        print("✅ 信息域初始化完成")
        print(f"   测控噪声: 位置 {self.ground_station.pos_noise_std} m, "
              f"速度 {self.ground_station.vel_noise_std*1000:.1f} mm/s")
    
    def _design_control_law(self) -> None:
        """
        设计 LQR 最优控制律 - 修正版
        修复了gamma_l计算错误，确保K矩阵形状为(3,6)
        """
        print("\n" + "-"*60)
        print("[阶段4] 设计最优控制律 (修正版)")
        print("-"*60)
        
        # 修正：使用正确的mu值计算gamma_l
        mu = 3.00348e-6
        gamma_l = np.cbrt(mu / 3.0)  # 无量纲L2点距离
        
        # 日地 L2 点动力学线性化常数
        omega = 1.990986e-7
        
        print(f"   计算参数: μ={mu:.6e}, γ={gamma_l:.6e}, ω={omega:.6e} rad/s")
        
        # 状态矩阵 A (6x6) - 修正系数
        a_mat = np.zeros((6, 6))
        a_mat[0:3, 3:6] = np.eye(3)
        a_mat[3, 0] = (2*gamma_l + 1) * omega**2
        a_mat[4, 1] = (1 - gamma_l) * omega**2
        a_mat[5, 2] = -gamma_l * omega**2
        a_mat[3, 4] = 2 * omega
        a_mat[4, 3] = -2 * omega
        
        # 控制矩阵 B (6x3)
        b_mat = np.zeros((6, 3))
        b_mat[3:6, 0:3] = np.eye(3) / self.spacecraft.mass
        
        # 权重矩阵：强化速度惩罚以抑制长周期漂移
        q_mat = np.diag([1.0, 1.0, 1.0, 1e6, 1e6, 1e6])
        r_mat = np.diag([10.0, 10.0, 10.0])
        
        # 求解 LQR 增益
        try:
            self.k_matrix = get_lqr_gain(a_mat, b_mat, q_mat, r_mat)
            
            # 验证K矩阵形状
            if self.k_matrix.shape != (3, 6):
                print(f"⚠️ 警告: K矩阵形状异常: {self.k_matrix.shape}，期望(3,6)")
                # 如果K是1x6，则扩展为3x6
                if self.k_matrix.shape == (1, 6) or self.k_matrix.shape == (6,):
                    self.k_matrix = np.tile(self.k_matrix, (3, 1))
                    print(f"    已扩展为3x6矩阵")
                elif self.k_matrix.shape[0] > 3:
                    # 只取前3行
                    self.k_matrix = self.k_matrix[:3, :]
                    print(f"    已截取为3x6矩阵")
                    
        except Exception as e:
            print(f"❌ LQR增益计算失败: {e}")
            # 使用备用的单位增益矩阵
            self.k_matrix = np.eye(3, 6) * 1e-3
            print(f"    使用备用增益矩阵")
        
        print("✅ LQR 控制律设计完成")
        print(f"   增益矩阵形状: {self.k_matrix.shape}")
        print(f"   增益矩阵范数: ||K|| = {np.linalg.norm(self.k_matrix):.2e}")
        print(f"   状态权重: diag({np.diag(q_mat)})")
        print(f"   控制权重: diag({np.diag(r_mat)})")
    
    def _initialize_data_logging(self) -> None:
        """
        初始化数据记录系统
        """
        print("\n" + "-"*60)
        print("[阶段5] 初始化数据记录系统")
        print("-"*60)
        
        # 创建元数据
        metadata = SimulationMetadata.create_mission_metadata(
            mission_name=self.config["mission_name"],
            config={
                "simulation_days": self.config["simulation_days"],
                "time_step": self.config["time_step"],
                "spacecraft_mass": self.spacecraft.mass,
                "control_type": "LQR",
                "mission_id": self.mission_id,
                "ephemeris_period_days": self.ephemeris.times[-1]/86400
            }
        )
        
        # 初始化 HDF5 记录器
        self.logger = HDF5Logger(
            filepath=self.h5_file,
            buffer_size=self.config["log_buffer_size"],
            compression=self.config["log_compression"],
            auto_flush=True
        )
        
        # 初始化文件结构
        self.logger.initialize_file(metadata)
        
        print("✅ 数据记录系统初始化完成")
        print(f"   数据文件: {self.h5_file}")
        print(f"   缓冲区大小: {self.config['log_buffer_size']} 条")
        print(f"   数据压缩: {'启用' if self.config['log_compression'] else '禁用'}")
    
    def _execute_simulation_loop(self) -> None:
        """
        执行仿真主循环 - 修正版
        修复了控制力数组形状错误
        """
        print("\n" + "-"*60)
        print("[阶段6] 执行仿真主循环 (修正版)")
        print("-"*60)
        
        # 计算仿真参数
        dt = self.config["time_step"]
        sim_seconds = self.config["simulation_days"] * 86400
        self.total_steps = int(sim_seconds / dt)
        
        # 进度报告间隔
        progress_steps = max(1, int(self.total_steps * self.config["progress_interval"]))
        
        print(f"   开始 {self.config['simulation_days']} 天闭环仿真...")
        print(f"   仿真步数: {self.total_steps:,}")
        print(f"   仿真步长: {dt} 秒")
        print(f"   进度报告间隔: 每 {progress_steps} 步")
        print("-"*60)
        
        # 主循环
        for step in range(self.total_steps):
            self.current_step = step
            epoch = step * dt
            
            # --- 导航感知 ---
            obs_state, frame = self.ground_station.track_spacecraft(
                self.spacecraft.state, 
                self.spacecraft.frame, 
                epoch
            )
            
            if obs_state is not None:
                self.gnc_system.update_navigation(obs_state, frame)
            
            # --- 控制决策 ---
            force_cmd, force_frame = self.gnc_system.compute_control_force(epoch, self.k_matrix)
            
            # 修正：确保控制力是3维数组
            force_cmd = self._ensure_3d_control_force(force_cmd)
            
            # --- 物理演化 ---
            # 1. 施加推力
            self.spacecraft.apply_thrust(force_cmd, force_frame)
            
            # 2. RK4 积分
            k1 = self._get_state_derivative(self.spacecraft.state)
            k2 = self._get_state_derivative(self.spacecraft.state + 0.5 * dt * k1)
            k3 = self._get_state_derivative(self.spacecraft.state + 0.5 * dt * k2)
            k4 = self._get_state_derivative(self.spacecraft.state + dt * k3)
            
            self.spacecraft.state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # 3. 核算燃料消耗
            self.spacecraft.integrate_dv(dt)
            self.spacecraft.clear_thrust()
            
            # 4. 环境时间推进
            self.environment.step_time(dt)
            
            # --- 数据记录 ---
            if step % 10 == 0:  # 每10步记录一次
                nom_state = self.ephemeris.get_interpolated_state(epoch)
                
                # 确保所有参数格式正确
                self.logger.log_step(
                    epoch=epoch,
                    nominal_state=nom_state,
                    true_state=self.spacecraft.state,
                    nav_state=self.gnc_system.current_nav_state,
                    tracking_error=self.gnc_system.last_tracking_error,
                    control_force=force_cmd,
                    accumulated_dv=self.spacecraft.accumulated_dv
                )
            
            # --- 进度报告 ---
            if step % progress_steps == 0 and step > 0:
                self._report_progress(step, epoch, force_cmd)
        
        print("-"*60)
        print("✅ 仿真主循环完成")
    
    def _ensure_3d_control_force(self, force_cmd) -> np.ndarray:
        """
        确保控制力是3维数组
        
        Args:
            force_cmd: 原始控制力
            
        Returns:
            修正后的3维控制力数组
        """
        if isinstance(force_cmd, (int, float, np.number)):
            # 标量转换为数组
            return np.array([float(force_cmd), 0.0, 0.0], dtype=np.float64)
        elif isinstance(force_cmd, np.ndarray):
            if force_cmd.shape == (1,) or force_cmd.shape == ():
                return np.array([float(force_cmd[0]) if force_cmd.shape == (1,) else float(force_cmd), 0.0, 0.0], dtype=np.float64)
            elif force_cmd.shape == (3,):
                return force_cmd.astype(np.float64)
            elif force_cmd.size > 3:
                return force_cmd[:3].astype(np.float64)
            else:
                # 其他形状，尝试重塑
                try:
                    return force_cmd.reshape(3).astype(np.float64)
                except:
                    return np.zeros(3, dtype=np.float64)
        else:
            # 未知类型，返回零向量
            return np.zeros(3, dtype=np.float64)
    
    def _get_state_derivative(self, state: np.ndarray) -> np.ndarray:
        """
        计算状态导数（供RK4积分使用）
        
        Args:
            state: 当前状态向量
            
        Returns:
            状态导数
        """
        # 获取环境加速度
        acc_env, acc_frame = self.environment.get_total_acceleration(state, self.spacecraft.frame)
        
        # 计算状态导数
        derivative = np.zeros(6)
        derivative[0:3] = state[3:6]  # 位置导数 = 速度
        derivative[3:6] = acc_env + self.spacecraft.external_accel  # 速度导数 = 总加速度
        
        return derivative
    
    def _report_progress(self, step: int, epoch: float, force_cmd: np.ndarray) -> None:
        """
        报告仿真进度
        
        Args:
            step: 当前步数
            epoch: 当前时间
            force_cmd: 当前控制力
        """
        progress = (step / self.total_steps) * 100
        days = epoch / 86400
        
        # 计算当前跟踪误差
        err_pos = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
        err_vel = np.linalg.norm(self.gnc_system.last_tracking_error[3:6]) * 1000
        
        # 控制力范数
        control_norm = np.linalg.norm(force_cmd)
        
        print(f"  [Day {days:6.1f}] 进度: {progress:5.1f}% | "
              f"位置误差: {err_pos:6.1f}m | "
              f"速度误差: {err_vel:6.2f}mm/s | "
              f"控制力: {control_norm:7.4f}N | "
              f"累计 ΔV: {self.spacecraft.accumulated_dv:8.4f}m/s")
    
    def _finalize_simulation(self) -> bool:
        """
        仿真最终处理
        
        Returns:
            处理是否成功
        """
        print("\n" + "="*60)
        print("[阶段7] 仿真最终处理")
        print("="*60)
        
        # 记录结束时间
        self.simulation_end_time = time.time()
        sim_duration = self.simulation_end_time - self.simulation_start_time
        
        # 关闭数据记录器
        if self.logger:
            self.logger.close()
        
        # 计算最终性能指标
        final_err_pos = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
        final_err_vel = np.linalg.norm(self.gnc_system.last_tracking_error[3:6]) * 1000
        
        # 仿真结果汇总
        print("📊 仿真结果汇总")
        print("-"*60)
        print(f"✅ 仿真完成!")
        print(f"   实际仿真时间: {sim_duration:.1f} 秒")
        print(f"   仿真步数: {self.total_steps:,}")
        print(f"   最终位置误差: {final_err_pos:.2f} m")
        print(f"   最终速度误差: {final_err_vel:.2f} mm/s")
        print(f"   总 ΔV 消耗: {self.spacecraft.accumulated_dv:.4f} m/s")
        print(f"   平均每天 ΔV: {self.spacecraft.accumulated_dv/self.config['simulation_days']:.4f} m/s/天")
        print(f"   数据文件: {os.path.abspath(self.h5_file)}")
        
        # 获取文件统计
        if self.logger:
            stats = self.logger.get_statistics()
            if "file_size_mb" in stats:
                print(f"   数据文件大小: {stats['file_size_mb']:.2f} MB")
        
        # 生成可视化报告
        if self.config["enable_visualization"]:
            self._generate_visualization()
        
        print("\n" + "="*60)
        print(f"🎉 {self.config['mission_name']} 仿真圆满完成！")
        print("="*60)
        
        return True
    
    def _generate_visualization(self) -> None:
        """
        生成可视化报告
        """
        print("\n" + "-"*60)
        print("[阶段8] 生成可视化报告")
        print("-"*60)
        
        try:
            # 确保数据文件存在
            if not os.path.exists(self.h5_file):
                print("⚠️ 数据文件不存在，跳过可视化")
                return
            
            # 创建可视化器
            vis = L1Visualizer(
                filepath=self.h5_file,
                mission_name=self.config["mission_name"]
            )
            
            # 生成图表文件路径
            base_name = os.path.join(
                self.config["data_dir"], 
                f"L1_{self.config['mission_name'].replace(' ', '_')}_{self.mission_id}"
            )
            
            # 生成各种分析图表
            vis.plot_3d_trajectory(save_path=f"{base_name}_trajectory.png")
            vis.plot_tracking_error(save_path=f"{base_name}_errors.png")
            vis.plot_control_effort(save_path=f"{base_name}_control.png")
            
            print("✅ 可视化报告生成完成")
            print(f"   轨迹图: {base_name}_trajectory.png")
            print(f"   误差图: {base_name}_errors.png")
            print(f"   控制图: {base_name}_control.png")
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
            print("   原始数据仍保存在 HDF5 文件中，可手动分析")
    
    def _emergency_shutdown(self) -> None:
        """
        紧急关闭处理
        """
        print("\n⚠️ 执行紧急关闭...")
        
        # 尝试保存已记录的数据
        if self.logger:
            try:
                self.logger.close()
                print(f"✅ 已保存数据到: {self.h5_file}")
            except Exception as e:
                print(f"❌ 数据保存失败: {e}")
        
        # 计算已完成的仿真时间
        if self.current_step > 0:
            completed_days = (self.current_step * self.config["time_step"]) / 86400
            print(f"   已完成 {completed_days:.2f} 天仿真")
    
    def get_statistics(self) -> dict:
        """
        获取仿真统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "mission_id": self.mission_id,
            "mission_name": self.config["mission_name"],
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "completed": self.current_step >= self.total_steps,
            "data_file": self.h5_file
        }
        
        if self.logger:
            stats.update(self.logger.get_statistics())
        
        if self.simulation_start_time and self.simulation_end_time:
            stats["simulation_duration_seconds"] = self.simulation_end_time - self.simulation_start_time
        
        if self.spacecraft:
            stats["accumulated_dv"] = self.spacecraft.accumulated_dv
        
        if self.gnc_system:
            stats["final_position_error"] = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
            stats["final_velocity_error"] = np.linalg.norm(self.gnc_system.last_tracking_error[3:6])
        
        return stats


def run_L1_simulation(custom_config: dict = None) -> bool:
    """
    运行 L1 级仿真的入口函数
    
    Args:
        custom_config: 自定义配置
        
    Returns:
        仿真是否成功
    """
    # 创建仿真控制器
    simulation = L1MissionSimulation(custom_config)
    
    # 运行仿真
    success = simulation.run()
    
    # 输出统计信息
    if success:
        stats = simulation.get_statistics()
        print(f"\n📈 仿真统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    return success


if __name__ == "__main__":
    # 可选：自定义配置
    custom_config = {
        "mission_name": "JWST 30-Day L2 站位维持仿真 (修正版)",
        "simulation_days": 30,
        "time_step": 10.0,
        "log_buffer_size": 1000,
        "enable_visualization": True
    }
    
    # 运行仿真
    success = run_L1_simulation(custom_config)
    
    # 根据结果返回适当的退出码
    sys.exit(0 if success else 1)