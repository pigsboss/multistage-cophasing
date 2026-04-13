"""
求解器框架 - 基础类定义

基于 map-reduce 架构的通用求解器框架，支持多种并行计算方式和算法。
遵循 MCPC 编码标准：UTF-8 编码，运行时输出使用英文。
"""
from __future__ import annotations

import abc
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable
import numpy as np


class SolverStatus(Enum):
    """求解器状态枚举"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeviceType(Enum):
    """计算设备类型枚举"""
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    FPGA = "fpga"
    AUTO = "auto"


class ParallelBackend(Enum):
    """并行计算后端枚举"""
    NUMPY = "numpy"
    NUMBA = "numba"
    OPENCL = "opencl"
    CUDA = "cuda"
    MULTIPROCESSING = "multiprocessing"
    THREADING = "threading"
    MPI = "mpi"


class AlgorithmType(Enum):
    """算法类型枚举"""
    OPTIMIZATION = "optimization"
    ROOT_FINDING = "root_finding"
    INTEGRATION = "integration"
    DIFFERENTIAL_CORRECTION = "differential_correction"
    MONTE_CARLO = "monte_carlo"
    LINEAR_SOLVER = "linear_solver"


@runtime_checkable
class DeviceInterface(Protocol):
    """设备接口协议（将在_devices.py中实现）"""
    
    @property
    def device_type(self) -> DeviceType:
        """获取设备类型"""
        ...
    
    @property
    def device_name(self) -> str:
        """获取设备名称"""
        ...
    
    def is_available(self) -> bool:
        """检查设备是否可用"""
        ...
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取设备性能指标"""
        ...
    
    def initialize(self) -> bool:
        """初始化设备"""
        ...
    
    def cleanup(self) -> None:
        """清理设备资源"""
        ...
    
    def execute_kernel(self, kernel_name: str, *args, **kwargs) -> Any:
        """执行计算内核"""
        ...


@runtime_checkable
class WorkerInterface(Protocol):
    """工作者接口协议（将在_workers.py中实现）"""
    
    @property
    def algorithm_type(self) -> AlgorithmType:
        """获取算法类型"""
        ...
    
    @property
    def supported_backends(self) -> List[ParallelBackend]:
        """获取支持的并行后端"""
        ...
    
    def validate_input(self, task_data: Any) -> bool:
        """验证输入数据"""
        ...
    
    def execute_task(self, task_data: Any, device: DeviceInterface) -> Any:
        """执行计算任务"""
        ...
    
    def validate_result(self, result: Any) -> bool:
        """验证计算结果"""
        ...
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        ...


@runtime_checkable
class SchedulerInterface(Protocol):
    """调度器接口协议（将在_schedulers.py中实现）"""
    
    def schedule_tasks(self, tasks: List[Any], workers: List[WorkerInterface], 
                       devices: List[DeviceInterface]) -> Dict[Any, Any]:
        """调度任务到工作者和设备"""
        ...
    
    def monitor_progress(self) -> Dict[str, Any]:
        """监控任务进度"""
        ...
    
    def collect_results(self, results: Dict[Any, Any]) -> Any:
        """收集和归并结果"""
        ...
    
    def get_scheduling_metrics(self) -> Dict[str, Any]:
        """获取调度性能指标"""
        ...


@runtime_checkable
class LoggerInterface(Protocol):
    """日志器接口协议（将在_loggers.py中实现）"""
    
    def log(self, level: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """记录日志"""
        ...
    
    def save_checkpoint(self, solver_state: Dict[str, Any]) -> str:
        """保存检查点"""
        ...
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        ...
    
    def save_results(self, results: Any, metadata: Dict[str, Any]) -> str:
        """保存结果数据"""
        ...
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        ...


class SolverBase(ABC):
    """
    求解器基类 - 遵循 map-reduce 架构
    
    职责：
    1. 管理整个求解流程
    2. 协调设备、工作者、调度器和日志器
    3. 提供统一的接口给用户
    4. 支持断点续算和状态恢复
    """
    
    def __init__(
        self,
        algorithm_type: AlgorithmType,
        device_type: DeviceType = DeviceType.AUTO,
        parallel_backend: ParallelBackend = ParallelBackend.NUMPY,
        max_workers: int = 4,
        timeout: Optional[float] = None,
        checkpoint_interval: float = 60.0,
        verbose: bool = True
    ):
        """
        初始化求解器
        
        Args:
            algorithm_type: 算法类型
            device_type: 计算设备类型
            parallel_backend: 并行计算后端
            max_workers: 最大工作线程数
            timeout: 超时时间（秒）
            checkpoint_interval: 检查点保存间隔（秒）
            verbose: 是否输出详细信息
        """
        self.algorithm_type = algorithm_type
        self.device_type = device_type
        self.parallel_backend = parallel_backend
        self.max_workers = max_workers
        self.timeout = timeout
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        
        # 状态管理
        self.status = SolverStatus.INITIALIZED
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.progress: float = 0.0
        
        # 组件实例（将在子类或后续模块中初始化）
        self.devices: List[DeviceInterface] = []
        self.workers: List[WorkerInterface] = []
        self.scheduler: Optional[SchedulerInterface] = None
        self.logger: Optional[LoggerInterface] = None
        
        # 结果存储
        self.results: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        
        # 内部状态
        self._checkpoint_counter: int = 0
        self._last_checkpoint_time: Optional[float] = None
        self._task_counter: int = 0
    
    @abstractmethod
    def _setup_impl(self, problem_data: Any, algorithm_params: Dict[str, Any]) -> bool:
        """
        具体的设置实现（由子类实现）
        
        Args:
            problem_data: 问题数据
            algorithm_params: 算法参数
            
        Returns:
            bool: 设置是否成功
        """
        pass
    
    def setup(self, problem_data: Any, algorithm_params: Dict[str, Any]) -> bool:
        """
        设置求解器，准备计算环境
        
        Args:
            problem_data: 问题数据
            algorithm_params: 算法参数
            
        Returns:
            bool: 设置是否成功
        """
        try:
            self.status = SolverStatus.RUNNING
            self.start_time = time.time()
            
            # 记录开始信息
            if self.verbose:
                print(f"[INFO] Solver setup started")
                print(f"       Algorithm: {self.algorithm_type.value}")
                print(f"       Device type: {self.device_type.value}")
                print(f"       Parallel backend: {self.parallel_backend.value}")
            
            # 初始化组件
            self._initialize_components()
            
            # 执行具体设置
            success = self._setup_impl(problem_data, algorithm_params)
            
            if success and self.verbose:
                print(f"[INFO] Solver setup completed successfully")
            
            return success
            
        except Exception as e:
            self.status = SolverStatus.FAILED
            if self.verbose:
                print(f"[ERROR] Solver setup failed: {str(e)}")
            raise
    
    def _initialize_components(self) -> None:
        """初始化各组件"""
        # 这里留空，由子类或具体实现来填充
        # 例如：扫描设备、创建工作者、初始化调度器等
        pass
    
    @abstractmethod
    def _map_phase(self, input_data: Any) -> List[Any]:
        """
        Map阶段：将输入数据分解为并行任务
        
        Args:
            input_data: 输入数据
            
        Returns:
            List[Any]: 任务列表
        """
        pass
    
    @abstractmethod
    def _reduce_phase(self, partial_results: Dict[Any, Any]) -> Any:
        """
        Reduce阶段：归并部分结果
        
        Args:
            partial_results: 部分结果字典
            
        Returns:
            Any: 最终结果
        """
        pass
    
    def solve(self, input_data: Any) -> Any:
        """
        执行求解 - 遵循 map-reduce 模式
        
        Args:
            input_data: 输入数据
            
        Returns:
            Any: 求解结果
        """
        try:
            if self.status != SolverStatus.RUNNING:
                self.status = SolverStatus.RUNNING
            
            if self.verbose:
                print(f"[INFO] Solving started, progress: {self.progress:.1%}")
            
            # 1. Map 阶段：任务分解
            tasks = self._map_phase(input_data)
            
            if self.verbose:
                print(f"[DEBUG] Generated {len(tasks)} tasks")
            
            # 2. 并行计算阶段
            if self.scheduler and self.workers and self.devices:
                # 使用调度器分配任务
                partial_results = self.scheduler.schedule_tasks(
                    tasks, self.workers, self.devices
                )
            else:
                # 如果没有调度器，则顺序执行
                partial_results = self._sequential_execution(tasks)
            
            # 3. Reduce 阶段：结果归并
            final_result = self._reduce_phase(partial_results)
            
            # 保存结果
            self.results = final_result
            self.progress = 1.0
            self.end_time = time.time()
            self.status = SolverStatus.COMPLETED
            
            # 记录完成信息
            if self.verbose:
                elapsed = self.end_time - self.start_time if self.start_time else 0
                print(f"[INFO] Solving completed")
                print(f"       Status: {self.status.value}")
                print(f"       Elapsed time: {elapsed:.2f}s")
                print(f"       Number of tasks: {len(tasks)}")
            
            return final_result
            
        except Exception as e:
            self.status = SolverStatus.FAILED
            if self.verbose:
                print(f"[ERROR] Solving failed: {str(e)}")
            
            # 尝试保存检查点以便恢复
            self._save_checkpoint()
            raise
    
    def _sequential_execution(self, tasks: List[Any]) -> Dict[Any, Any]:
        """顺序执行任务（用于测试或小规模问题）"""
        results = {}
        total_tasks = len(tasks)
        
        for i, task in enumerate(tasks):
            try:
                # 使用第一个工作者执行任务
                if self.workers:
                    result = self.workers[0].execute_task(task, self.devices[0] if self.devices else None)
                    results[i] = result
                else:
                    # 如果没有工作者，直接返回任务本身（用于测试）
                    results[i] = task
                
                # 更新进度
                self.progress = (i + 1) / total_tasks
                
                # 定期保存检查点
                self._maybe_save_checkpoint()
                
                if self.verbose and i % max(1, total_tasks // 10) == 0:
                    print(f"[DEBUG] Task {i+1}/{total_tasks} completed, progress: {self.progress:.1%}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Task {i} failed: {str(e)}")
        
        return results
    
    def _maybe_save_checkpoint(self) -> None:
        """根据需要保存检查点"""
        if not self.logger:
            return
        
        current_time = time.time()
        if (self._last_checkpoint_time is None or 
            current_time - self._last_checkpoint_time >= self.checkpoint_interval):
            self._save_checkpoint()
    
    def _save_checkpoint(self) -> None:
        """保存求解器状态到检查点"""
        if not self.logger:
            return
        
        solver_state = {
            "status": self.status.value,
            "progress": self.progress,
            "results": self.results,
            "metadata": self.metadata,
            "timestamp": time.time(),
            "checkpoint_id": f"checkpoint_{self._checkpoint_counter:04d}"
        }
        
        checkpoint_id = self.logger.save_checkpoint(solver_state)
        self._checkpoint_counter += 1
        self._last_checkpoint_time = time.time()
        
        if self.verbose:
            print(f"[DEBUG] Checkpoint saved: {checkpoint_id}")
    
    def resume_from_checkpoint(self, checkpoint_id: str) -> bool:
        """
        从检查点恢复求解
        
        Args:
            checkpoint_id: 检查点ID
            
        Returns:
            bool: 恢复是否成功
        """
        try:
            if not self.logger:
                return False
            
            solver_state = self.logger.load_checkpoint(checkpoint_id)
            if not solver_state:
                return False
            
            # 恢复状态
            self.status = SolverStatus(solver_state.get("status", SolverStatus.INITIALIZED.value))
            self.progress = solver_state.get("progress", 0.0)
            self.results = solver_state.get("results")
            self.metadata = solver_state.get("metadata", {})
            
            if self.verbose:
                print(f"[INFO] Resumed from checkpoint: {checkpoint_id}")
                print(f"       Progress: {self.progress:.1%}")
                print(f"       Status: {self.status.value}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Failed to resume from checkpoint: {str(e)}")
            return False
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取求解器状态报告"""
        elapsed = 0.0
        if self.start_time:
            if self.end_time:
                elapsed = self.end_time - self.start_time
            else:
                elapsed = time.time() - self.start_time
        
        return {
            "status": self.status.value,
            "progress": self.progress,
            "elapsed_time": elapsed,
            "algorithm_type": self.algorithm_type.value,
            "device_type": self.device_type.value,
            "parallel_backend": self.parallel_backend.value,
            "num_devices": len(self.devices),
            "num_workers": len(self.workers),
            "has_scheduler": self.scheduler is not None,
            "has_logger": self.logger is not None,
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        # 清理设备
        for device in self.devices:
            if hasattr(device, 'cleanup'):
                device.cleanup()
        
        # 更新状态
        if self.status == SolverStatus.RUNNING:
            self.status = SolverStatus.CANCELLED
        
        if self.verbose:
            print(f"[INFO] Solver cleanup completed")
    
    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动清理"""
        self.cleanup()


class OptimizationSolver(SolverBase):
    """优化问题求解器基类"""
    
    def __init__(self, **kwargs):
        kwargs['algorithm_type'] = AlgorithmType.OPTIMIZATION
        super().__init__(**kwargs)
    
    def _setup_impl(self, problem_data: Any, algorithm_params: Dict[str, Any]) -> bool:
        # 优化问题特定设置
        self.metadata.update({
            "problem_type": "optimization",
            "algorithm_params": algorithm_params,
            "objective_function": problem_data.get("objective_function", "unknown"),
        })
        return True
    
    def _map_phase(self, input_data: Any) -> List[Any]:
        # 优化问题的任务分解策略
        if isinstance(input_data, dict) and 'parameters' in input_data:
            # 参数扫描优化
            params = input_data['parameters']
            return self._create_parameter_tasks(params)
        elif isinstance(input_data, dict) and 'initial_guesses' in input_data:
            # 多初始点优化
            guesses = input_data['initial_guesses']
            return [{"initial_guess": guess} for guess in guesses]
        else:
            # 单次优化
            return [input_data]
    
    def _create_parameter_tasks(self, parameters: Dict[str, List[Any]]) -> List[Dict]:
        """创建参数扫描任务"""
        # 实现参数组合生成
        import itertools
        keys = list(parameters.keys())
        values = list(parameters.values())
        
        tasks = []
        for combo in itertools.product(*values):
            task = dict(zip(keys, combo))
            tasks.append(task)
        
        return tasks
    
    def _reduce_phase(self, partial_results: Dict[Any, Any]) -> Any:
        # 找到最优结果
        best_result = None
        best_value = float('inf')
        
        for task_id, result in partial_results.items():
            if isinstance(result, dict):
                # 尝试获取目标函数值
                if 'objective_value' in result:
                    obj_val = result['objective_value']
                elif 'value' in result:
                    obj_val = result['value']
                else:
                    continue
                
                if obj_val < best_value:
                    best_value = obj_val
                    best_result = result
        
        return {
            "optimal_solution": best_result,
            "optimal_value": best_value,
            "num_evaluations": len(partial_results),
        }


class IntegrationSolver(SolverBase):
    """积分问题求解器基类"""
    
    def __init__(self, **kwargs):
        kwargs['algorithm_type'] = AlgorithmType.INTEGRATION
        super().__init__(**kwargs)
    
    def _setup_impl(self, problem_data: Any, algorithm_params: Dict[str, Any]) -> bool:
        # 积分问题特定设置
        self.metadata.update({
            "problem_type": "integration",
            "algorithm_params": algorithm_params,
            "integrator": algorithm_params.get('integrator', 'rk4'),
            "relative_tolerance": algorithm_params.get('rtol', 1e-6),
            "absolute_tolerance": algorithm_params.get('atol', 1e-9),
        })
        return True
    
    def _map_phase(self, input_data: Any) -> List[Any]:
        # 时间分段积分
        if isinstance(input_data, dict):
            # 支持时间分段
            if 'time_span' in input_data:
                t_span = input_data['time_span']
                num_segments = input_data.get('num_segments', self.max_workers)
                return self._split_time_span(t_span, num_segments)
            # 支持多初始状态
            elif 'initial_states' in input_data:
                states = input_data['initial_states']
                return [{"initial_state": state} for state in states]
        
        return [input_data]
    
    def _split_time_span(self, t_span: Tuple[float, float], num_segments: int) -> List[Dict]:
        """将时间区间分割为多个子区间"""
        t_start, t_end = t_span
        segment_length = (t_end - t_start) / num_segments
        
        segments = []
        for i in range(num_segments):
            seg_start = t_start + i * segment_length
            seg_end = seg_start + segment_length if i < num_segments - 1 else t_end
            segments.append({
                "time_span": (seg_start, seg_end),
                "segment_id": i,
                "num_segments": num_segments
            })
        
        return segments
    
    def _reduce_phase(self, partial_results: Dict[Any, Any]) -> Any:
        # 合并积分结果
        if all(isinstance(r, dict) and 'states' in r for r in partial_results.values()):
            # 按时间顺序合并状态轨迹
            all_states = []
            all_times = []
            
            for seg_id in sorted(partial_results.keys()):
                result = partial_results[seg_id]
                all_states.extend(result['states'])
                all_times.extend(result['times'])
            
            return {
                "states": all_states,
                "times": all_times,
                "num_segments": len(partial_results),
            }
        else:
            # 简单合并所有结果
            return {
                "results": list(partial_results.values()),
                "num_results": len(partial_results),
            }


class DifferentialCorrectionSolver(SolverBase):
    """微分校正求解器基类"""
    
    def __init__(self, **kwargs):
        kwargs['algorithm_type'] = AlgorithmType.DIFFERENTIAL_CORRECTION
        super().__init__(**kwargs)
    
    def _setup_impl(self, problem_data: Any, algorithm_params: Dict[str, Any]) -> bool:
        # 微分校正特定设置
        self.metadata.update({
            "problem_type": "differential_correction",
            "algorithm_params": algorithm_params,
            "max_iterations": algorithm_params.get('max_iterations', 100),
            "tolerance": algorithm_params.get('tolerance', 1e-8),
            "jacobian_method": algorithm_params.get('jacobian_method', 'numerical'),
        })
        return True
    
    def _map_phase(self, input_data: Any) -> List[Any]:
        # 微分校正通常是串行算法，但可以并行化雅可比计算或参数扫描
        if isinstance(input_data, dict) and 'correction_parameters' in input_data:
            # 多参数校正
            params_list = input_data['correction_parameters']
            return [{"parameters": params} for params in params_list]
        else:
            return [input_data]
    
    def _reduce_phase(self, partial_results: Dict[Any, Any]) -> Any:
        # 收集所有校正结果
        corrections = []
        
        for task_id, result in partial_results.items():
            if isinstance(result, dict) and 'correction_result' in result:
                corrections.append(result['correction_result'])
            else:
                corrections.append(result)
        
        return {
            "corrections": corrections,
            "num_corrections": len(corrections),
            "successful": len(corrections) > 0,
        }


class MonteCarloSolver(SolverBase):
    """蒙特卡洛求解器基类"""
    
    def __init__(self, **kwargs):
        kwargs['algorithm_type'] = AlgorithmType.MONTE_CARLO
        super().__init__(**kwargs)
    
    def _setup_impl(self, problem_data: Any, algorithm_params: Dict[str, Any]) -> bool:
        # 蒙特卡洛特定设置
        self.metadata.update({
            "problem_type": "monte_carlo",
            "algorithm_params": algorithm_params,
            "num_samples": algorithm_params.get('num_samples', 1000),
            "random_seed": algorithm_params.get('random_seed', None),
            "distribution_type": algorithm_params.get('distribution', 'uniform'),
        })
        return True
    
    def _map_phase(self, input_data: Any) -> List[Any]:
        # 创建采样任务
        num_samples = self.metadata.get("num_samples", 1000)
        num_workers = min(self.max_workers, len(self.workers) if self.workers else 1)
        samples_per_worker = num_samples // num_workers
        
        tasks = []
        for i in range(num_workers):
            # 最后一个worker处理剩余的样本
            if i == num_workers - 1:
                samples = num_samples - i * samples_per_worker
            else:
                samples = samples_per_worker
            
            tasks.append({
                "task_id": i,
                "num_samples": samples,
                "worker_id": i,
                "random_seed": self.metadata.get("random_seed", None)
            })
        
        return tasks
    
    def _reduce_phase(self, partial_results: Dict[Any, Any]) -> Any:
        # 合并所有采样结果
        all_samples = []
        all_statistics = []
        
        for task_id, result in partial_results.items():
            if isinstance(result, dict):
                if 'samples' in result:
                    all_samples.extend(result['samples'])
                if 'statistics' in result:
                    all_statistics.append(result['statistics'])
        
        # 计算总体统计
        if all_samples:
            import numpy as np
            samples_array = np.array(all_samples)
            
            overall_stats = {
                "mean": np.mean(samples_array, axis=0).tolist(),
                "std": np.std(samples_array, axis=0).tolist(),
                "min": np.min(samples_array, axis=0).tolist(),
                "max": np.max(samples_array, axis=0).tolist(),
                "num_samples": len(all_samples),
            }
        else:
            overall_stats = {}
        
        return {
            "samples": all_samples,
            "worker_statistics": all_statistics,
            "overall_statistics": overall_stats,
            "num_workers": len(partial_results),
        }
