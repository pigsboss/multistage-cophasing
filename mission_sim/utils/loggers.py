import os
import h5py
import numpy as np
from enum import Enum

class HDF5Logger:
    """
    航天高维时序数据 HDF5 记录器
    特性：
    1. 增量写入 (Chunking)：避免 OOM，支持无限长仿真。
    2. 树状层级 (Hierarchy)：支持按航天器 ID、子系统对数据进行分类。
    3. 元数据绑定 (Attributes)：支持将坐标系、物理常数刻印在数据结构上。
    """
    def __init__(self, filepath: str, flush_interval: int = 100):
        """
        :param filepath: 目标 .h5 或 .hdf5 文件路径
        :param flush_interval: 缓存多少个时间步后集中进行一次磁盘 I/O 写入
        """
        self.filepath = filepath
        self.flush_interval = flush_interval
        
        # 确保目标路径的文件夹存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 以截断写模式打开文件 (实际工程中可改为按时间戳生成文件名)
        self.file = h5py.File(self.filepath, 'w')
        
        self._buffer = {}
        self._buffer_count = 0

    def set_metadata(self, path: str, meta_dict: dict):
        """
        为特定的 Group 或 Dataset 永久打上属性标签 (元数据)。
        这是维持坐标系契约的关键：把 CoordinateFrame 存进文件！
        
        :param path: HDF5 内部路径，如 "/" 或 "/Probe-Alpha"
        :param meta_dict: 字典格式的属性键值对
        """
        if path not in self.file:
            self.file.create_group(path)
            
        target = self.file[path]
        for key, value in meta_dict.items():
            # 枚举类无法直接存入 HDF5，需自动转换为字符串名称
            if isinstance(value, Enum):
                value = value.name
            target.attrs[key] = value

    def log(self, group: str, dataset: str, data: np.ndarray):
        """
        将一帧数据推入内存缓存。
        
        :param group: HDF5 组路径，如 "Probe-Alpha"
        :param dataset: 数据集名称，如 "true_state"
        :param data: 当前步的数据 (标量、向量或张量均可)
        """
        # 强制转换为至少一维的 np.float64 数组，保持物理量纲严谨
        data_np = np.atleast_1d(np.array(data, dtype=np.float64))
        
        if group not in self._buffer:
            self._buffer[group] = {}
        if dataset not in self._buffer[group]:
            self._buffer[group][dataset] = []
            
        self._buffer[group][dataset].append(data_np)
        
    def step(self):
        """
        通知 Logger 仿真时间前进了一步。如果缓存堆积满，则自动落盘。
        """
        self._buffer_count += 1
        if self._buffer_count >= self.flush_interval:
            self.flush()

    def flush(self):
        """
        将内存缓存集中写入 HDF5 磁盘文件 (增量 Appending)。
        """
        if self._buffer_count == 0:
            return
            
        for grp_name, dsets in self._buffer.items():
            if grp_name not in self.file:
                self.file.create_group(grp_name)
            grp = self.file[grp_name]
            
            for dset_name, data_list in dsets.items():
                if not data_list:
                    continue
                    
                # 将缓存的一批数据拼成矩阵: shape (N, ...)
                chunk_data = np.stack(data_list, axis=0)
                
                if dset_name not in grp:
                    # 首次遇到该数据集：开启 maxshape，允许沿时间轴(axis=0)无限延伸
                    data_shape = chunk_data.shape[1:]
                    grp.create_dataset(
                        dset_name,
                        data=chunk_data,
                        maxshape=(None, *data_shape),
                        chunks=True  # 开启分块存储，极大提升读写性能
                    )
                else:
                    # 追加写入：拉长数据集，放入新数据
                    dset = grp[dset_name]
                    old_size = dset.shape[0]
                    new_size = old_size + chunk_data.shape[0]
                    
                    dset.resize(new_size, axis=0)
                    dset[old_size:new_size] = chunk_data
                    
                # 清空已落盘的缓存
                self._buffer[grp_name][dset_name] = []
                
        self._buffer_count = 0

    def close(self):
        """确保在程序结束或异常退出时，残余缓存全部落盘。"""
        self.flush()
        self.file.close()
        
    def __enter__(self):
        """支持 Python 的 with 语句上下文管理"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
