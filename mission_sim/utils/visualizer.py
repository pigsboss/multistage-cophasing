# mission_sim/utils/visualizer.py
import os
import h5py
import matplotlib.pyplot as plt

class BaseVisualizer:
    """
    仿真可视化基类 (Infrastructure)
    职责：为 L1 到 L5 的所有可视化子类提供通用的文件读取、图表工厂和统一的保存逻辑。
    """
    def __init__(self, filepath: str):
        """
        :param filepath: HDF5 仿真数据文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[Visualizer] 找不到数据文件: {filepath}")
            
        self.filepath = filepath
        
        # 设定全局一致的现代科研绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        self.default_figsize = (10, 6)

    def create_figure(self, rows: int, cols: int, title: str, figsize: tuple = None):
        """
        统一的图表生成工厂。
        确保所有子类生成的图表具有一致的标题格式和子图间距。
        """
        if figsize is None:
            figsize = self.default_figsize
            
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        return fig, axes

    def save_plot(self, fig_obj: plt.Figure, save_path: str):
        """
        统一的图表保存与内存清理逻辑。
        防止在批量生成图表时发生内存泄漏 (OOM)。
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 自动调整边距以适应全局标题
        fig_obj.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_obj.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 显式关闭当前 Figure 对象释放内存 
        plt.close(fig_obj) 
        
        print(f"📊 图表已保存: {save_path}")

    def load_dataset(self, dataset_path: str):
        """
        通用的 HDF5 数据集安全提取器。
        :param dataset_path: HDF5 内部的层级路径 (例如 'epochs' 或 'Formation/rel_state_lvlh')
        """
        with h5py.File(self.filepath, 'r') as f:
            if dataset_path not in f:
                raise KeyError(f"[Visualizer] HDF5 文件中不存在数据集: {dataset_path}")
            # [()] 用于将 HDF5 dataset 直接提取为 Numpy array 在内存中操作
            return f[dataset_path][()]
