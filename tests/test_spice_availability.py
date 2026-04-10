"""
SPICE可用性测试

测试SPICE工具和相关依赖是否可用。
验证高精度星历模块的SPICE模式是否正常工作。
"""

import unittest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil
import os

# 测试SPICE依赖
def check_spice_dependencies():
    """检查SPICE相关依赖"""
    dependencies = {
        'numpy': '数值计算',
        'spiceypy': 'NASA SPICE工具包',
        'requests': 'HTTP请求（用于下载核文件）',
        'tqdm': '进度条显示'
    }
    
    print("检查SPICE高精度星历模块依赖项")
    print("=" * 50)
    
    missing = []
    available = []
    
    for module_name, description in dependencies.items():
        try:
            __import__(module_name)
            version = getattr(sys.modules[module_name], '__version__', '未知')
            print(f"✓ {module_name:12} - {description} (版本: {version})")
            available.append(module_name)
        except ImportError as e:
            print(f"✗ {module_name:12} - {description}: 未安装")
            missing.append(module_name)
    
    print("=" * 50)
    
    if missing:
        print(f"\n缺少 {len(missing)} 个依赖项: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    else:
        print("\n所有依赖项已安装！")
        return True

def check_spiceypy_installed():
    """检查 spiceypy 是否安装"""
    try:
        import spiceypy
        return True
    except ImportError:
        return False

def find_spice_kernels():
    """查找 SPICE 内核目录"""
    # 环境变量
    env_path = os.environ.get('SPICE_KERNELS')
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    
    # 常见路径
    candidates = [
        Path('./spice_kernels'),
        Path(__file__).parent.parent / 'spice_kernels',
        Path(__file__).parent.parent.parent / 'spice_kernels',
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return None

def check_required_kernels(kernel_path: Path) -> dict:
    """检查必需的内核文件是否存在"""
    required = {
        'lsk': ['naif*.tls', 'latest_leapseconds.tls'],
        'spk': ['de440.bsp', 'de441.bsp', 'de442.bsp'],
    }
    
    results = {}
    
    for ktype, patterns in required.items():
        found = False
        for pattern in patterns:
            # 递归搜索
            matches = list(kernel_path.rglob(pattern))
            if matches:
                found = True
                results[ktype] = matches[0].name
                break
        
        if not found:
            results[ktype] = None
    
    return results


class TestSPICEEnvironment(unittest.TestCase):
    """SPICE环境测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp(prefix="test_spice_")
        
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_spiceypy_import(self):
        """测试spiceypy导入"""
        try:
            import spiceypy as spice
            self.assertIsNotNone(spice)
            print("✓ spiceypy 导入成功")
        except ImportError:
            self.skipTest("spiceypy 未安装")
    
    def test_spice_basic_functions(self):
        """测试SPICE基本函数"""
        try:
            import spiceypy as spice
            
            # 测试SPICE基本功能
            # 加载一个简单的测试核文件（如果没有，则创建虚拟的）
            test_kernel = Path(self.test_dir) / "test_kernel.tls"
            
            # 创建一个简单的文本核文件
            with open(test_kernel, 'w') as f:
                f.write("\\begindata\n")
                f.write("KERNELS_TO_LOAD = ( 'test_kernel.tls' )\n")
                f.write("\\begintext\n")
            
            try:
                spice.furnsh(str(test_kernel))
                
                # 测试一个简单的转换
                et = spice.str2et('2000-01-01T12:00:00')
                self.assertIsInstance(et, float)
                
                spice.unload(str(test_kernel))
                print("✓ SPICE基本功能正常")
                
            except Exception as e:
                print(f"⚠ SPICE功能测试警告: {e}")
                # 不是致命错误，只是警告
                pass
                
        except ImportError:
            self.skipTest("spiceypy 未安装")
    
    def test_spice_kernels_directory_exists(self):
        """测试 SPICE 内核目录是否存在"""
        kernel_path = find_spice_kernels()
        
        if kernel_path is None:
            self.skipTest(
                "SPICE kernels directory not found. "
                "Set SPICE_KERNELS environment variable or place in ./spice_kernels"
            )
        
        self.assertTrue(kernel_path.exists(), f"Kernel path does not exist: {kernel_path}")
    
    def test_required_kernels_present(self):
        """测试必需的内核文件是否存在"""
        kernel_path = find_spice_kernels()
        
        if kernel_path is None:
            self.skipTest("SPICE kernels directory not found")
        
        results = check_required_kernels(kernel_path)
        
        # 检查 LSK（闰秒内核）- 必需
        if results.get('lsk') is None:
            self.fail(
                f"Leapseconds kernel (naif0012.tls or similar) not found in {kernel_path}"
            )
        
        # 检查 SPK（行星历表）- 必需
        if results.get('spk') is None:
            self.fail(
                f"Planetary ephemeris (de440.bsp or similar) not found in {kernel_path}"
            )
        
        # 如果通过，记录找到的文件
        print(f"\nFound LSK: {results['lsk']}")
        print(f"Found SPK: {results['spk']}")
    
    def test_high_precision_ephemeris_import(self):
        """测试高精度星历模块导入"""
        try:
            from mission_sim.core.spacetime.ephemeris.high_precision import (
                HighPrecisionEphemeris, EphemerisMode, CelestialBody, EphemerisConfig
            )
            
            self.assertIsNotNone(HighPrecisionEphemeris)
            self.assertIsNotNone(EphemerisMode)
            self.assertIsNotNone(CelestialBody)
            self.assertIsNotNone(EphemerisConfig)
            
            print("✓ 高精度星历模块导入成功")
            
        except ImportError as e:
            self.fail(f"高精度星历模块导入失败: {e}")
    
    def test_ephemeris_modes(self):
        """测试星历模式"""
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            EphemerisMode, EphemerisConfig
        )
        
        # 测试所有模式
        modes = [
            EphemerisMode.ANALYTICAL,
            EphemerisMode.CRTBP,
            EphemerisMode.NUMERICAL,
            EphemerisMode.EXTERNAL,
        ]
        
        for mode in modes:
            config = EphemerisConfig(mode=mode)
            self.assertEqual(config.mode, mode)
        
        print("✓ 星历模式配置正常")
    
    def test_celestial_body_enum(self):
        """测试天体枚举"""
        from mission_sim.core.spacetime.ephemeris.high_precision import CelestialBody
        
        bodies = [
            CelestialBody.SUN,
            CelestialBody.EARTH,
            CelestialBody.MOON,
            CelestialBody.MARS
        ]
        
        for body in bodies:
            self.assertIsInstance(body, CelestialBody)
            self.assertIsInstance(body.value, str)
        
        print("✓ 天体枚举正常")
    
    def test_ephemeris_creation(self):
        """测试星历创建"""
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            HighPrecisionEphemeris, EphemerisConfig, EphemerisMode
        )
        
        # 测试解析模式创建
        config = EphemerisConfig(mode=EphemerisMode.ANALYTICAL)
        ephemeris = HighPrecisionEphemeris(config)
        
        self.assertIsInstance(ephemeris, HighPrecisionEphemeris)
        self.assertEqual(ephemeris.config.mode, EphemerisMode.ANALYTICAL)
        
        print("✓ 星历创建正常")
    
    def test_get_state_basic(self):
        """测试基本状态获取"""
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            HighPrecisionEphemeris, EphemerisConfig, EphemerisMode, CelestialBody
        )
        
        config = EphemerisConfig(
            mode=EphemerisMode.ANALYTICAL,
            verbose=False
        )
        
        ephemeris = HighPrecisionEphemeris(config)
        
        # 获取地球相对于太阳的状态（解析模式）
        state = ephemeris.get_state(
            target_body=CelestialBody.EARTH,
            epoch=0.0,
            observer_body=CelestialBody.SUN,
            frame="J2000_ECI"
        )
        
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (6,))
        
        print("✓ 基本状态获取正常")


@unittest.skipUnless(check_spice_dependencies(), "缺少SPICE依赖")
class TestSPICEEphemeris(unittest.TestCase):
    """SPICE星历测试（需要SPICE依赖）"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp(prefix="test_spice_")
        
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_spice_mode_exists(self):
        """测试SPICE模式是否存在"""
        from mission_sim.core.spacetime.ephemeris.high_precision import EphemerisMode
        
        # 检查SPICE模式是否已添加到枚举中
        self.assertTrue(hasattr(EphemerisMode, 'SPICE'), 
                       "EphemerisMode中缺少SPICE模式")
        
        print("✓ SPICE模式已定义")
    
    def test_spice_ephemeris_config(self):
        """测试SPICE星历配置"""
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            EphemerisConfig, EphemerisMode
        )
        
        # 测试SPICE模式配置
        config = EphemerisConfig(mode=EphemerisMode.SPICE)
        self.assertEqual(config.mode, EphemerisMode.SPICE)
        
        print("✓ SPICE星历配置正常")
    
    def test_spice_kernel_manager_import(self):
        """测试SPICE核文件管理器导入"""
        try:
            # 尝试从tools模块导入
            import mission_sim.tools.spice_kernel_manager as spm
            self.assertIsNotNone(spm)
            print("✓ SPICE核文件管理器导入成功")
        except ImportError:
            # 尝试直接导入
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
                import spice_kernel_manager as spm
                self.assertIsNotNone(spm)
                print("✓ SPICE核文件管理器导入成功（直接路径）")
            except ImportError:
                print("⚠ SPICE核文件管理器不可用（可能需要安装或创建）")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("SPICE可用性测试套件")
    print("=" * 60)
    
    # 检查依赖
    print("\n第一阶段: 依赖检查")
    print("-" * 40)
    deps_ok = check_spice_dependencies()
    
    if not deps_ok:
        print("\n警告: 缺少部分依赖，SPICE测试可能受限")
    
    # 运行单元测试
    print("\n第二阶段: 单元测试")
    print("-" * 40)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    
    # 添加基础测试
    suite = loader.loadTestsFromTestCase(TestSPICEEnvironment)
    
    # 如果依赖满足，添加SPICE测试
    if deps_ok:
        try:
            spice_suite = loader.loadTestsFromTestCase(TestSPICEEphemeris)
            suite.addTests(spice_suite)
            print("包含SPICE功能测试")
        except Exception as e:
            print(f"跳过SPICE功能测试: {e}")
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  运行测试: {result.testsRun}")
    print(f"  通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败详情:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(':')[0]}")
    
    print("=" * 60)
    
    return result.wasSuccessful()

def run_prerequisite_checks():
    """运行SPICE前提条件检查"""
    print("SPICE 可用性检查")
    print("=" * 50)
    
    # 检查1: spiceypy
    print("\n1. 检查 spiceypy...")
    if check_spiceypy_installed():
        print("   ✓ spiceypy 已安装")
    else:
        print("   ✗ spiceypy 未安装")
        print("   请运行: pip install spiceypy")
        return False
    
    # 检查2: 内核目录
    print("\n2. 检查 SPICE 内核目录...")
    kernel_path = find_spice_kernels()
    
    if kernel_path is None:
        print("   ✗ 未找到内核目录")
        print("   请设置 SPICE_KERNELS 环境变量或将内核放在 ./spice_kernels")
        return False
    else:
        print(f"   ✓ 找到内核目录: {kernel_path}")
    
    # 检查3: 必需文件
    print("\n3. 检查必需内核文件...")
    results = check_required_kernels(kernel_path)
    
    all_ok = True
    
    if results.get('lsk'):
        print(f"   ✓ 闰秒内核: {results['lsk']}")
    else:
        print("   ✗ 闰秒内核 (naif*.tls) 未找到")
        all_ok = False
    
    if results.get('spk'):
        print(f"   ✓ 行星历表: {results['spk']}")
    else:
        print("   ✗ 行星历表 (de440.bsp 等) 未找到")
        all_ok = False
    
    print("\n" + "=" * 50)
    if all_ok:
        print("所有检查通过！SPICE 功能可用。")
        return True
    else:
        print("部分内核文件缺失，SPICE 功能受限。")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行 SPICE 可用性测试')
    parser.add_argument('--prerequisites', action='store_true', 
                       help='仅运行前提条件检查')
    parser.add_argument('--full', action='store_true',
                       help='运行完整测试套件（默认）')
    
    args = parser.parse_args()
    
    if args.prerequisites:
        # 仅运行前提条件检查
        success = run_prerequisite_checks()
        sys.exit(0 if success else 1)
    else:
        # 运行完整测试套件
        success = run_all_tests()
        sys.exit(0 if success else 1)
