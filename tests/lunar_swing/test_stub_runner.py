"""
运行 LUNAR-SWING 桩测试的便捷入口

此模块提供运行所有桩测试的功能，验证接口设计合理性。
符合 pytest 命名规范 (test_*.py)，可被 pytest 自动发现。
"""
import sys
import os
import pytest

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

def test_stub_runner():
    """
    运行所有 lunar-swing 桩测试的入口测试函数。
    
    注意：这个测试函数本身不包含断言，它依赖于 pytest 自动发现
    并运行所有 test_*.py 文件。添加这个函数是为了：
    1. 符合 test_*.py 命名规范
    2. 提供文档说明
    3. 保持命令行接口功能
    """
    # 这个测试函数总是通过，实际测试由 pytest 自动发现机制执行
    assert True, "桩测试运行器占位符"

def run_stub_tests_cli():
    """
    命令行接口：运行所有桩测试
    
    保留原有的命令行功能，便于手动执行
    """
    print("=" * 70)
    print("运行 LUNAR-SWING 桩测试（第一阶段：接口设计验证）")
    print("=" * 70)
    
    # 测试文件列表（与之前相同）
    test_files = [
        'test_ephemeris_stub.py',
        'test_crtbp_stub.py', 
        'test_geopotential_stub.py',
        'test_targeter_stub.py',
        'test_stm_calculator_stub.py'
    ]
    
    # 构建完整路径
    test_paths = [os.path.join(os.path.dirname(__file__), f) for f in test_files]
    
    # 运行测试
    args = [
        '-v',
        '--tb=short',  # 简短回溯
        '--disable-warnings',
        '--capture=no'  # 显示打印输出
    ]
    
    # 添加测试文件
    args.extend(test_paths)
    
    print(f"\n运行 {len(test_files)} 个桩测试文件...")
    result = pytest.main(args)
    
    if result == 0:
        print("\n✅ 所有桩测试通过！接口设计合理。")
    else:
        print("\n❌ 部分桩测试失败。请检查接口设计。")
    
    return result

if __name__ == '__main__':
    # 命令行入口
    sys.exit(run_stub_tests_cli())
