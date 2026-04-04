"""
运行 LUNAR-SWING 桩测试

此脚本运行第一阶段的所有桩测试，验证接口设计合理性。
"""
import sys
import os
import pytest

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

def run_stub_tests():
    """运行所有桩测试"""
    print("=" * 70)
    print("运行 LUNAR-SWING 桩测试（第一阶段：接口设计验证）")
    print("=" * 70)
    
    # 测试文件列表
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
    sys.exit(run_stub_tests())
