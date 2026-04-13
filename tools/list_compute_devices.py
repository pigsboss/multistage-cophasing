#!/usr/bin/env python3
"""
命令行工具：列出当前环境中可用的计算设备。

调用 mission_sim.utils.solvers._devices 模块来检测和显示 CPU、GPU 等设备。
遵循 MCPC 编码标准：UTF-8 编码，运行时输出使用英文。
"""

import sys
import os
import argparse
import json
from typing import List, Dict, Any

# 添加项目根目录到 Python 路径，以便导入 mission_sim 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)


def main():
    """主函数：解析命令行参数并列出计算设备"""
    parser = argparse.ArgumentParser(
        description="列出当前环境中可用的计算设备",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                    # 列出所有设备（默认详细模式）
  %(prog)s --brief            # 简要列出设备
  %(prog)s --json             # 以 JSON 格式输出设备信息
  %(prog)s --type cpu         # 只列出 CPU 设备
  %(prog)s --type gpu         # 只列出 GPU 设备
  %(prog)s --quiet            # 静默模式，仅输出错误信息
  %(prog)s --check-libraries  # 检查并显示必要的计算库
        """
    )
    
    parser.add_argument(
        '--brief', '-b',
        action='store_true',
        help='简要输出模式，只显示设备名称和类型'
    )
    
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='以 JSON 格式输出设备信息'
    )
    
    parser.add_argument(
        '--type', '-t',
        choices=['cpu', 'gpu', 'all'],
        default='all',
        help='过滤设备类型：cpu, gpu, 或 all (默认)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，不输出常规信息'
    )
    
    parser.add_argument(
        '--check-libraries', '-c',
        action='store_true',
        help='检查并显示计算库的可用性'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='调试模式，显示详细的错误信息'
    )
    
    parser.add_argument(
        '--version', '-V',
        action='version',
        version='MCPC Device Scanner 1.0.0'
    )
    
    args = parser.parse_args()
    
    try:
        # 导入设备检测模块
        from mission_sim.utils.solvers._devices import (
            detect_available_devices,
            get_device_by_type,
            DeviceType,
            CPUDevice,
            GPUDevice
        )
        
        if not args.quiet:
            print("=" * 60)
            print("MCPC COMPUTE DEVICE SCANNER")
            print("=" * 60)
            print(f"Filter: {args.type.upper()} devices")
            print("=" * 60)
        
        # 检测所有可用设备
        all_devices = detect_available_devices()
        
        # 根据类型过滤设备
        if args.type == 'cpu':
            filtered_devices = [d for d in all_devices if d.device_type == DeviceType.CPU]
        elif args.type == 'gpu':
            filtered_devices = [d for d in all_devices if d.device_type == DeviceType.GPU]
        else:
            filtered_devices = all_devices
        
        if not args.quiet:
            print(f"\nFound {len(filtered_devices)} device(s):")
            print("-" * 60)
        
        if not filtered_devices:
            print("[WARNING] No compute devices found!")
            if args.debug:
                print(f"[DEBUG] All devices detected: {len(all_devices)}")
            sys.exit(1)
        
        # 准备设备信息
        device_info_list = []
        
        for i, device in enumerate(filtered_devices):
            metrics = device.get_performance_metrics()
            
            device_info = {
                "index": i,
                "type": device.device_type.value.upper(),
                "name": device.device_name,
                "available": device.is_available(),
                "initialized": getattr(device, '_initialized', False),
                "hardware": metrics.get("hardware_spec", {}),
                "libraries": metrics.get("libraries", {})
            }
            
            device_info_list.append(device_info)
            
            # 根据输出格式显示信息
            if args.json:
                continue  # 稍后统一输出JSON
            
            elif args.brief:
                status = "✓" if device.is_available() else "✗"
                print(f"{status} {device.device_type.value.upper():4} {device.device_name}")
            
            elif not args.quiet:
                # 详细输出模式
                print(f"\nDevice {i+1}: {device.device_name}")
                print(f"  Type: {device.device_type.value.upper()}")
                print(f"  Status: {'Available' if device.is_available() else 'Not Available'}")
                print(f"  Initialized: {'Yes' if getattr(device, '_initialized', False) else 'No'}")
                
                # 硬件规格
                if metrics.get("hardware_spec"):
                    spec = metrics["hardware_spec"]
                    if spec.get('cores'):
                        print(f"  Cores: {spec['cores']}")
                    if spec.get('memory_gb'):
                        print(f"  Memory: {spec['memory_gb']:.1f} GB")
                    if spec.get('clock_speed_ghz'):
                        print(f"  Clock: {spec['clock_speed_ghz']:.2f} GHz")
                    if spec.get('features'):
                        print(f"  Features: {', '.join(spec['features'])}")
                
                # 库信息（如果指定要检查）
                if args.check_libraries and metrics.get("libraries"):
                    print(f"  Libraries:")
                    for lib_name, lib_details in metrics["libraries"].items():
                        status = "✓" if lib_details.get("available", False) else "✗"
                        version = lib_details.get("version", "unknown")
                        backend = lib_details.get("backend", "")
                        backend_str = f" ({backend})" if backend else ""
                        print(f"    {status} {lib_name}: {version}{backend_str}")
        
        # JSON输出
        if args.json:
            output_data = {
                "device_count": len(device_info_list),
                "filter_type": args.type,
                "devices": device_info_list
            }
            print(json.dumps(output_data, indent=2, default=str))
        
        # 汇总信息
        if not args.quiet and not args.json:
            cpu_count = sum(1 for d in filtered_devices if d.device_type == DeviceType.CPU)
            gpu_count = sum(1 for d in filtered_devices if d.device_type == DeviceType.GPU)
            available_count = sum(1 for d in filtered_devices if d.is_available())
            
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Total devices: {len(filtered_devices)}")
            print(f"  CPUs: {cpu_count}")
            print(f"  GPUs: {gpu_count}")
            print(f"Available: {available_count}")
            
            # 建议
            if available_count == 0:
                print("\n[WARNING] No available compute devices found!")
                print("Suggestions:")
                print("  1. Check if necessary drivers are installed (for GPUs)")
                print("  2. Verify that CUDA/OpenCL libraries are installed")
                print("  3. Ensure Python packages are installed: numpy, numba, etc.")
            elif gpu_count == 0 and args.type == 'all':
                print("\n[INFO] No GPU devices found. Computation will use CPU only.")
            elif cpu_count == 0 and args.type == 'all':
                print("\n[INFO] No CPU devices detected. This is unusual.")
            
            print("=" * 60)
        
        # 检查库信息
        if args.check_libraries and not args.quiet and not args.json:
            print("\n" + "=" * 60)
            print("LIBRARY CHECK")
            print("=" * 60)
            
            # 检查关键库
            critical_libraries = ['numpy', 'scipy', 'numba']
            try:
                import importlib
                for lib in critical_libraries:
                    try:
                        module = importlib.import_module(lib)
                        version = getattr(module, '__version__', 'unknown')
                        print(f"✓ {lib}: {version}")
                    except ImportError:
                        print(f"✗ {lib}: NOT INSTALLED")
            except Exception as e:
                print(f"[ERROR] Failed to check libraries: {e}")
            
            print("=" * 60)
        
    except ImportError as e:
        print(f"[ERROR] Failed to import device detection module: {e}", file=sys.stderr)
        if args.debug:
            print(f"[DEBUG] Python path: {sys.path}", file=sys.stderr)
            print(f"[DEBUG] Current directory: {os.getcwd()}", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
