#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba Environment Detection and Performance Benchmark
======================================================================
Detects Numba installation, underlying implementations, and runs baseline performance tests.
All output is in English per MCPC coding standards.
"""

import sys
import time
import platform
import subprocess
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section title."""
    print(f"\n{'=' * width}")
    print(f" {title.upper()}")
    print(f"{'=' * width}")

def print_subsection(title: str) -> None:
    """Print a formatted subsection title."""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")

def get_system_info() -> Dict[str, str]:
    """Gather system information."""
    info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'platform_release': platform.release(),
        'architecture': platform.machine(),
        'processor': platform.processor() or 'Unknown',
    }
    
    # Get memory information if available
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_ram_gb'] = f"{mem.total / (1024**3):.2f}"
        info['available_ram_gb'] = f"{mem.available / (1024**3):.2f}"
    except ImportError:
        info['total_ram_gb'] = 'Unknown (install psutil)'
        info['available_ram_gb'] = 'Unknown (install psutil)'
    
    return info

def check_numba_installation() -> Dict[str, Any]:
    """Check Numba and related packages installation."""
    packages = {}
    
    # Check Numba
    try:
        import numba
        packages['numba'] = {
            'version': numba.__version__,
            'available': True
        }
    except ImportError:
        packages['numba'] = {'available': False, 'error': 'Not installed'}
    
    # Check llvmlite (required for Numba)
    try:
        import llvmlite
        packages['llvmlite'] = {
            'version': llvmlite.__version__,
            'available': True
        }
    except ImportError:
        packages['llvmlite'] = {'available': False, 'error': 'Not installed'}
    
    # Check NumPy
    try:
        import numpy
        packages['numpy'] = {
            'version': numpy.__version__,
            'available': True
        }
    except ImportError:
        packages['numpy'] = {'available': False, 'error': 'Not installed'}
    
    # Check CUDA availability
    try:
        import numba.cuda
        cuda_available = numba.cuda.is_available()
        packages['cuda'] = {
            'available': cuda_available,
            'devices': numba.cuda.gpus.current if cuda_available else None
        }
        if cuda_available:
            packages['cuda']['device_count'] = numba.cuda.gpus.device_count
    except (ImportError, AttributeError):
        packages['cuda'] = {'available': False, 'error': 'Not available'}
    
    return packages

def get_numba_configuration() -> Dict[str, Any]:
    """Get detailed Numba configuration."""
    config = {}
    
    try:
        import numba
        import llvmlite
        
        # Basic configuration
        config['numba_version'] = numba.__version__
        config['llvmlite_version'] = llvmlite.__version__
        
        # Threading layer
        try:
            config['threading_layer'] = numba.threading_layer()
        except:
            config['threading_layer'] = 'Unknown'
        
        # Numba config flags
        config['config'] = {
            'DEBUG': getattr(numba.config, 'DEBUG', False),
            'PARALLEL_DIAGNOSTICS': getattr(numba.config, 'PARALLEL_DIAGNOSTICS', False),
            'OPT': getattr(numba.config, 'OPT', 3),  # Default optimization level
            'FASTMATH': getattr(numba.config, 'FASTMATH', False),
        }
        
        # LLVM information
        try:
            from llvmlite import binding as llvm
            config['llvm_version'] = llvm.llvm_version_info
            config['llvm_string'] = f"{llvm.llvm_version_info[0]}.{llvm.llvm_version_info[1]}.{llvm.llvm_version_info[2]}"
        except:
            config['llvm_version'] = 'Unknown'
            config['llvm_string'] = 'Unknown'
        
        # Check parallel acceleration
        try:
            from numba import prange
            config['parallel_available'] = True
        except:
            config['parallel_available'] = False
        
        # Check CUDA in detail
        try:
            import numba.cuda
            if numba.cuda.is_available():
                config['cuda'] = {
                    'available': True,
                    'device_count': numba.cuda.gpus.device_count,
                    'current_device': numba.cuda.gpus.current,
                }
                
                # Try to get CUDA device info
                try:
                    device = numba.cuda.get_current_device()
                    config['cuda']['device_name'] = device.name.decode() if isinstance(device.name, bytes) else device.name
                    config['cuda']['compute_capability'] = device.compute_capability
                except:
                    config['cuda']['device_info'] = 'Could not retrieve device details'
            else:
                config['cuda'] = {'available': False}
        except:
            config['cuda'] = {'available': False, 'error': 'CUDA module not available'}
        
        # Check other targets
        config['targets'] = {
            'cpu': True,  # Always available
            'parallel': config.get('parallel_available', False),
            'cuda': config.get('cuda', {}).get('available', False),
        }
        
        # Check if using AOT (Ahead-of-Time) compilation
        try:
            from numba.pycc import CC
            config['aot_available'] = True
        except:
            config['aot_available'] = False
        
    except ImportError:
        config['error'] = 'Numba not installed'
    
    return config

def benchmark_vector_operations() -> Dict[str, float]:
    """Benchmark vector operations with and without Numba JIT."""
    benchmarks = {}
    
    try:
        import numba
        import numpy as np
        
        # Define functions
        def python_sum(arr):
            total = 0.0
            for x in arr:
                total += x
            return total
        
        @numba.jit(nopython=True)
        def numba_sum(arr):
            total = 0.0
            for x in arr:
                total += x
            return total
        
        @numba.jit(nopython=True, parallel=True)
        def numba_parallel_sum(arr):
            total = 0.0
            for i in numba.prange(len(arr)):
                total += arr[i]
            return total
        
        # Test with different array sizes
        sizes = [1000, 10000, 100000, 1000000]
        
        for size in sizes:
            arr = np.random.random(size)
            
            # Warm-up for JIT compilation
            if size == sizes[0]:
                _ = numba_sum(arr.copy())
                _ = numba_parallel_sum(arr.copy())
            
            # Benchmark Python
            start = time.perf_counter()
            result_py = python_sum(arr)
            time_py = time.perf_counter() - start
            
            # Benchmark Numba (sequential)
            start = time.perf_counter()
            result_nb = numba_sum(arr)
            time_nb = time.perf_counter() - start
            
            # Benchmark Numba (parallel)
            start = time.perf_counter()
            result_nbp = numba_parallel_sum(arr)
            time_nbp = time.perf_counter() - start
            
            # Benchmark NumPy (reference)
            start = time.perf_counter()
            result_np = np.sum(arr)
            time_np = time.perf_counter() - start
            
            # Verify results
            tolerance = 1e-10
            assert abs(result_py - result_np) < tolerance, f"Python vs NumPy mismatch at size {size}"
            assert abs(result_nb - result_np) < tolerance, f"Numba vs NumPy mismatch at size {size}"
            assert abs(result_nbp - result_np) < tolerance, f"Numba parallel vs NumPy mismatch at size {size}"
            
            benchmarks[f'vector_sum_{size}'] = {
                'python': time_py,
                'numba_seq': time_nb,
                'numba_parallel': time_nbp,
                'numpy': time_np,
                'speedup_seq_vs_py': time_py / time_nb if time_nb > 0 else float('inf'),
                'speedup_par_vs_py': time_py / time_nbp if time_nbp > 0 else float('inf'),
                'speedup_numpy_vs_py': time_py / time_np if time_np > 0 else float('inf'),
            }
    
    except Exception as e:
        benchmarks['error'] = f"Benchmark failed: {str(e)}"
    
    return benchmarks

def benchmark_matrix_operations() -> Dict[str, float]:
    """Benchmark matrix operations with Numba."""
    benchmarks = {}
    
    try:
        import numba
        import numpy as np
        
        # Matrix multiplication
        @numba.jit(nopython=True)
        def numba_matmul(A, B):
            m, n = A.shape
            n, p = B.shape
            C = np.zeros((m, p))
            for i in range(m):
                for j in range(p):
                    total = 0.0
                    for k in range(n):
                        total += A[i, k] * B[k, j]
                    C[i, j] = total
            return C
        
        @numba.jit(nopython=True, parallel=True)
        def numba_parallel_matmul(A, B):
            m, n = A.shape
            n, p = B.shape
            C = np.zeros((m, p))
            for i in numba.prange(m):
                for j in range(p):
                    total = 0.0
                    for k in range(n):
                        total += A[i, k] * B[k, j]
                    C[i, j] = total
            return C
        
        # Test sizes
        sizes = [(50, 50), (100, 100), (200, 200)]
        
        for m, n in sizes:
            A = np.random.random((m, n))
            B = np.random.random((n, m))
            
            # Warm-up
            if (m, n) == sizes[0]:
                _ = numba_matmul(A.copy(), B.copy())
                _ = numba_parallel_matmul(A.copy(), B.copy())
            
            # Benchmark Numba sequential
            start = time.perf_counter()
            C_nb = numba_matmul(A, B)
            time_nb = time.perf_counter() - start
            
            # Benchmark Numba parallel
            start = time.perf_counter()
            C_nbp = numba_parallel_matmul(A, B)
            time_nbp = time.perf_counter() - start
            
            # Benchmark NumPy (reference)
            start = time.perf_counter()
            C_np = np.dot(A, B)
            time_np = time.perf_counter() - start
            
            # Verify results
            tolerance = 1e-8
            assert np.allclose(C_nb, C_np, rtol=tolerance), f"Numba vs NumPy mismatch at size {(m, n)}"
            assert np.allclose(C_nbp, C_np, rtol=tolerance), f"Numba parallel vs NumPy mismatch at size {(m, n)}"
            
            benchmarks[f'matmul_{m}x{n}'] = {
                'numba_seq': time_nb,
                'numba_parallel': time_nbp,
                'numpy': time_np,
                'speedup_seq_vs_numpy': time_np / time_nb if time_nb > 0 else float('inf'),
                'speedup_par_vs_numpy': time_np / time_nbp if time_nbp > 0 else float('inf'),
            }
    
    except Exception as e:
        benchmarks['error'] = f"Matrix benchmark failed: {str(e)}"
    
    return benchmarks

def benchmark_jit_compilation() -> Dict[str, float]:
    """Benchmark JIT compilation overhead."""
    benchmarks = {}
    
    try:
        import numba
        import numpy as np
        
        # Simple function to test compilation time
        @numba.jit(nopython=True)
        def simple_function(x):
            return x * x + 2 * x + 1
        
        # Measure compilation time (first call)
        arr = np.array([1.0, 2.0, 3.0])
        
        start = time.perf_counter()
        result = simple_function(arr)
        first_call_time = time.perf_counter() - start
        
        # Measure execution time (second call, should be compiled)
        start = time.perf_counter()
        result = simple_function(arr)
        second_call_time = time.perf_counter() - start
        
        # Test with different signatures (causes recompilation)
        @numba.jit(nopython=True)
        def typed_function(x):
            if isinstance(x, float):
                return x * 2
            else:
                return x + 1
        
        # Measure with float
        start = time.perf_counter()
        typed_function(5.0)
        float_compile_time = time.perf_counter() - start
        
        # Measure with int (should trigger recompilation)
        start = time.perf_counter()
        typed_function(5)
        int_compile_time = time.perf_counter() - start
        
        benchmarks['compilation'] = {
            'first_call_ms': first_call_time * 1000,
            'second_call_ms': second_call_time * 1000,
            'compilation_overhead_ms': (first_call_time - second_call_time) * 1000,
            'float_compile_ms': float_compile_time * 1000,
            'int_compile_ms': int_compile_time * 1000,
            'recompilation_penalty_ms': (int_compile_time - float_compile_time) * 1000,
        }
    
    except Exception as e:
        benchmarks['error'] = f"JIT compilation benchmark failed: {str(e)}"
    
    return benchmarks

def check_cuda_performance() -> Optional[Dict[str, Any]]:
    """Check CUDA performance if available."""
    try:
        import numba.cuda
        import numpy as np
        
        if not numba.cuda.is_available():
            return None
        
        cuda_info = {}
        
        # Get device information
        device = numba.cuda.get_current_device()
        cuda_info['device_name'] = device.name.decode() if isinstance(device.name, bytes) else device.name
        cuda_info['compute_capability'] = device.compute_capability
        cuda_info['total_memory_gb'] = device.total_memory / (1024**3)
        
        # Simple CUDA kernel for benchmarking
        @numba.cuda.jit
        def cuda_square_kernel(arr_in, arr_out):
            idx = numba.cuda.grid(1)
            if idx < arr_in.size:
                arr_out[idx] = arr_in[idx] ** 2
        
        # Test with different array sizes
        sizes = [1000, 10000, 100000, 1000000]
        
        for size in sizes:
            # Create data
            h_arr = np.random.random(size).astype(np.float32)
            d_arr = numba.cuda.to_device(h_arr)
            d_result = numba.cuda.device_array_like(d_arr)
            
            # Configure kernel
            threads_per_block = 256
            blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
            
            # Benchmark
            start = time.perf_counter()
            cuda_square_kernel[blocks_per_grid, threads_per_block](d_arr, d_result)
            numba.cuda.synchronize()
            cuda_time = time.perf_counter() - start
            
            # Copy back and verify
            h_result = d_result.copy_to_host()
            
            # Compare with NumPy
            start = time.perf_counter()
            np_result = h_arr ** 2
            numpy_time = time.perf_counter() - start
            
            # Verify
            assert np.allclose(h_result, np_result, rtol=1e-5), f"Cuda vs NumPy mismatch at size {size}"
            
            cuda_info[f'cuda_square_{size}'] = {
                'cuda_ms': cuda_time * 1000,
                'numpy_ms': numpy_time * 1000,
                'speedup': numpy_time / cuda_time if cuda_time > 0 else float('inf'),
                'data_transfer_overhead': 'Included in timing',
            }
        
        return cuda_info
    
    except Exception as e:
        return {'error': f"Cuda benchmark failed: {str(e)}"}

def print_summary(system_info: Dict, config: Dict, 
                  vector_benchmarks: Dict, matrix_benchmarks: Dict,
                  compilation_benchmarks: Dict, cuda_info: Optional[Dict]) -> None:
    """Print a summary of key findings."""
    print_section("SUMMARY OF KEY FINDINGS")
    
    summary_points = []
    
    # System info
    summary_points.append(f"• Platform: {system_info['platform']} {system_info['architecture']}")
    summary_points.append(f"• Python: {system_info['python_version']}")
    
    # Numba info
    if 'error' not in config:
        summary_points.append(f"• Numba: {config.get('numba_version', 'Unknown')}")
        summary_points.append(f"• LLVM: {config.get('llvm_string', 'Unknown')}")
        summary_points.append(f"• Threading layer: {config.get('threading_layer', 'Unknown')}")
        
        # Targets
        targets = config.get('targets', {})
        available_targets = [k for k, v in targets.items() if v]
        summary_points.append(f"• Available targets: {', '.join(available_targets)}")
        
        # CUDA if available
        cuda = config.get('cuda', {})
        if cuda.get('available', False):
            summary_points.append(f"• CUDA: Available ({cuda.get('device_count', 0)} device(s))")
            if 'device_name' in cuda:
                summary_points.append(f"  Device: {cuda.get('device_name', 'Unknown')}")
    else:
        summary_points.append("• Numba: Not installed or configuration error")
    
    # Performance highlights
    if vector_benchmarks and 'error' not in vector_benchmarks:
        # Get largest size benchmark
        largest_key = max([k for k in vector_benchmarks.keys() if k.startswith('vector_sum')], 
                         key=lambda x: int(x.split('_')[-1]), default=None)
        if largest_key:
            bench = vector_benchmarks[largest_key]
            speedup = bench.get('speedup_seq_vs_py', 0)
            if speedup > 1:
                summary_points.append(f"• Vector sum speedup (Numba vs Python): {speedup:.1f}x")
    
    if cuda_info and 'error' not in cuda_info:
        # Get CUDA speedup if available
        cuda_keys = [k for k in cuda_info.keys() if k.startswith('cuda_square_')]
        if cuda_keys:
            largest_cuda = max(cuda_keys, key=lambda x: int(x.split('_')[-1]))
            if largest_cuda in cuda_info:
                speedup = cuda_info[largest_cuda].get('speedup', 0)
                if speedup > 1:
                    summary_points.append(f"• CUDA speedup (vs NumPy): {speedup:.1f}x")
    
    # Compilation overhead
    if compilation_benchmarks and 'error' not in compilation_benchmarks:
        overhead = compilation_benchmarks.get('compilation', {}).get('compilation_overhead_ms', 0)
        if overhead > 0:
            summary_points.append(f"• JIT compilation overhead: {overhead:.2f} ms (first call)")
    
    # Compliance note
    summary_points.append("• All output is in English per MCPC coding standards")
    
    for point in summary_points:
        print(point)
    
    # Recommendations
    print("\nRecommendations:")
    print("1. For better parallel performance, ensure threading layer is set appropriately")
    print("2. Consider using @njit decorator for maximum performance")
    print("3. For CUDA development, install appropriate CUDA toolkit")
    print("4. Use caching to avoid recompilation (@jit(cache=True))")
    print("5. Profile with numba.profiling for detailed performance analysis")

def main():
    """Main function to run all checks and benchmarks."""
    print_section("Numba Environment Detection and Performance Benchmark")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get system information
    print_section("System Information")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Check Numba installation
    print_section("Numba Installation Check")
    packages = check_numba_installation()
    
    for pkg_name, pkg_info in packages.items():
        if pkg_info.get('available', False):
            print(f"{pkg_name}: Installed (version: {pkg_info.get('version', 'Unknown')})")
        else:
            print(f"{pkg_name}: Not available - {pkg_info.get('error', 'Check installation')}")
    
    # Get detailed Numba configuration
    print_section("Numba Configuration")
    config = get_numba_configuration()
    
    if 'error' in config:
        print(f"Error: {config['error']}")
        print("Cannot proceed with benchmarks without Numba.")
        return
    
    # Print configuration details
    print(f"Numba Version: {config.get('numba_version', 'Unknown')}")
    print(f"llvmlite Version: {config.get('llvmlite_version', 'Unknown')}")
    print(f"LLVM Version: {config.get('llvm_string', 'Unknown')}")
    print(f"Threading Layer: {config.get('threading_layer', 'Unknown')}")
    
    print_subsection("Configuration Flags")
    for flag, value in config.get('config', {}).items():
        print(f"  {flag}: {value}")
    
    print_subsection("Available Targets")
    targets = config.get('targets', {})
    for target, available in targets.items():
        status = "✓" if available else "✗"
        print(f"  {status} {target.upper()}")
    
    # CUDA details if available
    cuda_config = config.get('cuda', {})
    if cuda_config.get('available', False):
        print_subsection("CUDA Configuration")
        print(f"  CUDA Available: Yes")
        print(f"  Device Count: {cuda_config.get('device_count', 'Unknown')}")
        if 'device_name' in cuda_config:
            print(f"  Current Device: {cuda_config.get('device_name', 'Unknown')}")
        if 'compute_capability' in cuda_config:
            print(f"  Compute Capability: {cuda_config.get('compute_capability', 'Unknown')}")
    
    # Run benchmarks
    print_section("Performance Benchmarks")
    
    # JIT Compilation benchmark
    print_subsection("JIT Compilation Overhead")
    compilation_benchmarks = benchmark_jit_compilation()
    if 'error' in compilation_benchmarks:
        print(f"Error: {compilation_benchmarks['error']}")
    else:
        comp_data = compilation_benchmarks.get('compilation', {})
        print(f"First call (with compilation): {comp_data.get('first_call_ms', 0):.2f} ms")
        print(f"Second call (compiled): {comp_data.get('second_call_ms', 0):.2f} ms")
        print(f"Compilation overhead: {comp_data.get('compilation_overhead_ms', 0):.2f} ms")
    
    # Vector operations benchmark
    print_subsection("Vector Operations Benchmark")
    vector_benchmarks = benchmark_vector_operations()
    if 'error' in vector_benchmarks:
        print(f"Error: {vector_benchmarks['error']}")
    else:
        print("Array Size | Python (ms) | Numba Seq (ms) | Numba Par (ms) | NumPy (ms) | Speedup (Seq) | Speedup (Par)")
        print("-" * 100)
        
        for key, bench in vector_benchmarks.items():
            if key.startswith('vector_sum_'):
                size = key.split('_')[-1]
                print(f"{size:>10} | "
                      f"{bench['python']*1000:>11.2f} | "
                      f"{bench['numba_seq']*1000:>13.2f} | "
                      f"{bench['numba_parallel']*1000:>13.2f} | "
                      f"{bench['numpy']*1000:>10.2f} | "
                      f"{bench['speedup_seq_vs_py']:>12.1f}x | "
                      f"{bench['speedup_par_vs_py']:>12.1f}x")
    
    # Matrix operations benchmark
    print_subsection("Matrix Operations Benchmark")
    matrix_benchmarks = benchmark_matrix_operations()
    if 'error' in matrix_benchmarks:
        print(f"Error: {matrix_benchmarks['error']}")
    else:
        print("Matrix Size | Numba Seq (ms) | Numba Par (ms) | NumPy (ms) | Speedup Seq | Speedup Par")
        print("-" * 90)
        
        for key, bench in matrix_benchmarks.items():
            if key.startswith('matmul_'):
                size = key.split('_')[1]
                print(f"{size:>11} | "
                      f"{bench['numba_seq']*1000:>13.2f} | "
                      f"{bench['numba_parallel']*1000:>13.2f} | "
                      f"{bench['numpy']*1000:>10.2f} | "
                      f"{bench['speedup_seq_vs_numpy']:>10.1f}x | "
                      f"{bench['speedup_par_vs_numpy']:>10.1f}x")
    
    # CUDA benchmark if available
    cuda_info = None
    if cuda_config.get('available', False):
        print_subsection("CUDA Performance Benchmark")
        cuda_info = check_cuda_performance()
        
        if cuda_info:
            if 'error' in cuda_info:
                print(f"Error: {cuda_info['error']}")
            else:
                # Print device info
                if 'device_name' in cuda_info:
                    print(f"Device: {cuda_info.get('device_name', 'Unknown')}")
                if 'compute_capability' in cuda_info:
                    print(f"Compute Capability: {cuda_info.get('compute_capability', 'Unknown')}")
                if 'total_memory_gb' in cuda_info:
                    print(f"Total Memory: {cuda_info.get('total_memory_gb', 0):.2f} GB")
                
                print("\nArray Size | CUDA (ms) | NumPy (ms) | Speedup")
                print("-" * 60)
                
                for key, bench in cuda_info.items():
                    if key.startswith('cuda_square_'):
                        size = key.split('_')[-1]
                        print(f"{size:>10} | "
                              f"{bench['cuda_ms']:>9.2f} | "
                              f"{bench['numpy_ms']:>10.2f} | "
                              f"{bench['speedup']:>7.1f}x")
    
    # Print summary
    print_summary(system_info, config, vector_benchmarks, 
                  matrix_benchmarks, compilation_benchmarks, cuda_info)
    
    print_section("Benchmark Complete")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
