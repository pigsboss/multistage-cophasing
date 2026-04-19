#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX Compilation Stack and Hardware Backend Detection
======================================================================
Detects JAX compilation system, XLA runtime, PJRT plugins for various
hardware accelerators. Displays available compute devices for JIT acceleration.
All output is in English per MCPC coding standards.
"""

import sys
import platform
import subprocess
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Environment variables for JAX/XLA logging control
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('JAX_LOG_LEVEL', 'WARNING')


@contextmanager
def suppress_stderr():
    """Suppress stderr output (for C++ library logs)."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)


@dataclass
class BackendInfo:
    """Information about a JAX backend."""
    name: str
    platform: str
    available: bool
    device_kind: str = "Unknown"
    version: str = "Unknown"
    device_count: int = 0
    devices: List[Dict[str, Any]] = field(default_factory=list)
    pjrt_source: str = ""  # Package providing the plugin
    priority: int = 0  # Lower = higher priority
    error: Optional[str] = None


def get_system_info() -> Dict[str, str]:
    """Gather system information."""
    info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'architecture': platform.machine(),
        'cpu_cores': 'Unknown',
        'total_ram_gb': 'Unknown',
    }
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_ram_gb'] = f"{mem.total / (1024**3):.2f}"
        info['cpu_cores'] = str(psutil.cpu_count(logical=False) or 'Unknown')
    except:
        pass
    
    if platform.system() == 'Darwin':
        info['macos_version'] = platform.mac_ver()[0]
        info['apple_silicon'] = 'Yes' if platform.machine() == 'arm64' else 'No'
    
    return info


def detect_jax_stack() -> Dict[str, Any]:
    """Detect JAX compilation stack: JAX → XLA → PJRT → Hardware."""
    stack = {
        'jax': {'available': False, 'version': None},
        'xla': {'backend': None, 'platform': None, 'x64_enabled': False},
        'pjrt_plugins': {},
        'devices': [],
    }
    
    try:
        with suppress_stderr():
            import jax
            import jax.numpy as jnp
            
            # 1. JAX Frontend Layer
            stack['jax'] = {
                'available': True,
                'version': jax.__version__,
                'python_version': platform.python_version(),
            }
            
            # 2. XLA Runtime Layer
            backend = jax.lib.xla_bridge.get_backend()
            stack['xla'] = {
                'backend': backend.platform,
                'platform_version': getattr(backend, 'platform_version', 'Unknown'),
                'device_count': backend.device_count(),
                'x64_enabled': jax.config.x64_enabled,
                'default_dtype': 'float64' if jax.config.x64_enabled else 'float32',
            }
            
            # 3. PJRT Plugin Layer & Hardware Devices
            all_devices = jax.devices()
            stack['devices'] = []
            
            for device in all_devices:
                dev_info = {
                    'id': device.id,
                    'platform': device.platform,
                    'kind': device.device_kind,
                    'local_hardware_id': getattr(device, 'local_hardware_id', None),
                }
                stack['devices'].append(dev_info)
            
            # Identify backend plugins by platform
            platforms = {d.platform.lower(): d for d in all_devices}
            
            # Metal (Apple)
            if 'metal' in platforms:
                dev = platforms['metal']
                stack['pjrt_plugins']['metal'] = BackendInfo(
                    name='Apple Metal GPU',
                    platform='metal',
                    available=True,
                    device_kind=dev.device_kind,
                    pjrt_source='jax-metal',
                    priority=10
                )
            
            # CUDA (NVIDIA)
            if 'cuda' in platforms:
                dev = platforms['cuda']
                stack['pjrt_plugins']['cuda'] = BackendInfo(
                    name='NVIDIA GPU (CUDA)',
                    platform='cuda',
                    available=True,
                    device_kind=dev.device_kind,
                    pjrt_source='jax[cuda]',
                    priority=10
                )
            
            # ROCm (AMD)
            if 'rocm' in platforms:
                dev = platforms['rocm']
                stack['pjrt_plugins']['rocm'] = BackendInfo(
                    name='AMD GPU (ROCm)',
                    platform='rocm',
                    available=True,
                    device_kind=dev.device_kind,
                    pjrt_source='jax-rocm',
                    priority=10
                )
            
            # CPU (Always available as fallback)
            if 'cpu' in platforms:
                dev = platforms['cpu']
                stack['pjrt_plugins']['cpu'] = BackendInfo(
                    name='CPU (Host)',
                    platform='cpu',
                    available=True,
                    device_kind=dev.device_kind,
                    pjrt_source='jaxlib (built-in)',
                    priority=100
                )
            
            # TPU
            if 'tpu' in platforms:
                dev = platforms['tpu']
                stack['pjrt_plugins']['tpu'] = BackendInfo(
                    name='Google Cloud TPU',
                    platform='tpu',
                    available=True,
                    device_kind=dev.device_kind,
                    pjrt_source='jax[tpu]',
                    priority=5
                )
            
            # Intel GPU
            intel_devices = [d for d in all_devices 
                           if any(x in d.device_kind.lower() for x in ['intel', 'arc', 'xe'])]
            if intel_devices:
                stack['pjrt_plugins']['intel'] = BackendInfo(
                    name='Intel GPU',
                    platform=intel_devices[0].platform,
                    available=True,
                    device_kind=intel_devices[0].device_kind,
                    pjrt_source='intel-extension-for-jax',
                    priority=20
                )
            
    except ImportError:
        stack['error'] = 'JAX not installed'
    
    return stack


def print_hierarchy(stack: Dict[str, Any], verbose: bool = False) -> None:
    """Print JAX compilation stack in hierarchical format."""
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│              JAX COMPILATION STACK                              │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Layer 1: JAX Frontend
    jax_info = stack.get('jax', {})
    if jax_info.get('available'):
        print("\n▶ Layer 1: JAX Frontend (Python API)")
        print(f"  Version: {jax_info.get('version')}")
        if verbose:
            print(f"  Python: {jax_info.get('python_version')}")
    else:
        print("\n▶ Layer 1: JAX Frontend [NOT INSTALLED]")
        print("  Install: pip install jax jaxlib")
        return
    
    # Layer 2: XLA Runtime
    xla_info = stack.get('xla', {})
    print("\n▶ Layer 2: XLA Runtime (Accelerated Linear Algebra)")
    if xla_info.get('backend'):
        print(f"  Active Backend: {xla_info['backend'].upper()}")
        print(f"  Platform Version: {xla_info.get('platform_version', 'Unknown')}")
        print(f"  Default Precision: {xla_info.get('default_dtype', 'float32')}")
        if verbose:
            print(f"  X64 (Float64) Support: {'Enabled' if xla_info.get('x64_enabled') else 'Disabled'}")
            print(f"  Device Count: {xla_info.get('device_count', 0)}")
    
    # Layer 3: PJRT Plugins
    plugins = stack.get('pjrt_plugins', {})
    active_plugins = [p for p in plugins.values() if p.available]
    
    print("\n▶ Layer 3: PJRT Plugins (Hardware Interface)")
    if active_plugins:
        sorted_plugins = sorted(active_plugins, key=lambda x: x.priority)
        for plugin in sorted_plugins:
            status = "✓ ACTIVE" if plugin.priority <= 30 else "○ Available"
            print(f"  [{status}] {plugin.name}")
            print(f"           Source: {plugin.pjrt_source}")
            print(f"           Platform: {plugin.platform}")
            if plugin.device_kind and plugin.device_kind != 'Unknown':
                print(f"           Hardware: {plugin.device_kind}")
    else:
        print("  No active hardware plugins detected")
    
    # Layer 4: Compute Devices (JIT-ready)
    devices = stack.get('devices', [])
    if devices:
        print("\n▶ Layer 4: Compute Devices (JIT-ready)")
        for i, dev in enumerate(devices[:4]):
            platform = dev['platform']
            kind = dev['kind']
            
            accel_marker = "[CPU]" if platform.lower() == 'cpu' else "[GPU]"
            print(f"  {accel_marker} {kind} (platform={platform}, id={dev['id']})")
            if verbose and dev.get('local_hardware_id') is not None:
                print(f"           Hardware ID: {dev['local_hardware_id']}")
        
        if len(devices) > 4:
            print(f"  ... and {len(devices) - 4} more devices")


def print_quick_summary(stack: Dict[str, Any]) -> None:
    """Print quick summary of available compute capabilities."""
    plugins = stack.get('pjrt_plugins', {})
    xla_info = stack.get('xla', {})
    
    accelerators = [p for p in plugins.values() 
                   if p.available and p.platform.lower() != 'cpu']
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if accelerators:
        best = min(accelerators, key=lambda x: x.priority)
        print(f"✓ Primary Accelerator: {best.name} ({best.device_kind})")
        print(f"✓ XLA Backend: {xla_info.get('backend', 'Unknown')}")
        print(f"✓ JIT Compilation: Ready")
        
        other = [p for p in accelerators if p.platform != best.platform]
        if other:
            print(f"✓ Additional Backends: {', '.join(p.name for p in other)}")
    else:
        cpu_plugin = plugins.get('cpu')
        if cpu_plugin and cpu_plugin.available:
            print(f"✓ Available: CPU only ({cpu_plugin.device_kind})")
            print("⚠ No GPU/Accelerator detected")
        else:
            print("✗ No compute backends available")
    
    print("\nQuick Start:")
    print("  >>> import jax")
    print("  >>> jax.devices()  # List available devices")
    if accelerators:
        best = min(accelerators, key=lambda x: x.priority)
        if best.platform.lower() == 'metal':
            print("  # JAX-Metal active: Use float32 for best performance")
        elif best.platform.lower() == 'cuda':
            print("  # JAX-CUDA active: GPU acceleration ready")
    print("="*70)


def print_verbose_details(stack: Dict[str, Any], system_info: Dict) -> None:
    """Print verbose system and configuration details."""
    print("\n" + "="*70)
    print("SYSTEM ENVIRONMENT (Verbose)")
    print("="*70)
    
    print(f"Operating System: {system_info.get('platform', 'Unknown')}")
    if 'macos_version' in system_info:
        print(f"macOS Version: {system_info['macos_version']}")
        print(f"Apple Silicon: {system_info.get('apple_silicon', 'Unknown')}")
    print(f"CPU Cores: {system_info.get('cpu_cores', 'Unknown')}")
    print(f"Total RAM: {system_info.get('total_ram_gb', 'Unknown')} GB")
    
    xla_info = stack.get('xla', {})
    print("\nXLA Configuration:")
    print(f"  XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")
    print(f"  TF_CPP_MIN_LOG_LEVEL: {os.environ.get('TF_CPP_MIN_LOG_LEVEL', 'Not set')}")
    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '')
    print(f"  Compilation Cache: {'Enabled' if cache_dir else 'Disabled'}")
    
    print("\n" + "="*70)
    print("INSTALLATION GUIDE")
    print("="*70)
    
    plugins = stack.get('pjrt_plugins', {})
    active_platforms = {p.platform.lower() for p in plugins.values() if p.available}
    
    if 'metal' not in active_platforms and system_info.get('apple_silicon') == 'Yes':
        print("• Apple Metal: pip install jax-metal")
    if 'cuda' not in active_platforms:
        print("• NVIDIA CUDA: pip install 'jax[cuda12]' (Linux/Windows)")
    if 'rocm' not in active_platforms:
        print("• AMD ROCm: pip install jax-rocm (Linux only)")
    if 'tpu' not in active_platforms:
        print("• Google TPU: pip install jax[tpu] (GCP only)")
    print("• Intel GPU: https://github.com/intel/intel-extension-for-jax")
    
    print("\nArchitecture Notes:")
    print("  JAX (Python) → XLA (Graph Compiler) → PJRT (Plugin API) → Driver → Hardware")
    print("  PJRT = Portable Runtime for JAX hardware plugins")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="JAX Compilation Stack and Hardware Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              # Default: hierarchical stack view
  %(prog)s --verbose    # Full system details and installation guide
  %(prog)s --json       # Machine-readable JSON output
        """
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed system environment and installation guide"
    )
    
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON format for programmatic use"
    )
    
    parser.add_argument(
        "--output", type=str,
        help="Output file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    with suppress_stderr():
        stack = detect_jax_stack()
        system_info = get_system_info() if args.verbose else {}
    
    if args.json:
        output = {
            'jax_version': stack.get('jax', {}).get('version'),
            'xla_backend': stack.get('xla', {}).get('backend'),
            'devices': stack.get('devices', []),
            'plugins': {
                k: {
                    'name': v.name,
                    'platform': v.platform,
                    'available': v.available,
                    'source': v.pjrt_source
                } for k, v in stack.get('pjrt_plugins', {}).items()
            }
        }
        
        json_str = json.dumps(output, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_str)
            print(f"Results saved to: {args.output}")
        else:
            print(json_str)
        return
    
    print_hierarchy(stack, verbose=args.verbose)
    print_quick_summary(stack)
    
    if args.verbose:
        print_verbose_details(stack, system_info)
    
    print("\nAll output is in English per MCPC coding standards.")


if __name__ == "__main__":
    main()
