"""
SPICE Availability Tests

Minimal prerequisite checks for MCPC SPICE integration.
Validates spiceypy installation and required kernel files.
"""

import warnings
# Filter out requests library warnings about urllib3/chardet compatibility
warnings.filterwarnings("ignore", category=DeprecationWarning, module="requests")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

import pytest
import sys
from pathlib import Path
import os


def check_spiceypy_installed():
    """Check if spiceypy is installed."""
    try:
        import spiceypy
        return True
    except ImportError:
        return False


def find_spice_kernels():
    """Locate SPICE kernels directory."""
    env_path = os.environ.get('SPICE_KERNELS')
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    
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
    """Verify required kernel files exist."""
    required = {
        'lsk': ['naif*.tls', 'latest_leapseconds.tls'],
        'spk': ['de440.bsp', 'de441.bsp', 'de442.bsp'],
    }
    
    results = {}
    
    for ktype, patterns in required.items():
        found = False
        for pattern in patterns:
            matches = list(kernel_path.rglob(pattern))
            if matches:
                found = True
                results[ktype] = matches[0].name
                break
        
        if not found:
            results[ktype] = None
    
    return results


class TestSPICEPrerequisites:
    """SPICE prerequisite validation tests."""
    
    def test_spiceypy_installed(self):
        """Verify spiceypy package is installed."""
        if not check_spiceypy_installed():
            pytest.skip("spiceypy not installed. Install with: pip install spiceypy")
        
        # Verify import works
        import spiceypy as spice
        assert spice is not None
    
    def test_spice_kernels_directory_exists(self):
        """Verify SPICE kernels directory is present."""
        kernel_path = find_spice_kernels()
        
        if kernel_path is None:
            pytest.skip(
                "SPICE kernels directory not found. "
                "Set SPICE_KERNELS environment variable or place in ./spice_kernels"
            )
        
        assert kernel_path.exists(), f"Kernel path does not exist: {kernel_path}"
    
    def test_required_kernels_present(self):
        """Verify essential kernel files are available."""
        kernel_path = find_spice_kernels()
        
        if kernel_path is None:
            pytest.skip("SPICE kernels directory not found")
        
        results = check_required_kernels(kernel_path)
        
        # Check LSK (leapseconds kernel) - Required
        if results.get('lsk') is None:
            pytest.fail(
                f"Leapseconds kernel (naif0012.tls or similar) not found in {kernel_path}"
            )
        
        # Check SPK (planetary ephemeris) - Required
        if results.get('spk') is None:
            pytest.fail(
                f"Planetary ephemeris (de440.bsp or similar) not found in {kernel_path}"
            )
        
        # Log found kernels for debugging
        print(f"Found LSK: {results['lsk']}")
        print(f"Found SPK: {results['spk']}")


# Command-line execution support
if __name__ == "__main__":
    print("SPICE Availability Check")
    print("=" * 50)
    
    # Check 1: spiceypy
    print("\n1. Checking spiceypy installation...")
    if check_spiceypy_installed():
        print("   [OK] spiceypy is installed")
    else:
        print("   [FAIL] spiceypy not installed")
        print("   Run: pip install spiceypy")
        sys.exit(1)
    
    # Check 2: Kernel directory
    print("\n2. Checking SPICE kernels directory...")
    kernel_path = find_spice_kernels()
    
    if kernel_path is None:
        print("   [FAIL] Kernel directory not found")
        print("   Set SPICE_KERNELS env var or place in ./spice_kernels")
        sys.exit(1)
    else:
        print(f"   [OK] Found: {kernel_path}")
    
    # Check 3: Required files
    print("\n3. Checking required kernel files...")
    results = check_required_kernels(kernel_path)
    
    all_ok = True
    
    if results.get('lsk'):
        print(f"   [OK] Leapseconds kernel: {results['lsk']}")
    else:
        print("   [FAIL] Leapseconds kernel (naif*.tls) not found")
        all_ok = False
    
    if results.get('spk'):
        print(f"   [OK] Planetary ephemeris: {results['spk']}")
    else:
        print("   [FAIL] Planetary ephemeris (de440.bsp etc.) not found")
        all_ok = False
    
    print("\n" + "=" * 50)
    if all_ok:
        print("All checks passed. SPICE functionality is available.")
        sys.exit(0)
    else:
        print("Required kernel files missing. SPICE functionality limited.")
        sys.exit(1)
