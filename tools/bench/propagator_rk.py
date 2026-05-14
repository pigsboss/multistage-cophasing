#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark the Runge‑Kutta integrators (RK45, DOP853, DP8(7)).

Measures speed and accuracy for a simple harmonic oscillator problem.
"""

import time
import numpy as np
from numba import njit

from mission_sim.utils.propagators.rk import (
    integrate_rk45,
    integrate_dop853,
    integrate_dp8,
    TABLE_DOP853,
    TABLE_DP8,
    _make_rk_step,
)


# ---------------------------------------------------------------------------
# Test problem: harmonic oscillator  y'' = -y,  y(0)=[1,0],  t∈[0,2π]
# ---------------------------------------------------------------------------
@njit
def harmonic_f(t, y):
    """d/dt [x, v] = [v, -x]"""
    return np.array([y[1], -y[0]])


def analytical_harmonic(t):
    """Return true [x, v] at time t for initial [1, 0]."""
    return np.array([np.cos(t), -np.sin(t)])


# ---------------------------------------------------------------------------
# Nonlinear test problem: van der Pol oscillator (mu = 2)
# ---------------------------------------------------------------------------
MU_VDP = 2.0

@njit
def vdp_f(t, y):
    """Van der Pol oscillator (nonlinear, smooth)."""
    x, v = y
    return np.array([v, MU_VDP * (1.0 - x * x) * v - x])


# ---------------------------------------------------------------------------
# Benchmark routine
# ---------------------------------------------------------------------------
def bench_integrator(name, integrate_fn, t0, y0, t_span, tol_dict, n_repeat=5):
    """
    Run the integrator n_repeat times, return (avg_time, max_error).
    """
    # Warm-up (precompile)
    _, _ = integrate_fn(harmonic_f, t0, y0, t_span, **tol_dict)

    times = []
    y_final = None
    for _ in range(n_repeat):
        start = time.perf_counter()
        t_arr, y_arr = integrate_fn(harmonic_f, t0, y0, t_span, **tol_dict)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        y_final = y_arr[-1].copy()

    avg_time = np.mean(times)
    true_final = analytical_harmonic(t_span[1])
    max_error = np.abs(y_final - true_final).max()
    return avg_time, max_error


# ---------------------------------------------------------------------------
# Local error estimate fidelity test
# ---------------------------------------------------------------------------
def bench_local_error_estimate():
    """
    Fixed-step single-step test on a nonlinear smooth problem.
    
    DP8(7) uses a 7th-order embedded formula (only 1 order below its 8th-
    order main solution), whereas DOP853 uses a 5th-order embedded formula
    (3 orders below).  Consequently DP8(7)'s |y_high - y_low| tracks the
    true |y_high - y_true| far more closely.
    """
    _step_dop = _make_rk_step(TABLE_DOP853)
    _step_dp8 = _make_rk_step(TABLE_DP8)

    y0 = np.array([2.0, 0.0])
    t0 = 0.0
    step_sizes = [0.5, 0.2, 0.1, 0.05]

    print("\n" + "=" * 72)
    print("Local Error Estimate Accuracy (fixed single step)")
    print("Problem: van der Pol oscillator (mu=2), t0=0, y0=[2,0]")
    print("Ratio = |y_high - y_low|_inf / |y_high - y_true|_inf")
    print("=" * 72)
    print(f"{'h':>8} {'Method':>10} {'Est error':>14} {'True error':>14} {'Ratio':>10}")
    print("-" * 72)

    for h in step_sizes:
        # Ground truth: DOP853 at very tight tolerance with tiny initial step
        _, y_ref_arr = integrate_dop853(
            vdp_f, t0, y0.copy(), (t0, t0 + h),
            rtol=1e-14, atol=1e-16, h0=min(1e-4, h * 1e-2)
        )
        y_true = y_ref_arr[-1]

        for name, table, step_fn in [
            ("DOP853", TABLE_DOP853, _step_dop),
            ("DP8(7)", TABLE_DP8, _step_dp8),
        ]:
            y_high, _, k = step_fn(vdp_f, t0, y0, h, 1e-14, 1e-16)

            # Reconstruct low-order solution from returned stages
            y_low = y0.copy()
            for i in range(table.s):
                y_low += h * table.B_low[i] * k[i]

            est_err = np.max(np.abs(y_high - y_low))
            true_err = np.max(np.abs(y_high - y_true))
            ratio = est_err / true_err if true_err > 1e-18 else np.nan

            print(f"{h:8.4f} {name:>10} {est_err:14.6e} {true_err:14.6e} {ratio:10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Runge‑Kutta Integrator Benchmark")
    print("Problem: harmonic oscillator, t∈[0, 2π], y0=[1,0]")
    print("=" * 72)

    t0, tf = 0.0, 2.0 * np.pi
    y0 = np.array([1.0, 0.0])

    # Tolerance configurations to test
    tol_configs = {
        "default": {"rtol": 1e-8,  "atol": 1e-12},
        "tight":   {"rtol": 1e-12, "atol": 1e-14},
        "loose":   {"rtol": 1e-4,  "atol": 1e-8},
    }

    integrators = [
        ("RK45",   integrate_rk45),
        ("DOP853", integrate_dop853),
        ("DP8(7)", integrate_dp8),
    ]

    for tol_name, tol_dict in tol_configs.items():
        print(f"\n--- Tolerance: rtol={tol_dict['rtol']}, atol={tol_dict['atol']} ---")
        header = f"{'Integrator':<10} {'Avg time (ms)':<15} {'Max error':<15}"
        print(header)
        print("-" * len(header))

        for name, integrate_fn in integrators:
            avg_t, max_err = bench_integrator(
                name, integrate_fn, t0, y0, (t0, tf), tol_dict
            )
            print(f"{name:<10} {avg_t*1e3:>12.4f} ms   {max_err:>12.6e}")

    # Local error estimate fidelity test
    bench_local_error_estimate()

    # Optional: show the fastest method
    print("\n" + "=" * 72)
    print("Benchmark complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
