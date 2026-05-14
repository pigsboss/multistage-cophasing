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

    # Optional: show the fastest method
    print("\n" + "=" * 72)
    print("Benchmark complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
