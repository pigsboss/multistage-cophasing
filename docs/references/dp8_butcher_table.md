# DP8 (Dormand‑Prince 8(7)) Butcher Tableau

## Description

The DP8 integrator implemented in `mission_sim/utils/propagators/rk.py` is a 13‑stage embedded Runge–Kutta method of orders 8 and 7, published by **Prince & Dormand (1981)**.  
The method is FSAL (First Same As Last): the last stage evaluation is reused as the first stage of the next step.

---

## Coefficients

### Node vector `c` (13 elements)

