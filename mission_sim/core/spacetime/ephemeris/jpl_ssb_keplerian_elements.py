# mission_sim/core/spacetime/ephemeris/jpl_ssb_keplerian_elements.py
"""
Hey there! Keplerian elements for solar‑system bodies at J2000.0.

Data source:
  JPL Solar System Dynamics Group
  Keplerian Elements and Rates, Table 1
  (with respect to J2000.0 mean ecliptic and equinox, valid 1800 AD – 2050 AD)

This module converts the raw columns "long.peri." and "long.node." into standard
longitude of ascending node Ω, argument of periapsis ω, and mean anomaly M₀.
All angles are in radians, time unit is Julian century (36525 days).

Each body dictionary contains:
  - a0, a_dot        (au, au/cy)
  - e0, e_dot        (—, rad/cy)
  - i0, i_dot        (rad, rad/cy)
  - Omega0, Omega_dot (rad, rad/cy)
  - omega0, omega_dot (rad, rad/cy)
  - M0, M_dot        (rad, rad/cy)

For EM Bary (Earth/Moon barycenter), the longitude of ascending node Ω and
its rate are both zero.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 原始表格数值 (便于日后核对)
# ---------------------------------------------------------------------------
_RAW = {
    "Mercury": {
        "a": (0.38709927, 0.00000037),
        "e": (0.20563593, 0.00001906),
        "I": (7.00497902, -0.00594749),
        "L": (252.25032350, 149472.67411175),
        "long.peri.": (77.45779628, 0.16047689),
        "long.node.": (48.33076593, -0.12534081),
    },
    "Venus": {
        "a": (0.72333566, 0.00000390),
        "e": (0.00677672, -0.00004107),
        "I": (3.39467605, -0.00078890),
        "L": (181.97909950, 58517.81538729),
        "long.peri.": (131.60246718, 0.00268329),
        "long.node.": (76.67984255, -0.27769418),
    },
    "EM Bary": {
        "a": (1.00000261, 0.00000562),
        "e": (0.01671123, -0.00004392),
        "I": (-0.00001531, -0.01294668),
        "L": (100.46457166, 35999.37244981),
        "long.peri.": (102.93768193, 0.32327364),
        "long.node.": (0.0, 0.0),
    },
    "Mars": {
        "a": (1.52371034, 0.00001847),
        "e": (0.09339410, 0.00007882),
        "I": (1.84969142, -0.00813131),
        "L": (-4.55343205, 19140.30268499),
        "long.peri.": (-23.94362959, 0.44441088),
        "long.node.": (49.55953891, -0.29257343),
    },
    "Jupiter": {
        "a": (5.20288700, -0.00011607),
        "e": (0.04838624, -0.00013253),
        "I": (1.30439695, -0.00183714),
        "L": (34.39644051, 3034.74612775),
        "long.peri.": (14.72847983, 0.21252668),
        "long.node.": (100.47390909, 0.20469106),
    },
    "Saturn": {
        "a": (9.53667594, -0.00125060),
        "e": (0.05386179, -0.00050991),
        "I": (2.48599187, 0.00193609),
        "L": (49.95424423, 1222.49362201),
        "long.peri.": (92.59887831, -0.41897216),
        "long.node.": (113.66242448, -0.28867794),
    },
    "Uranus": {
        "a": (19.18916464, -0.00196176),
        "e": (0.04725744, -0.00004397),
        "I": (0.77263783, -0.00242939),
        "L": (313.23810451, 428.48202785),
        "long.peri.": (170.95427630, 0.40805281),
        "long.node.": (74.01692503, 0.04240589),
    },
    "Neptune": {
        "a": (30.06992276, 0.00026291),
        "e": (0.00859048, 0.00005105),
        "I": (1.77004347, 0.00035372),
        "L": (-55.12002969, 218.45945325),
        "long.peri.": (44.96476227, -0.32241464),
        "long.node.": (131.78422574, -0.00508664),
    },
}

# ---------------------------------------------------------------------------
# 转换为标准 Kepler 根数 (Ω, ω, M₀) 并全部使用弧度
# ---------------------------------------------------------------------------
_deg = np.pi / 180.0
_CENTURY = 36525.0  # Julian 世纪 (日)

PLANETS_DATA = {}

for name, raw in _RAW.items():
    a0, a_dot = raw["a"]
    e0, e_dot = raw["e"]          # e 的变化率已经是 rad/cy
    I_deg, I_dot_deg = raw["I"]
    L_deg, L_dot_deg = raw["L"]
    lp_deg, lp_dot_deg = raw["long.peri."]
    ln_deg, ln_dot_deg = raw["long.node."]

    # 角度转弧度
    I0 = I_deg * _deg
    I_dot = I_dot_deg * _deg
    L0 = L_deg * _deg
    L_dot = L_dot_deg * _deg
    lp0 = lp_deg * _deg
    lp_dot = lp_dot_deg * _deg
    ln0 = ln_deg * _deg
    ln_dot = ln_dot_deg * _deg

    Omega0 = ln0
    Omega_dot = ln_dot

    omega0 = lp0 - ln0
    omega_dot = lp_dot - ln_dot

    M0 = L0 - lp0
    M_dot = L_dot - lp_dot

    # 确保角度在 [0, 2π) 内
    Omega0 = Omega0 % (2.0 * np.pi)
    omega0 = omega0 % (2.0 * np.pi)
    M0 = M0 % (2.0 * np.pi)

    PLANETS_DATA[name] = {
        "a0": a0, "a_dot": a_dot,
        "e0": e0, "e_dot": e_dot,
        "i0": I0, "i_dot": I_dot,
        "Omega0": Omega0, "Omega_dot": Omega_dot,
        "omega0": omega0, "omega_dot": omega_dot,
        "M0": M0, "M_dot": M_dot,
    }


# ---------------------------------------------------------------------------
# 便捷查询函数
# ---------------------------------------------------------------------------
def get_elements(body: str, jd_j2000: float = 0.0) -> np.ndarray:
    """
    返回指定天体在 J2000 后 jd_j2000 儒略世纪时刻的 6 个平均轨道根数
    (a, e, i, Ω, ω, M)，单位为米、弧度。

    若 jd_j2000=0（默认），返回的是 J2000.0 历元的瞬时平均根数。
    """
    d = PLANETS_DATA[body]
    d_cy = jd_j2000 * 36525.0 / 36525.0  # 实际上 jd_j2000 应为从 J2000 起算的儒略世纪数
    # 为保持接口简单，这里假定输入为 Julian 世纪数
    a = d["a0"] + d["a_dot"] * jd_j2000            # AU
    e = d["e0"] + d["e_dot"] * jd_j2000            # rad/cy 适用
    i = d["i0"] + d["i_dot"] * jd_j2000
    Omega = d["Omega0"] + d["Omega_dot"] * jd_j2000
    omega = d["omega0"] + d["omega_dot"] * jd_j2000
    M = d["M0"] + d["M_dot"] * jd_j2000
    # 归一化角度
    Omega = Omega % (2.0 * np.pi)
    omega = omega % (2.0 * np.pi)
    M = M % (2.0 * np.pi)
    return np.array([a, e, i, Omega, omega, M])
