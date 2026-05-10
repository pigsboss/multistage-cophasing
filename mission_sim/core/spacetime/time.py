"""
高精度时间系统转换模块（更随和版）

轻松搞定 UTC 字符串 (ISO 8601) 与常用时间系统间的双向转换：
  - TAI (International Atomic Time)
  - TT  (Terrestrial Time)
  - TDB (Barycentric Dynamical Time)
  - Julian Date (UTC)
  - Unix timestamp
  - 平滑 UTC 秒 (不含闰秒，自 J2000.0 UTC 起算)

闰秒数据从本地静态文件 ``Leap_Second.dat`` 读取，不再硬编码。
若文件缺失或过期，会在模块加载时提示用户运行
``tools/update_leap_second.py`` 进行初始化或更新。
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# 历元常量
# ---------------------------------------------------------------------------
J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)  # J2000.0 UTC
J2000_JD = 2451545.0  # Julian Date of J2000.0 (TT 历元，但常用作参考)
TAI_OFFSET_AT_J2000 = 32.0  # 在 J2000.0 UTC 时，TAI - UTC = 32 s

# ---------------------------------------------------------------------------
# 闰秒数据文件路径
# ---------------------------------------------------------------------------
_LEAP_FILE_PATH = os.path.join(os.path.dirname(__file__), "Leap_Second.dat")

# ---------------------------------------------------------------------------
# 从本地文件加载闰秒表
# ---------------------------------------------------------------------------
def _parse_leap_second_file(path: str) -> list:
    """
    读取 Leap_Second.dat 文件，返回闰秒生效日期 (datetime) 的升序列表。
    文件格式示例：
          41317.0    1  1 1972       10
    其中第2~4列分别为日、月、年；第5列为 TAI-UTC (s)，本函数仅使用日期。
    """
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                day = int(parts[1])
                month = int(parts[2])
                year = int(parts[3])
                # 第 5 列是 TAI-UTC 值，此处忽略
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                events.append(dt)
            except (ValueError, IndexError):
                continue
    events.sort()
    return events


def _check_file_freshness(path: str) -> None:
    """
    检查本地闰秒文件是否过期（超过当前半年周期）。
    若文件修改时间早于当前半年周期的开始，则认为过期并提示用户更新。
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return  # 文件不存在时已在调用处处理
    file_mtime = datetime.fromtimestamp(mtime, tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)

    # 半年周期：1月1日 / 7月1日
    if now.month >= 7:
        period_start = datetime(now.year, 7, 1, tzinfo=timezone.utc)
    else:
        period_start = datetime(now.year, 1, 1, tzinfo=timezone.utc)

    if file_mtime < period_start:
        print(
            f"⚠️  闰秒文件 {path} 可能已过期（最后修改于 {file_mtime.date()}）。\n"
            "   建议运行 `python tools/update_leap_second.py` 获取最新数据。",
            file=sys.stderr,
        )


# 实际加载并检查（仅在模块导入时执行一次）
if not os.path.exists(_LEAP_FILE_PATH):
    print(
        f"⚠️  闰秒文件 {_LEAP_FILE_PATH} 不存在。\n"
        "   请运行 `python tools/update_leap_second.py` 初始化本地数据。",
        file=sys.stderr,
    )
    _LEAP_SECONDS_DATES = []
else:
    _LEAP_SECONDS_DATES = _parse_leap_second_file(_LEAP_FILE_PATH)
    _check_file_freshness(_LEAP_FILE_PATH)


# ---------------------------------------------------------------------------
# 闰秒管理
# ---------------------------------------------------------------------------
def leap_seconds(utc_time: datetime) -> int:
    """
    返回给定 UTC datetime 之前的累计闰秒数 (TAI - UTC)。
    若时间早于第一个闰秒日期，则返回 0。
    """
    cnt = 0
    for d in _LEAP_SECONDS_DATES:
        if d <= utc_time:
            cnt += 1
        else:
            break
    return cnt


def add_leap_second(date_str: str) -> None:
    """
    向内存闰秒表中临时添加一个闰秒日期（不写入文件，仅本次进程有效）。

    Parameters
    ----------
    date_str : str
        ISO 格式日期, 如 "2026-01-01"
    """
    new_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    _LEAP_SECONDS_DATES.append(new_date)
    _LEAP_SECONDS_DATES.sort()


# ---------------------------------------------------------------------------
# 核心转换函数（保持不变）
# ---------------------------------------------------------------------------
def utc_string_to_utc_smooth(utc_iso: str) -> float:
    """将 UTC 字符串转换为平滑 UTC 秒数 (不含闰秒，自 J2000.0 UTC)。"""
    dt = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)
    return (dt - J2000_UTC).total_seconds()


def utc_smooth_to_utc_string(utc_smooth_sec: float) -> str:
    """反向：将平滑 UTC 秒数转换为 UTC ISO 字符串"""
    dt = J2000_UTC + timedelta(seconds=utc_smooth_sec)
    return dt.isoformat()


def utc_string_to_tai(utc_iso: str) -> float:
    """UTC -> TAI 秒 (自 J2000.0 TAI 历元)"""
    dt = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)
    utc_smooth = (dt - J2000_UTC).total_seconds()
    leap = leap_seconds(dt)
    return utc_smooth + leap - TAI_OFFSET_AT_J2000


def tai_to_utc_string(tai_sec: float) -> str:
    """TAI 秒 -> UTC ISO 字符串 (逆过程)"""
    for leap in range(0, 60):
        utc_smooth = tai_sec - leap + TAI_OFFSET_AT_J2000
        dt = J2000_UTC + timedelta(seconds=utc_smooth)
        if leap_seconds(dt) == leap:
            return dt.isoformat()
    raise ValueError(f"无法找到合法的 UTC 时间对应 TAI={tai_sec}")


def utc_string_to_tt(utc_iso: str) -> float:
    """UTC -> TT 秒 (自 J2000.0 TT 历元)"""
    tai = utc_string_to_tai(utc_iso)
    return tai + 32.184  # TAI 与 TT 的固定偏移


def tt_to_utc_string(tt_sec: float) -> str:
    """TT 秒 -> UTC 字符串"""
    tai = tt_sec - 32.184
    return tai_to_utc_string(tai)


def utc_string_to_tdb(utc_iso: str) -> float:
    """UTC -> TDB 秒 (自 J2000.0 TDB 历元)，简化解析近似"""
    tt = utc_string_to_tt(utc_iso)
    t_tt = tt / 86400.0 / 36525.0
    g = (357.528 + 35999.05 * t_tt) * math.radians(1)
    tdb_offset = 0.001658 * math.sin(g + 0.0167 * math.sin(g))
    return tt + tdb_offset


def tdb_to_utc_string(tdb_sec: float) -> str:
    """TDB 秒 -> UTC 字符串，迭代求解"""
    tt = tdb_sec
    for _ in range(5):
        t_tt = tt / 86400.0 / 36525.0
        g = (357.528 + 35999.05 * t_tt) * math.radians(1)
        offset = 0.001658 * math.sin(g + 0.0167 * math.sin(g))
        tt = tdb_sec - offset
    return tt_to_utc_string(tt)


def utc_string_to_jd(utc_iso: str) -> float:
    """UTC -> Julian Date (UTC 尺度的连续 JD)"""
    utc_smooth = utc_string_to_utc_smooth(utc_iso)
    return J2000_JD + utc_smooth / 86400.0


def jd_to_utc_string(jd: float) -> str:
    """Julian Date (UTC) -> UTC 字符串"""
    utc_smooth = (jd - J2000_JD) * 86400.0
    return utc_smooth_to_utc_string(utc_smooth)


def utc_string_to_unix(utc_iso: str) -> float:
    """UTC -> Unix timestamp (POSIX 秒)"""
    dt = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def unix_to_utc_string(unixtime: float) -> str:
    """Unix timestamp -> UTC 字符串"""
    dt = datetime.fromtimestamp(unixtime, tz=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# 兼容旧式函数 (如 astro.py 中的名称)
# ---------------------------------------------------------------------------
def utc2tai(utc_jd: float) -> float:
    """旧接口：UTC Julian date -> TAI Julian date (假设连续 UTC JD)"""
    utc_smooth = (utc_jd - J2000_JD) * 86400.0
    dt = J2000_UTC + timedelta(seconds=utc_smooth)
    leap = leap_seconds(dt)
    return utc_jd + leap / 86400.0


def utc2tdt(utc_jd: float) -> float:
    """UTC JD -> TDT (TT) JD"""
    return utc2tai(utc_jd) + 32.184 / 86400.0


def utc2tdb(utc_jd: float) -> float:
    """UTC JD -> TDB JD (简化)"""
    tdt = utc2tdt(utc_jd)
    jc = (utc_jd - J2000_JD) / 36525.0
    g = 2.0 * math.pi * (357.528 + 35999.05 * jc) / 360.0
    return tdt + 0.001658 * math.sin(g + 0.0167 * math.sin(g)) / 86400.0


def unix2utc(t: float) -> float:
    """Unix timestamp -> UTC Julian day"""
    return t / 86400.0 + 2440587.5


def utc2unix(utc_jd: float) -> float:
    """UTC Julian day -> Unix timestamp"""
    return (utc_jd - 2440587.5) * 86400.0
