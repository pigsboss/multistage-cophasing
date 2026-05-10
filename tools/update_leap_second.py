#!/usr/bin/env python3
"""
工具：更新闰秒数据 (Leap Second Data Updater)
-----------------------------------------------
从巴黎天文台的官方文件 Leap_Second.dat 中提取所有历史闰秒日期。
文件地址：https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat

用法：
    python tools/update_leap_second.py                       # 仅显示
    python tools/update_leap_second.py --sync                # 显示并同步到运行时模块

依赖：
    仅使用 Python 标准库 (urllib, datetime, argparse, re)。
"""

import argparse
import datetime
import re
import sys
import urllib.request
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------
LEAP_SECOND_FILE_URL = "https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat"

# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------
def fetch_leap_second_data() -> str:
    """下载 Leap_Second.dat 文件的内容"""
    try:
        with urllib.request.urlopen(LEAP_SECOND_FILE_URL, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"下载 Leap_Second.dat 失败: {e}") from e

# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def parse_leap_second_file(text: str) -> List[Tuple[str, int]]:
    """
    解析 Leap_Second.dat 文本，返回 (日期字符串 YYYY-MM-DD, TAI-UTC 偏移) 列表。
    文件格式示例：
        41317.0    1  1 1972       10
    其中第一列为 MJD，第二至四列为日、月、年，第五列为 TAI-UTC 差值。
    """
    events = []
    # 只处理数据行（数字开头）
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # 匹配数字列：MJD 日 月 年 TAI-UTC
        match = re.match(r'\s*(\d+\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if not match:
            continue
        mjd = float(match.group(1))
        day = int(match.group(2))
        month = int(match.group(3))
        year = int(match.group(4))
        tai_utc = int(match.group(5))

        # 构造 UTC 日期（MJD 自 1858-11-17 开始，也可以直接用年月日）
        date = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
        date_str = date.strftime("%Y-%m-%d")
        events.append((date_str, tai_utc))

    # 按日期排序（应当已经有序，但为了安全）
    events.sort(key=lambda x: x[0])
    return events

# ---------------------------------------------------------------------------
# 显示
# ---------------------------------------------------------------------------
def display_events(events: List[Tuple[str, int]]) -> None:
    """友好打印闰秒列表"""
    header = f"{'Date':<15} {'TAI-UTC (s)':<12}"
    print(header)
    print("-" * len(header))
    for date_str, offset in events:
        print(f"{date_str:<15} {offset:<12}")
    print(f"\nTotal records: {len(events)}")

# ---------------------------------------------------------------------------
# 同步到运行时模块
# ---------------------------------------------------------------------------
def sync_to_runtime(events: List[Tuple[str, int]]) -> int:
    """将新的闰秒日期添加到 time 模块（幂等）。"""
    from mission_sim.core.spacetime.time import add_leap_second, _LEAP_SECONDS_DATES

    existing = {dt.strftime("%Y-%m-%d") for dt in _LEAP_SECONDS_DATES}
    added = 0
    for date_str, _ in events:
        if date_str not in existing:
            add_leap_second(date_str)
            added += 1
            existing.add(date_str)
    return added

# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="从巴黎天文台 Leap_Second.dat 文件提取闰秒日期。"
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="将新闰秒同步到 mission_sim.core.spacetime.time 模块。"
    )
    args = parser.parse_args()

    print("Fetching Leap_Second.dat from Paris Observatory...")
    try:
        raw = fetch_leap_second_data()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    events = parse_leap_second_file(raw)
    if not events:
        print("No leap second entries found in the file.")
        return

    print("\n===== Extracted Leap Second History =====")
    display_events(events)

    if args.sync:
        try:
            added = sync_to_runtime(events)
        except ImportError as e:
            print(
                "\nERROR: Cannot import mission_sim. "
                "Make sure you run this script from the project root directory.\n"
                f"  {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            print(f"\nERROR: Sync failed: {e}", file=sys.stderr)
            sys.exit(1)

        if added:
            print(f"\n✅ Added {added} new leap second record(s) to the runtime module.")
        else:
            print("\n✅ No new leap second records to add (all already present).")

    print("\nDone.")

if __name__ == "__main__":
    main()
