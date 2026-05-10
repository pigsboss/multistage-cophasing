#!/usr/bin/env python3
"""
工具：更新闰秒数据 (Leap Second Data Updater)
-----------------------------------------------
从 IERS Bulletin C 的系列文本文件中自动提取所有历史闰秒日期。
通过遍历以下 URL 获得资料：
    https://datacenter.iers.org/data/16/bulletinc-{num:03d}.txt
其中 {num} 从 10 开始递增。

用法：
    python tools/update_leap_second.py                       # 仅显示
    python tools/update_leap_second.py --sync                # 显示并同步到运行时模块
    python tools/update_leap_second.py --start 10 --end 100  # 自定义编号范围

依赖：
    仅使用 Python 标准库 (urllib, datetime, argparse, re)。
"""

import argparse
import datetime
import re
import sys
import urllib.error
import urllib.request
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 数据获取
# ---------------------------------------------------------------------------

BULLETIN_URL_TEMPLATE = "https://datacenter.iers.org/data/16/bulletinc-{num:03d}.txt"


def fetch_bulletin(num: int) -> Optional[str]:
    """
    下载指定编号的 Bulletin C 文本。

    Parameters
    ----------
    num : int
        Bulletin 编号。

    Returns
    -------
    str or None
        如果文件存在则返回纯文本内容，否则返回 None。
    """
    url = BULLETIN_URL_TEMPLATE.format(num=num)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None

    # 提取 <pre> 标签内的原始文本
    match = re.search(r"<pre[^>]*>(.*?)</pre>", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 如果服务器直接返回纯文本（极少情况），直接使用
    return raw.strip()


# ---------------------------------------------------------------------------
# 文本解析
# ---------------------------------------------------------------------------

def parse_bulletin(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    从 Bulletin C 文本中提取闰秒日期和 UTC-TAI 偏移。

    Returns
    -------
    leap_date_str : str or None
        闰秒发生的日期（格式 YYYY-MM-DD），若无闰秒则返回 None。
    tai_utc_offset : int or None
        公告中声明的 UTC-TAI 差值（秒），取非负整数值（即 TAI-UTC）。
    """
    # 匹配闰秒公告行
    leap_match = re.search(
        r"A positive leap second will be introduced at the end of (\w+ \d{4})",
        text, re.IGNORECASE
    )
    leap_date = None
    if leap_match:
        month_year = leap_match.group(1)  # 如 "December 1995"
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        parts = month_year.split()
        if len(parts) == 2:
            month_str, year_str = parts[0].lower(), parts[1]
            month = month_map.get(month_str)
            if month is not None:
                year = int(year_str)
                # 闰秒在月末的 23:59:60 UTC 之后，即次月 1 日 0h UTC
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                # 构造该月最后一天的 datetime，然后加一天
                leap_dt = datetime.datetime(year, month, last_day, tzinfo=datetime.timezone.utc)
                leap_dt += datetime.timedelta(days=1)
                leap_date = leap_dt.strftime("%Y-%m-%d")

    # 匹配 UTC-TAI 差值（常见格式："UTC-TAI = -37 s"）
    offset_match = re.search(
        r"UTC-TAI\s*=\s*-(\d+)\s*s", text, re.IGNORECASE
    )
    tai_utc_offset = None
    if offset_match:
        tai_utc_offset = int(offset_match.group(1))
    else:
        # 备选："TAI-UTC = 37 s"
        alt_match = re.search(r"TAI-UTC\s*=\s*(\d+)\s*s", text, re.IGNORECASE)
        if alt_match:
            tai_utc_offset = int(alt_match.group(1))

    return leap_date, tai_utc_offset


# ---------------------------------------------------------------------------
# 自动遍历所有可用公告
# ---------------------------------------------------------------------------

def scan_all_bulletins(start: int = 10, max_attempts: int = 200) -> List[Tuple[str, int]]:
    """
    从 start 编号开始依次下载公告，提取所有闰秒事件。

    Returns
    -------
    entries : list of (date_str, tai_utc_offset)
        按日期递增排序的闰秒事件列表。
    """
    events = []
    consecutive_missing = 0

    for num in range(start, start + max_attempts):
        text = fetch_bulletin(num)
        if text is None:
            consecutive_missing += 1
            if consecutive_missing >= 5:  # 连续 5 个 404 则停止
                break
            continue
        else:
            consecutive_missing = 0

        leap_date, offset = parse_bulletin(text)
        if leap_date:
            events.append((leap_date, offset if offset is not None else 0))
            print(f"  [{num:03d}] Found leap second: {leap_date}  (TAI-UTC={offset})")
        else:
            print(f"  [{num:03d}] No leap second announced.")

    # 按日期排序
    events.sort(key=lambda x: x[0])
    return events


# ---------------------------------------------------------------------------
# 显示列表
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
        description="从 IERS Bulletin C 存档中提取闰秒日期。"
    )
    parser.add_argument(
        "--start", type=int, default=10,
        help="起始 Bulletin 编号 (默认: 10)"
    )
    parser.add_argument(
        "--end", type=int, default=200,
        help="最大尝试编号 (默认: 200)"
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="将新闰秒同步到 mission_sim.core.spacetime.time 模块。"
    )
    args = parser.parse_args()

    print("Scanning IERS Bulletin C archive for leap seconds...\n")
    events = scan_all_bulletins(start=args.start, max_attempts=args.end - args.start + 1)

    if not events:
        print("No leap second events found.")
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
