#!/usr/bin/env python3
"""
工具：更新闰秒数据 (Leap Second Data Updater)
-----------------------------------------------
从巴黎天文台镜像（IERS 公报 C 的规范来源）获取最新的闰秒表，
解析所有宣布的闰秒插入日期，并在终端中显示它们。

用法：
    python tools/update_leap_second.py                       # 仅显示
    python tools/update_leap_second.py --sync                # 显示并更新运行时间模块
    python tools/update_leap_second.py --sync --url <URL>    # 指定自定义数据源

说明：
    默认数据源：https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat
    此为巴黎天文台维护的公开纯文本文件，是闰秒数据的权威来源。

--sync 选项将调用 mission_sim.core.spacetime.time 模块中的 add_leap_second()
函数，把任何新发现的闰秒日期追加到内部的 _LEAP_SECONDS_DATES 列表。该模块的
缓存更新是幂等的：已存在的日期不会被重复添加。

依赖：
    仅使用 Python 标准库 (urllib, datetime, argparse)。
    为了运行 --sync 选项，需要在项目根目录下执行，确保 mission_sim 包可导入。
"""

import argparse
import datetime
import sys
import urllib.request

# ---------------------------------------------------------------------------
# 默认数据 URL
# ---------------------------------------------------------------------------
DEFAULT_LEAP_SECOND_URL = (
    "https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat"
)


# ---------------------------------------------------------------------------
# 获取原始数据
# ---------------------------------------------------------------------------
def fetch_leap_seconds_data(url: str) -> str:
    """
    从给定的 URL 获取 Leap_Second.dat 文件内容。

    Parameters
    ----------
    url : str
        数据文件 URL。

    Returns
    -------
    str
        文件全部文本。

    Raises
    ------
    urllib.error.URLError
        如果网络请求失败。
    """
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read().decode("utf-8")
    return raw


# ---------------------------------------------------------------------------
# 解析数据
# ---------------------------------------------------------------------------
def parse_leap_seconds_data(text: str) -> list[tuple[str, int]]:
    """
    解析 Leap_Second.dat 文本，返回 (日期字符串, TAI-UTC 差值) 元组列表。

    格式示例::
        1972-01-01  10
        1972-07-01  11
        1973-01-01  12
        ...

    以 '#' 开头的行将被忽略。

    Parameters
    ----------
    text : str
        Leap_Second.dat 的完整文本内容。

    Returns
    -------
    list of (str, int)
        列表中第一个元素对应最早的闰秒。
    """
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        date_str = parts[0]
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        # 验证日期可解析
        try:
            datetime.datetime.fromisoformat(date_str)
        except ValueError:
            continue
        entries.append((date_str, offset))
    # 按日期排序（确保顺序）
    entries.sort(key=lambda x: x[0])
    return entries


# ---------------------------------------------------------------------------
# 显示列表
# ---------------------------------------------------------------------------
def display_leap_seconds(entries: list[tuple[str, int]]) -> None:
    """
    友好打印闰秒日期及对应的 TAI-UTC 差值。

    Parameters
    ----------
    entries : list of (str, int)
        解析结果。
    """
    header = f"{'Date':<15} {'TAI-UTC (s)':<12}"
    print(header)
    print("-" * len(header))
    for date_str, offset in entries:
        print(f"{date_str:<15} {offset:<12}")
    print(f"\nTotal records: {len(entries)}")


# ---------------------------------------------------------------------------
# 同步到运行时模块
# ---------------------------------------------------------------------------
def sync_to_runtime(entries: list[tuple[str, int]]) -> int:
    """
    将解析出的新闰秒日期同步到运行时的 time 模块。

    调用 mission_sim.core.spacetime.time.add_leap_second() 为每个
    尚未存在于内部列表中的日期添加条目。该函数是幂等的。

    Parameters
    ----------
    entries : list of (str, int)
        从权威源解析得到的闰秒日期列表。

    Returns
    -------
    int
        新增的闰秒条目数量。
    """
    from mission_sim.core.spacetime.time import add_leap_second, _LEAP_SECONDS_DATES

    # 构建已经存在的日期集合（仅日期部分，忽略时分秒）
    existing = set()
    for dt in _LEAP_SECONDS_DATES:
        existing.add(dt.strftime("%Y-%m-%d"))

    added = 0
    for date_str, _ in entries:
        if date_str not in existing:
            add_leap_second(date_str)
            added += 1
            existing.add(date_str)  # 避免本批次重复

    return added


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="获取并解析 IERS 闰秒数据，可选同步到运行时模块。"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_LEAP_SECOND_URL,
        help=f"Leap_Second.dat 数据的 URL（默认: {DEFAULT_LEAP_SECOND_URL}）",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="将新发现的闰秒日期同步到 mission_sim.core.spacetime.time 模块。",
    )
    args = parser.parse_args()

    # 1. 获取数据
    print(f"Fetching leap second data from: {args.url}")
    try:
        text = fetch_leap_seconds_data(args.url)
    except Exception as e:
        print(f"ERROR: Failed to fetch data: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. 解析
    entries = parse_leap_seconds_data(text)
    if not entries:
        print("WARNING: No leap second records found in the data.", file=sys.stderr)
        sys.exit(0)

    # 3. 显示
    print("\n===== IERS Leap Second Data =====")
    display_leap_seconds(entries)

    # 4. 同步
    if args.sync:
        try:
            added = sync_to_runtime(entries)
        except ImportError as e:
            print(
                f"\nERROR: Cannot import mission_sim. "
                f"Make sure you run this script from the project root directory.\n"
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
