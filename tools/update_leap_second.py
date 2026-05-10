#!/usr/bin/env python3
"""
工具：更新闰秒数据 (Leap Second Data Updater)
-----------------------------------------------
从巴黎天文台的官方文件 Leap_Second.dat 中获取最新闰秒数据，
并与本地缓存文件 mission_sim/core/spacetime/Leap_Second.dat 比较。
支持手动/自动更新本地文件和运行时模块。

用法：
    python tools/update_leap_second.py               # 对比并显示差异
    python tools/update_leap_second.py --sync        # 对比后自动更新本地文件和运行时表

依赖：
    仅使用 Python 标准库 (urllib, datetime, argparse, re, os)。
"""

import argparse
import datetime
import os
import re
import sys
import urllib.request
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------
REMOTE_URL = "https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "mission_sim", "core", "spacetime", "Leap_Second.dat")

# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------
def fetch_remote_data() -> str:
    """下载远程 Leap_Second.dat 文件内容"""
    try:
        with urllib.request.urlopen(REMOTE_URL, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"下载 Leap_Second.dat 失败: {e}") from e

# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def parse_leap_second_data(text: str) -> List[Tuple[str, int]]:
    """
    解析 Leap_Second.dat 文本，返回 (日期字符串 YYYY-MM-DD, TAI-UTC 偏移) 列表。
    """
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # 匹配数字列：MJD 日 月 年 TAI-UTC
        match = re.match(r'\s*(\d+\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if not match:
            continue
        day = int(match.group(2))
        month = int(match.group(3))
        year = int(match.group(4))
        tai_utc = int(match.group(5))
        date = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
        date_str = date.strftime("%Y-%m-%d")
        events.append((date_str, tai_utc))
    events.sort(key=lambda x: x[0])
    return events

# ---------------------------------------------------------------------------
# 本地文件操作
# ---------------------------------------------------------------------------
def load_local_data() -> Tuple[str, List[Tuple[str, int]]]:
    """读取本地 Leap_Second.dat 文本并解析，若不存在则返回空字符串和空列表"""
    if not os.path.exists(LOCAL_PATH):
        return "", []
    with open(LOCAL_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    events = parse_leap_second_data(text)
    return text, events

def save_local_data(text: str) -> None:
    """将最新数据文本写入本地文件"""
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
    with open(LOCAL_PATH, 'w', encoding='utf-8') as f:
        f.write(text)

# ---------------------------------------------------------------------------
# 比较与显示
# ---------------------------------------------------------------------------
def compare_events(local_events: List[Tuple[str, int]], remote_events: List[Tuple[str, int]]) -> None:
    """打印两个事件列表的差异"""
    local_dict = dict(local_events)
    remote_dict = dict(remote_events)

    added = set(remote_dict) - set(local_dict)
    removed = set(local_dict) - set(remote_dict)
    changed = {
        date for date in (set(local_dict) & set(remote_dict))
        if local_dict[date] != remote_dict[date]
    }

    if not (added or removed or changed):
        print("✅ 本地数据与远程数据完全一致，无需更新。")
        return

    print("\n⚠️  闰秒数据存在差异：")
    if added:
        print("  新增/未来日期:")
        for d in sorted(added):
            print(f"    + {d} (TAI-UTC={remote_dict[d]})")
    if removed:
        print("  远程丢失日期（不应出现）:")
        for d in sorted(removed):
            print(f"    - {d} (TAI-UTC={local_dict[d]})")
    if changed:
        print("  偏移量变更:")
        for d in sorted(changed):
            print(f"    ~ {d} : {local_dict[d]} -> {remote_dict[d]}")

# ---------------------------------------------------------------------------
# 同步到运行时模块
# ---------------------------------------------------------------------------
def sync_to_runtime(remote_events: List[Tuple[str, int]]) -> int:
    """将新的闰秒日期添加到 time 模块（幂等）。"""
    from mission_sim.core.spacetime.time import add_leap_second, _LEAP_SECONDS_DATES

    existing = {dt.strftime("%Y-%m-%d") for dt in _LEAP_SECONDS_DATES}
    added = 0
    for date_str, _ in remote_events:
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
        description="从巴黎天文台获取最新闰秒数据，并与本地文件进行比较。"
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="对比后自动更新本地文件 (Leap_Second.dat) 以及运行时模块。"
    )
    args = parser.parse_args()

    # 下载远程数据
    print("🌐 正在从巴黎天文台获取最新 Leap_Second.dat...")
    try:
        remote_text = fetch_remote_data()
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    remote_events = parse_leap_second_data(remote_text)

    # 加载本地数据
    print(f"📂 读取本地文件 {LOCAL_PATH}...")
    local_text, local_events = load_local_data()
    if not local_events:
        print("   本地文件不存在或为空，视为全新数据。")

    # 比较并显示差异
    compare_events(local_events, remote_events)

    if args.sync:
        # 更新本地文件
        print("\n🔄 正在更新本地文件...")
        save_local_data(remote_text)
        print("   本地 Leap_Second.dat 已更新。")

        # 更新运行时模块
        print("🔗 正在同步到运行时模块...")
        try:
            added = sync_to_runtime(remote_events)
            if added:
                print(f"   ✅ 已添加 {added} 条新闰秒记录到 time 模块。")
            else:
                print("   ✅ 运行时表已是最新，无需添加。")
        except Exception as e:
            print(f"   ❌ 同步运行时模块失败: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if local_events != remote_events:
            print("\n💡 提示：使用 --sync 参数可自动更新本地文件及运行时表。")

    print("\n✅ 检查完成。")

if __name__ == "__main__":
    main()
