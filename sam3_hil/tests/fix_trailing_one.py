#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_trailing_one.py — 把 data/participants/PX/ 底下「不小心多打了結尾 1」的資料夾改回來

只處理 data/participants/P*/ 的「直接子資料夾」(不遞迴進 coco 等子層)。
只動資料夾，檔案(含 .zip)一律不碰。

安全機制：
  * 自動保護合法的段落資料夾：去掉尾巴 1 之後若變成 "..._Q"(代表原本是合法的 _Q1)，不動。
  * 若改名後的目標名稱已經存在(例如 videob 和 videob1 同時在)，跳過不覆蓋。
  * 預設只是「掃描預覽」，要真的改名必須加 --apply。
  * 每次 apply 會寫一份 log，可用 undo 還原。

用法：
    python fix_trailing_one.py scan            # 預覽會改哪些(預設、安全)
    python fix_trailing_one.py apply            # 真的改名(會存還原紀錄)
    python fix_trailing_one.py undo             # 還原上一次改名
    可加 --root <path> 覆蓋根目錄
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/arg-4090/boats_dataset_processing/sam3_hil")
LOG_DIRNAME = ".rename_logs"

# 去掉尾巴 1 後，若符合這個樣式代表「原本就是合法名稱」，要保護不動。
# 例：VideoB_Q1 -> 去 1 -> VideoB_Q (結尾 _Q) → 合法段落，保護。
PROTECT_AFTER_STRIP = re.compile(r"_Q$")


def find_candidates(root: Path):
    """回傳直接子資料夾中、名稱以 1 結尾的清單，並判定每一個的處理狀態。"""
    base = root / "data" / "participants"
    rows = []  # dict: px, old, new, status, old_path, new_path
    if not base.exists():
        return rows, base
    for px in sorted(base.glob("P*")):
        if not px.is_dir():
            continue
        for child in sorted(px.iterdir()):
            if not child.is_dir():
                continue                      # 檔案(含 .zip)跳過
            name = child.name
            if not name.endswith("1"):
                continue                      # 只看結尾是 1 的
            new_name = name[:-1]              # 砍掉一個結尾的 1
            new_path = child.with_name(new_name)

            if new_name == "":
                status = "SKIP_EMPTY"          # 名字只有 "1"，不處理
            elif PROTECT_AFTER_STRIP.search(new_name):
                status = "PROTECT_SEGMENT"     # 合法 _Q1 段落，保護
            elif new_path.exists():
                status = "SKIP_EXISTS"         # 目標已存在，避免覆蓋
            else:
                status = "RENAME"
            rows.append({
                "px": px.name, "old": name, "new": new_name, "status": status,
                "old_path": str(child), "new_path": str(new_path),
            })
    return rows, base


STATUS_LABEL = {
    "RENAME": "改名",
    "PROTECT_SEGMENT": "保護(合法_Q1段落)",
    "SKIP_EXISTS": "跳過(目標已存在)",
    "SKIP_EMPTY": "跳過(名稱只有1)",
}


def print_table(rows):
    if not rows:
        print("  沒有任何結尾是 1 的資料夾。")
        return
    width = max(len(r["old"]) for r in rows) + 2
    for r in rows:
        arrow = f'{r["old"]:<{width}} -> {r["new"]}' if r["status"] == "RENAME" \
            else f'{r["old"]}'
        print(f'  [{STATUS_LABEL[r["status"]]:<16}] {r["px"]:<4} {arrow}')


def cmd_scan(root, rows):
    print(f"\n=== SCAN  (預覽，不會改任何東西) ===\nROOT = {root}\n")
    print_table(rows)
    to_do = [r for r in rows if r["status"] == "RENAME"]
    protect = [r for r in rows if r["status"] == "PROTECT_SEGMENT"]
    skip = [r for r in rows if r["status"].startswith("SKIP")]
    print("\n--- 摘要 ---")
    print(f"  將改名         : {len(to_do)}")
    print(f"  保護(合法段落) : {len(protect)}")
    print(f"  跳過           : {len(skip)}")
    if to_do:
        print("\n確認上面『改名』清單沒問題後，執行：  python fix_trailing_one.py apply")


def cmd_apply(root, rows):
    to_do = [r for r in rows if r["status"] == "RENAME"]
    print(f"\n=== APPLY  ===\nROOT = {root}\n")
    if not to_do:
        print("  沒有需要改名的資料夾。")
        # 仍把保護/跳過的列出來讓你知道
        others = [r for r in rows if r["status"] != "RENAME"]
        if others:
            print("  (以下被保護或跳過：)")
            print_table(others)
        return

    print("  將進行以下改名：")
    print_table(to_do)

    done = []
    for r in to_do:
        old_p, new_p = Path(r["old_path"]), Path(r["new_path"])
        if new_p.exists():
            print(f"  ! 改名前發現目標已存在，跳過：{new_p}")
            continue
        old_p.rename(new_p)
        done.append({"from": str(old_p), "to": str(new_p)})

    log_dir = root / "data" / "participants" / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / datetime.now().strftime("rename_%Y%m%d_%H%M%S.json")
    log_path.write_text(json.dumps(
        {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "renames": done},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  完成改名 {len(done)} 個。")
    print(f"  還原紀錄：{log_path}")
    print("  若要還原，執行：  python fix_trailing_one.py undo")


def latest_log(root: Path):
    log_dir = root / "data" / "participants" / LOG_DIRNAME
    if not log_dir.exists():
        return None
    logs = sorted(p for p in log_dir.glob("rename_*.json")
                  if not p.name.endswith(".undone.json"))
    return logs[-1] if logs else None


def cmd_undo(root):
    lp = latest_log(root)
    if not lp:
        print("找不到改名紀錄，沒有東西可以還原。")
        return
    data = json.loads(lp.read_text(encoding="utf-8"))
    renames = data.get("renames", [])
    print(f"\n=== UNDO  使用紀錄：{lp} ===")
    print(f"  將把 {len(renames)} 個資料夾改回原名\n")
    n = 0
    for r in reversed(renames):
        cur = Path(r["to"])        # 現在的名字(去掉1後的)
        orig = Path(r["from"])     # 原本(帶1)的名字
        if not cur.exists():
            print(f"  ! 找不到 {cur}，跳過")
            continue
        if orig.exists():
            print(f"  ! 原名 {orig} 已存在，跳過避免覆蓋")
            continue
        cur.rename(orig)
        print(f"  {cur.name} -> {orig.name}")
        n += 1
    lp.rename(lp.with_suffix(".undone.json"))
    print(f"\n  已還原 {n} 個。此紀錄標記為已撤銷。")


def main():
    ap = argparse.ArgumentParser(description="移除 data/participants/PX 資料夾結尾多打的 1")
    ap.add_argument("--root", default=str(ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", help="預覽(預設)")
    sub.add_parser("apply", help="真的改名")
    sub.add_parser("undo", help="還原上一次改名")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[錯誤] ROOT 不存在：{root}  請用 --root 指定。")
        sys.exit(1)

    if args.cmd == "undo":
        cmd_undo(root)
        return

    rows, base = find_candidates(root)
    if not base.exists():
        print(f"[錯誤] 找不到 {base}")
        sys.exit(1)

    if args.cmd == "scan":
        cmd_scan(root, rows)
    elif args.cmd == "apply":
        cmd_apply(root, rows)


if __name__ == "__main__":
    main()