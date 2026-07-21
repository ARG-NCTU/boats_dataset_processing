#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_labelme_to_coco.py — 一次把所有 tests/exp/p*/LabelMe/ 的 Labelme JSON 轉成 COCO

直接 import 你現有的 labelme_to_coco_instances 的函式,行為跟你手動跑
    python tools/labelme_to_coco_instances.py --input tests/exp/p1va/LabelMe/ \
        --output tests/exp/p1va/LabelMe/p1_labelme_coco.json --single-category boat
完全一致,只是自動掃過全部資料夾。

預設掃 <exp>/p*/LabelMe(剛好 24 個:p1va..p8va / p1vb..p8vb / p1ie..p8ie),
不會掃到 old/ 底下(因為那是多一層 p*/old/.../LabelMe)。

每個資料夾輸出寫進該 LabelMe/ 內,檔名預設 p{X}_labelme_coco.json(跟你的範例一樣)。
加 --include-scene 會變成 p{X}{scene}_labelme_coco.json(例如 p1va_…),
跨場景就不會撞名,之後若要把 coco 集中到一處會比較安全。

用法:
    python tools/batch_labelme_to_coco.py --dry-run        # 先預覽會處理哪些
    python tools/batch_labelme_to_coco.py                  # 正式轉(single-category=boat)
    python tools/batch_labelme_to_coco.py --include-scene  # 輸出檔名帶場景
    python tools/batch_labelme_to_coco.py --participants 1 2   # 只做 P1, P2
"""

import argparse
import json
import re
import sys
from pathlib import Path

# 讓本檔不管從哪裡執行都能 import 同目錄的轉換器
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from labelme_to_coco_instances import (  # noqa: E402
    convert_labelme_documents,
    iter_labelme_documents,
)

ROOT = Path("/home/arg-4090/boats_dataset_processing/sam3_hil")
VIDEOS = ["A", "B", "C", "D"]
FOLDER_RE = re.compile(r"^p(\d+)(va|vb|vc|vd|ie)$")   # p1va / p3vd / p8ie ...


def wanted_scenes(mode, videos):
    """回傳要處理的場景代號集合,例如 {'vc','vd'} 或 {'va','vb','vc','vd','ie'}。"""
    scenes = set()
    if mode in ("all", "video"):
        scenes |= {f"v{Y.lower()}" for Y in videos}
    if mode in ("all", "image"):
        scenes.add("ie")
    return scenes


def discover(exp_dir: Path, participants, scenes):
    """找出 exp_dir/p*/LabelMe 的資料夾,回傳 [(px_num, scene, labelme_dir), ...]。"""
    found = []
    for labelme_dir in sorted(exp_dir.glob("p*/LabelMe")):
        if not labelme_dir.is_dir():
            continue
        m = FOLDER_RE.match(labelme_dir.parent.name)
        if not m:
            continue
        px = int(m.group(1))
        scene = m.group(2)
        if participants and px not in participants:
            continue
        if scene not in scenes:
            continue
        found.append((px, scene, labelme_dir))
    return found


def out_name(px, scene, include_scene):
    return f"p{px}{scene}_labelme_coco.json" if include_scene else f"p{px}_labelme_coco.json"


def main():
    ap = argparse.ArgumentParser(description="批量 Labelme -> COCO")
    ap.add_argument("--root", default=str(ROOT), help=f"sam3_hil 根目錄(預設 {ROOT})")
    ap.add_argument("--exp-dir", default=None, help="直接指定 exp 目錄(預設 <root>/tests/exp)")
    ap.add_argument("--single-category", default="boat",
                    help="全部歸到一個類別,預設 boat;設成空字串 '' 則用 boat1->boat 規則")
    ap.add_argument("--include-scene", action="store_true",
                    help="輸出檔名帶場景 p{X}{scene}_labelme_coco.json")
    ap.add_argument("--participants", type=int, nargs="+", default=None,
                    help="只處理指定參與者,例如 --participants 1 2 3")
    ap.add_argument("--videos", nargs="+", choices=VIDEOS, default=None,
                    help="只處理指定影片,例如 --videos C D。預設全部 "
                         + " ".join(VIDEOS) + "。指定時若未給 --mode 則自動只做 video")
    ap.add_argument("--mode", choices=["all", "video", "image"], default=None,
                    help="做 video / image / all。預設 all;"
                         "但若指定了 --videos 而沒指定 --mode,會自動視為 video")
    ap.add_argument("--dry-run", action="store_true", help="只預覽,不寫檔")
    args = ap.parse_args()

    root = Path(args.root)
    exp_dir = Path(args.exp_dir) if args.exp_dir else root / "tests" / "exp"
    if not exp_dir.exists():
        print(f"[錯誤] 找不到 exp 目錄:{exp_dir}  (用 --root 或 --exp-dir 指定)")
        sys.exit(1)

    single_category = args.single_category if args.single_category != "" else None

    # 指定了 --videos 卻沒指定 --mode → 視為只做 video（跟 copy_stamp_all.py 一致）。
    videos = args.videos if args.videos else VIDEOS
    mode = args.mode if args.mode is not None else ("video" if args.videos else "all")
    scenes = wanted_scenes(mode, videos)

    jobs = discover(exp_dir, set(args.participants) if args.participants else None, scenes)

    tag = "  (DRY-RUN,不寫檔)" if args.dry_run else ""
    print(f"\n=== Labelme -> COCO  mode={mode}  videos={videos}  "
          f"single_category={single_category!r}{tag} ===")
    print(f"exp = {exp_dir}\n場景 = {sorted(scenes)}\n找到 {len(jobs)} 個 LabelMe 資料夾\n")

    ok = empty = failed = 0
    for px, scene, labelme_dir in jobs:
        out_path = labelme_dir / out_name(px, scene, args.include_scene)
        try:
            coco = convert_labelme_documents(
                iter_labelme_documents(labelme_dir),
                single_category=single_category,
            )
        except Exception as exc:                      # noqa: BLE001
            print(f"    [失敗] p{px}{scene}: {exc}")
            failed += 1
            continue

        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        if n_img == 0:
            print(f"    [空]   p{px}{scene}: {labelme_dir} 內沒有有效的 Labelme JSON")
            empty += 1
            continue

        if not args.dry_run:
            out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        verb = "would write" if args.dry_run else "wrote"
        print(f"    [OK]   p{px}{scene}: {n_img} imgs / {n_ann} anns  {verb} -> {out_path.name}")
        ok += 1

    print("\n--- 摘要 ---")
    print(f"  成功 : {ok}")
    print(f"  空的 : {empty}")
    print(f"  失敗 : {failed}")
    if args.dry_run:
        print("\n(DRY-RUN,沒有寫任何檔。)")


if __name__ == "__main__":
    main()
