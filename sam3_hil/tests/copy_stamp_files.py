#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copy_stamp_files.py — STAMP user study 檔案搬運工具

把每位參與者 (P1~P8) 在三個工具 (LabelMe / CVAT+SAM / STAMP) 下標註的結果，
依固定規則「複製」到 tests/exp/ 底下的實驗資料夾。原始檔案完全不動。

兩個主要指令：
    copy   依規則複製檔案（預設一次跑完 video + image，全部參與者）
    undo   一次刪除「上一次 copy 所新建立的檔案」（複製錯時用）

常用：
    python copy_stamp_files.py copy --dry-run          # 先預覽，不真的動檔案
    python copy_stamp_files.py copy                     # 正式複製
    python copy_stamp_files.py copy --mode video        # 只做 video
    python copy_stamp_files.py copy --participants 1 2   # 只做 P1, P2
    python copy_stamp_files.py undo --dry-run           # 預覽會刪哪些
    python copy_stamp_files.py undo                      # 真的刪掉上一次複製的
    python copy_stamp_files.py list                      # 列出歷次 copy 紀錄

每次 copy 都會在 <ROOT>/tests/exp/.stamp_copy_logs/ 留一份 manifest，
undo 只會刪掉「這次新建立」的檔案；若某個目的檔在複製前就已存在(被覆蓋)，
undo 不會碰它（避免誤刪你本來就有的東西）。
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
#  設定區（如果路徑或參與者數量不同，改這裡就好）
# ============================================================================

# 整個資料夾結構的根目錄。你訊息裡寫的是 home/boats_dataset_processing/sam3_hil/...
# 這裡假設是絕對路徑 /home/arg-4090/boats_dataset_processing/sam3_hil。
# 若不對，執行時用 --root 覆蓋，或直接改這行。
ROOT = Path("/home/arg-4090/boats_dataset_processing/sam3_hil")

PARTICIPANTS = list(range(1, 9))   # P1 ~ P8
VIDEOS = ["A", "B"]                # Video A, Video B

# 每位參與者「用 STAMP 標註的那一段」對應的 segment（從你上傳的對照表推出來）。
# req 2 (STAMP coco annotations) 與 req 4 (coco train2024 圖片) 的來源資料夾
# 都是 VideoY_QZ/coco/，這裡的 Z 就是 STAMP 那一欄。
STAMP_SEG = {
    (1, "A"): "Q3", (1, "B"): "Q4",
    (2, "A"): "Q4", (2, "B"): "Q1",
    (3, "A"): "Q1", (3, "B"): "Q2",
    (4, "A"): "Q2", (4, "B"): "Q3",
    (5, "A"): "Q4", (5, "B"): "Q1",
    (6, "A"): "Q1", (6, "B"): "Q2",
    (7, "A"): "Q2", (7, "B"): "Q3",
    (8, "A"): "Q3", (8, "B"): "Q4",
    # 以下 P9~P12 先放著，之後若加人可直接用
    (9, "A"): "Q2", (9, "B"): "Q3",
    (10, "A"): "Q3", (10, "B"): "Q4",
    (11, "A"): "Q4", (11, "B"): "Q1",
    (12, "A"): "Q1", (12, "B"): "Q2",
}

# req 4 要複製的圖片副檔名（train2024 內若全是圖片，也可設成 None 代表「全部檔案」）
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LOG_DIRNAME = ".stamp_copy_logs"

# ============================================================================
#  Job 規格：描述「一個來源 -> 一個目的」的搬運工作
# ============================================================================


class Job:
    """單一搬運工作。kind='files' 複製符合條件的檔案；kind='tree' 複製整個資料夾。"""

    def __init__(self, label, kind, src, dst, pattern=None, exts=None):
        self.label = label          # 報表用標籤，例如 "P1 / VideoA / LabelMe"
        self.kind = kind            # 'files' | 'tree'
        self.src = Path(src)
        self.dst = Path(dst)
        self.pattern = pattern      # files 模式：glob，例如 '*.json'；None=所有檔案
        self.exts = exts            # files 模式：限定副檔名集合；None=不限


def build_jobs(root: Path, participants, mode):
    """依規則展開所有搬運工作。mode in {'all','video','image'}。"""
    jobs = []
    exp = root / "tests" / "exp"

    do_video = mode in ("all", "video")
    do_image = mode in ("all", "image")

    for x in participants:
        # -------------------- VIDEO（Y = A, B）--------------------
        if do_video:
            for Y in VIDEOS:
                y = Y.lower()
                seg = STAMP_SEG.get((x, Y))           # e.g. "Q3"
                pxvy = exp / f"p{x}v{y}"              # tests/exp/p1va

                # req1: LabelMe json  participants/PX/VideoY/labelme/*.json -> pXvy/LabelMe/
                jobs.append(Job(
                    f"P{x}/Video{Y}/LabelMe(json)", "files",
                    root / "participants" / f"P{x}" / f"Video{Y}" / "labelme",
                    pxvy / "LabelMe",
                    pattern="*.json",
                ))

                # req2: STAMP coco annotations  data/.../VideoY_QZ/coco/annotations -> pXvy/STAMP/annotations
                stamp_coco = (root / "data" / "participants" / f"P{x}"
                              / f"Video{Y}_{seg}" / "coco") if seg else None
                jobs.append(Job(
                    f"P{x}/Video{Y}_{seg}/STAMP(annotations)", "tree",
                    (stamp_coco / "annotations") if stamp_coco else Path("__NO_SEG__"),
                    pxvy / "STAMP" / "annotations",
                ))

                # req3: CVAT annotations  data/.../videoy/annotations -> pXvy/CVAT/annotations
                jobs.append(Job(
                    f"P{x}/video{y}/CVAT(annotations)", "tree",
                    root / "data" / "participants" / f"P{x}" / f"video{y}" / "annotations",
                    pxvy / "CVAT" / "annotations",
                ))

                # req4: coco train2024 圖片  data/.../VideoY_QZ/coco/train2024/* -> pXvy/image/
                jobs.append(Job(
                    f"P{x}/Video{Y}_{seg}/image(train2024)", "files",
                    (stamp_coco / "train2024") if stamp_coco else Path("__NO_SEG__"),
                    pxvy / "image",
                    exts=IMAGE_EXTS,
                ))

        # -------------------- IMAGE（Image Set E）--------------------
        if do_image:
            pxie = exp / f"p{x}ie"                    # tests/exp/p1ie

            # req1: LabelMe json  participants/PX/ImageSetE/labelme/*.json -> pXie/LabelMe/
            jobs.append(Job(
                f"P{x}/ImageSetE/LabelMe(json)", "files",
                root / "participants" / f"P{x}" / "ImageSetE" / "labelme",
                pxie / "LabelMe",
                pattern="*.json",
            ))

            # req2: STAMP coco annotations  data/.../STAMP/coco/annotations -> pXie/STAMP/annotations
            jobs.append(Job(
                f"P{x}/STAMP/annotations(image)", "tree",
                root / "data" / "participants" / f"P{x}" / "STAMP" / "coco" / "annotations",
                pxie / "STAMP" / "annotations",
            ))

            # req3: CVAT annotations  data/.../imagee/annotations -> pXie/CVAT/annotations
            jobs.append(Job(
                f"P{x}/imagee/CVAT(annotations)", "tree",
                root / "data" / "participants" / f"P{x}" / "imagee" / "annotations",
                pxie / "CVAT" / "annotations",
            ))

            # req4: coco train2024 圖片  data/.../STAMP/coco/train2024/* -> pXie/image/
            jobs.append(Job(
                f"P{x}/STAMP/image(train2024)", "files",
                root / "data" / "participants" / f"P{x}" / "STAMP" / "coco" / "train2024",
                pxie / "image",
                exts=IMAGE_EXTS,
            ))

    return jobs


# ============================================================================
#  複製核心
# ============================================================================


def ensure_dir(path: Path, manifest, dry_run):
    """確保資料夾存在，記錄「本次新建立」的資料夾（給 undo 清理用）。"""
    ancestors = []
    cur = path
    while not cur.exists():
        ancestors.append(cur)
        cur = cur.parent
    for d in reversed(ancestors):          # 由上而下建立
        if not dry_run:
            d.mkdir(exist_ok=True)
        manifest["created_dirs"].append(str(d))


def copy_one_file(src: Path, dst: Path, manifest, dry_run, stats):
    """複製單一檔案，記錄是新建還是覆蓋。"""
    existed = dst.exists()
    if not dry_run:
        shutil.copy2(src, dst)             # copy2 保留 mtime 等 metadata
    if existed:
        manifest["overwritten_files"].append(str(dst))
        stats["overwritten"] += 1
    else:
        manifest["created_files"].append(str(dst))
        stats["copied"] += 1


def do_files_job(job: Job, manifest, dry_run, stats):
    if not job.src.exists() or not job.src.is_dir():
        manifest["missing_sources"].append(f"{job.label}  <-  {job.src}")
        stats["missing"] += 1
        return 0

    if job.pattern:
        candidates = sorted(job.src.glob(job.pattern))
    else:
        candidates = sorted(p for p in job.src.iterdir())
    # 只取檔案；若有限定副檔名再過濾
    files = [p for p in candidates if p.is_file()
             and (job.exts is None or p.suffix.lower() in job.exts)]

    if not files:
        # 來源在但沒檔案 — 當作可疑，記錄但不算 missing
        stats["empty"] += 1
        print(f"    [空] {job.label}: {job.src} 內沒有符合的檔案")
        return 0

    ensure_dir(job.dst, manifest, dry_run)
    for f in files:
        copy_one_file(f, job.dst / f.name, manifest, dry_run, stats)
    return len(files)


def do_tree_job(job: Job, manifest, dry_run, stats):
    if not job.src.exists() or not job.src.is_dir():
        manifest["missing_sources"].append(f"{job.label}  <-  {job.src}")
        stats["missing"] += 1
        return 0

    n = 0
    for src_file in sorted(job.src.rglob("*")):
        if src_file.is_file():
            rel = src_file.relative_to(job.src)
            dst_file = job.dst / rel
            ensure_dir(dst_file.parent, manifest, dry_run)
            copy_one_file(src_file, dst_file, manifest, dry_run, stats)
            n += 1
    if n == 0:
        stats["empty"] += 1
        print(f"    [空] {job.label}: {job.src} 是空資料夾")
    return n


def run_copy(root: Path, participants, mode, dry_run):
    jobs = build_jobs(root, participants, mode)

    manifest = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "mode": mode,
        "participants": participants,
        "created_files": [],
        "overwritten_files": [],
        "created_dirs": [],
        "missing_sources": [],
    }
    stats = {"copied": 0, "overwritten": 0, "missing": 0, "empty": 0}

    tag = "  (DRY-RUN，不會真的動檔案)" if dry_run else ""
    print(f"\n=== COPY  mode={mode}  participants={participants}{tag} ===")
    print(f"ROOT = {root}\n")

    for job in jobs:
        n = (do_files_job(job, manifest, dry_run, stats) if job.kind == "files"
             else do_tree_job(job, manifest, dry_run, stats))
        if n:
            verb = "would copy" if dry_run else "copied"
            print(f"    [{n:>4}] {verb:>10}  {job.label}  ->  {job.dst}")

    # 摘要
    print("\n--- 摘要 ---")
    print(f"  複製檔案(新建) : {stats['copied']}")
    print(f"  覆蓋既有檔案   : {stats['overwritten']}")
    print(f"  來源不存在     : {stats['missing']}")
    print(f"  來源為空       : {stats['empty']}")

    if manifest["missing_sources"]:
        print("\n--- 找不到的來源（請確認資料是否已就緒）---")
        for m in manifest["missing_sources"]:
            print(f"  ! {m}")

    if dry_run:
        print("\n(這是 DRY-RUN，沒有寫入任何檔案，也沒有存 manifest。)")
        return

    # 存 manifest（給 undo 用）
    log_dir = root / "tests" / "exp" / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    mpath = log_dir / f"{run_id}.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已記錄本次複製：{mpath}")
    print("若發現複製錯了，執行：  python copy_stamp_files.py undo")


# ============================================================================
#  Undo（刪掉上一次複製新建的檔案）
# ============================================================================


def latest_manifest(root: Path):
    log_dir = root / "tests" / "exp" / LOG_DIRNAME
    if not log_dir.exists():
        return None
    runs = sorted(log_dir.glob("run_*.json"))
    return runs[-1] if runs else None


def run_undo(root: Path, run_file, dry_run):
    mpath = Path(run_file) if run_file else latest_manifest(root)
    if not mpath or not mpath.exists():
        print("找不到任何 copy 紀錄(manifest)，沒有東西可以 undo。")
        return

    manifest = json.loads(Path(mpath).read_text(encoding="utf-8"))
    created_files = manifest.get("created_files", [])
    created_dirs = manifest.get("created_dirs", [])
    overwritten = manifest.get("overwritten_files", [])

    tag = "  (DRY-RUN，不會真的刪)" if dry_run else ""
    print(f"\n=== UNDO  使用紀錄：{mpath}{tag} ===")
    print(f"  時間 : {manifest.get('timestamp')}")
    print(f"  將刪除新建檔案 : {len(created_files)} 個")
    print(f"  將嘗試清空資料夾 : {len(set(created_dirs))} 個")
    if overwritten:
        print(f"  ⚠ 有 {len(overwritten)} 個檔案是『覆蓋既有檔』，undo 不會刪它們")

    deleted = 0
    for f in created_files:
        p = Path(f)
        if p.exists():
            if not dry_run:
                p.unlink()
            deleted += 1
    # 由深到淺刪空資料夾
    for d in sorted(set(created_dirs), key=lambda s: s.count("/"), reverse=True):
        p = Path(d)
        if p.exists() and p.is_dir():
            try:
                if not any(p.iterdir()):
                    if not dry_run:
                        p.rmdir()
            except OSError:
                pass

    print(f"\n  已{'(預計)' if dry_run else ''}刪除檔案 : {deleted}")

    if not dry_run:
        # 把這份 manifest 標記為已撤銷（改名，避免下次又被當成 latest）
        done = mpath.with_suffix(".undone.json")
        mpath.rename(done)
        print(f"  此紀錄已標記為已撤銷：{done}")


def list_runs(root: Path):
    log_dir = root / "tests" / "exp" / LOG_DIRNAME
    if not log_dir.exists():
        print("還沒有任何 copy 紀錄。")
        return
    active = [p for p in log_dir.glob("run_*.json") if not p.name.endswith(".undone.json")]
    undone = list(log_dir.glob("*.undone.json"))
    runs = sorted(active) + sorted(undone)
    if not runs:
        print("還沒有任何 copy 紀錄。")
        return
    print(f"\n=== copy 紀錄（{log_dir}）===")
    for r in runs:
        try:
            m = json.loads(r.read_text(encoding="utf-8"))
            state = "已撤銷" if r.name.endswith(".undone.json") else "可 undo"
            print(f"  {r.name}  [{state}]  {m.get('timestamp')}  "
                  f"新建 {len(m.get('created_files', []))} 檔  mode={m.get('mode')}")
        except Exception:
            print(f"  {r.name}  (無法讀取)")


# ============================================================================
#  CLI
# ============================================================================


def main():
    ap = argparse.ArgumentParser(description="STAMP user study 檔案搬運工具")
    ap.add_argument("--root", default=str(ROOT), help=f"資料夾根目錄(預設 {ROOT})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("copy", help="依規則複製檔案")
    pc.add_argument("--mode", choices=["all", "video", "image"], default="all")
    pc.add_argument("--participants", type=int, nargs="+", default=PARTICIPANTS,
                    help="指定參與者編號，例如 --participants 1 2 3")
    pc.add_argument("--dry-run", action="store_true", help="只預覽，不真的複製")

    pu = sub.add_parser("undo", help="刪除上一次複製新建的檔案")
    pu.add_argument("--run", default=None, help="指定某個 manifest 檔(預設最新一次)")
    pu.add_argument("--dry-run", action="store_true", help="只預覽，不真的刪")

    sub.add_parser("list", help="列出歷次 copy 紀錄")

    args = ap.parse_args()
    root = Path(args.root)

    if not root.exists():
        print(f"[錯誤] ROOT 不存在：{root}")
        print("       請用 --root 指定正確路徑，或修改檔案最上方的 ROOT。")
        sys.exit(1)

    if args.cmd == "copy":
        run_copy(root, args.participants, args.mode, args.dry_run)
    elif args.cmd == "undo":
        run_undo(root, args.run, args.dry_run)
    elif args.cmd == "list":
        list_runs(root)


if __name__ == "__main__":
    main()