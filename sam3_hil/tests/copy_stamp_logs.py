#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copy_stamp_logs.py — 把 STAMP 的 logs 資料夾複製到 tests/exp 的 STAMP/ 底下

video : data/participants/PX/VideoY_QZ/logs/  ->  tests/exp/pXvy/STAMP/logs/
image : data/participants/PX/STAMP/logs/      ->  tests/exp/pXie/STAMP/logs/
(Z = 該 P 的 STAMP 段落,從對照表推出;原始檔不動,只複製。)

指令:
    python copy_stamp_logs.py copy --dry-run        # 預覽
    python copy_stamp_logs.py copy                  # 正式複製
    python copy_stamp_logs.py copy --mode video      # 只做 video / image / all
    python copy_stamp_logs.py copy --participants 1 2
    python copy_stamp_logs.py undo                   # 刪掉上一次複製新建的檔
    python copy_stamp_logs.py list                   # 列出歷次紀錄
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/arg-4090/boats_dataset_processing/sam3_hil")
PARTICIPANTS = list(range(1, 9))
VIDEOS = ["A", "B"]
LOG_DIRNAME = ".stamp_logs_copy_logs"

# 各 P 用 STAMP 標的段落(對照表 STAMP 欄)。
STAMP_SEG = {
    (1, "A"): "Q3", (1, "B"): "Q4", (2, "A"): "Q4", (2, "B"): "Q1",
    (3, "A"): "Q1", (3, "B"): "Q2", (4, "A"): "Q2", (4, "B"): "Q3",
    (5, "A"): "Q4", (5, "B"): "Q1", (6, "A"): "Q1", (6, "B"): "Q2",
    (7, "A"): "Q2", (7, "B"): "Q3", (8, "A"): "Q3", (8, "B"): "Q4",
    (9, "A"): "Q2", (9, "B"): "Q3", (10, "A"): "Q3", (10, "B"): "Q4",
    (11, "A"): "Q4", (11, "B"): "Q1", (12, "A"): "Q1", (12, "B"): "Q2",
}


class Job:
    def __init__(self, label, src, dst):
        self.label = label
        self.src = Path(src)
        self.dst = Path(dst)


def _resolve_video_logs(root, x, Y, seg):
    """回傳 VideoY_QZ/logs 路徑;若精確段落資料夾不在,嘗試 VideoY_Q* 唯一候選。"""
    base = root / "data" / "participants" / f"P{x}"
    exact = base / f"Video{Y}_{seg}" / "logs"
    if exact.parent.exists():
        return exact
    cands = sorted(p for p in base.glob(f"Video{Y}_Q*") if p.is_dir())
    if len(cands) == 1:
        return cands[0] / "logs"
    return exact  # 交給呼叫端報 missing


def build_jobs(root, participants, mode):
    jobs = []
    exp = root / "tests" / "exp1"
    do_video = mode in ("all", "video")
    do_image = mode in ("all", "image")
    for x in participants:
        if do_video:
            for Y in VIDEOS:
                y = Y.lower()
                seg = STAMP_SEG.get((x, Y))
                src = _resolve_video_logs(root, x, Y, seg)
                jobs.append(Job(f"P{x}/Video{Y}_{seg}/logs", src,
                                exp / f"p{x}v{y}" / "STAMP" / "logs"))
        if do_image:
            jobs.append(Job(f"P{x}/STAMP/logs(image)",
                            root / "data" / "participants" / f"P{x}" / "STAMP" / "logs",
                            exp / f"p{x}ie" / "STAMP" / "logs"))
    return jobs


def ensure_dir(path: Path, manifest, dry_run):
    ancestors = []
    cur = path
    while not cur.exists():
        ancestors.append(cur)
        cur = cur.parent
    for d in reversed(ancestors):
        if not dry_run:
            d.mkdir(exist_ok=True)
        manifest["created_dirs"].append(str(d))


def copy_one_file(src, dst, manifest, dry_run, stats):
    existed = dst.exists()
    if not dry_run:
        shutil.copy2(src, dst)
    if existed:
        manifest["overwritten_files"].append(str(dst))
        stats["overwritten"] += 1
    else:
        manifest["created_files"].append(str(dst))
        stats["copied"] += 1


def do_tree_job(job: Job, manifest, dry_run, stats):
    if not job.src.exists() or not job.src.is_dir():
        manifest["missing_sources"].append(f"{job.label}  <-  {job.src}")
        stats["missing"] += 1
        return 0
    n = 0
    for src_file in sorted(job.src.rglob("*")):
        if src_file.is_file():
            dst_file = job.dst / src_file.relative_to(job.src)
            ensure_dir(dst_file.parent, manifest, dry_run)
            copy_one_file(src_file, dst_file, manifest, dry_run, stats)
            n += 1
    if n == 0:
        stats["empty"] += 1
        print(f"    [空] {job.label}: {job.src} 是空資料夾")
    return n


def run_copy(root, participants, mode, dry_run):
    jobs = build_jobs(root, participants, mode)
    manifest = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root), "mode": mode, "participants": participants,
        "created_files": [], "overwritten_files": [], "created_dirs": [], "missing_sources": [],
    }
    stats = {"copied": 0, "overwritten": 0, "missing": 0, "empty": 0}
    tag = "  (DRY-RUN,不會真的動檔案)" if dry_run else ""
    print(f"\n=== COPY logs  mode={mode}  participants={participants}{tag} ===")
    print(f"ROOT = {root}\n")

    for job in jobs:
        n = do_tree_job(job, manifest, dry_run, stats)
        if n:
            verb = "would copy" if dry_run else "copied"
            print(f"    [{n:>4}] {verb:>10}  {job.label}  ->  {job.dst}")

    print("\n--- 摘要 ---")
    print(f"  複製檔案(新建) : {stats['copied']}")
    print(f"  覆蓋既有檔案   : {stats['overwritten']}")
    print(f"  來源不存在     : {stats['missing']}")
    print(f"  來源為空       : {stats['empty']}")
    if manifest["missing_sources"]:
        print("\n--- 找不到的來源 ---")
        for m in manifest["missing_sources"]:
            print(f"  ! {m}")

    if dry_run:
        print("\n(DRY-RUN,沒有寫入任何檔案,也沒有存 manifest。)")
        return
    log_dir = root / "tests" / "exp1" / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    mpath = log_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S.json")
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已記錄本次複製:{mpath}")
    print("若複製錯了,執行:  python copy_stamp_logs.py undo")


def latest_manifest(root):
    log_dir = root / "tests" / "exp1" / LOG_DIRNAME
    if not log_dir.exists():
        return None
    runs = sorted(log_dir.glob("run_*.json"))
    return runs[-1] if runs else None


def run_undo(root, run_file, dry_run):
    mpath = Path(run_file) if run_file else latest_manifest(root)
    if not mpath or not mpath.exists():
        print("找不到任何 copy 紀錄,沒有東西可以 undo。")
        return
    manifest = json.loads(Path(mpath).read_text(encoding="utf-8"))
    created_files = manifest.get("created_files", [])
    created_dirs = manifest.get("created_dirs", [])
    overwritten = manifest.get("overwritten_files", [])
    tag = "  (DRY-RUN)" if dry_run else ""
    print(f"\n=== UNDO  使用紀錄:{mpath}{tag} ===")
    print(f"  將刪除新建檔案 : {len(created_files)}")
    if overwritten:
        print(f"  ⚠ {len(overwritten)} 個是覆蓋既有檔,undo 不刪")
    deleted = 0
    for f in created_files:
        p = Path(f)
        if p.exists():
            if not dry_run:
                p.unlink()
            deleted += 1
    for d in sorted(set(created_dirs), key=lambda s: s.count("/"), reverse=True):
        p = Path(d)
        if p.exists() and p.is_dir():
            try:
                if not any(p.iterdir()) and not dry_run:
                    p.rmdir()
            except OSError:
                pass
    print(f"\n  已{'(預計)' if dry_run else ''}刪除檔案 : {deleted}")
    if not dry_run:
        mpath.rename(mpath.with_suffix(".undone.json"))
        print("  此紀錄已標記為已撤銷。")


def list_runs(root):
    log_dir = root / "tests" / "exp1" / LOG_DIRNAME
    active = [p for p in log_dir.glob("run_*.json") if not p.name.endswith(".undone.json")] if log_dir.exists() else []
    undone = list(log_dir.glob("*.undone.json")) if log_dir.exists() else []
    runs = sorted(active) + sorted(undone)
    if not runs:
        print("還沒有任何 copy 紀錄。")
        return
    print(f"\n=== copy 紀錄({log_dir})===")
    for r in runs:
        m = json.loads(r.read_text(encoding="utf-8"))
        state = "已撤銷" if r.name.endswith(".undone.json") else "可 undo"
        print(f"  {r.name}  [{state}]  {m.get('timestamp')}  新建 {len(m.get('created_files', []))} 檔")


def main():
    ap = argparse.ArgumentParser(description="複製 STAMP logs 到 tests/exp")
    ap.add_argument("--root", default=str(ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("copy")
    pc.add_argument("--mode", choices=["all", "video", "image"], default="all")
    pc.add_argument("--participants", type=int, nargs="+", default=PARTICIPANTS)
    pc.add_argument("--dry-run", action="store_true")
    pu = sub.add_parser("undo")
    pu.add_argument("--run", default=None)
    pu.add_argument("--dry-run", action="store_true")
    sub.add_parser("list")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[錯誤] ROOT 不存在:{root}  請用 --root 指定。")
        sys.exit(1)
    if args.cmd == "copy":
        run_copy(root, args.participants, args.mode, args.dry_run)
    elif args.cmd == "undo":
        run_undo(root, args.run, args.dry_run)
    elif args.cmd == "list":
        list_runs(root)


if __name__ == "__main__":
    main()