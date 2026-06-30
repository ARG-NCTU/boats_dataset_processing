#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""實驗三：Presence Score 路由有效性分析（calibration）。

輸出（寫到 <exp3-dir>/_results/）
  exp3_detections.csv         逐 (幀, 偵測) 的 (score, iou, status, obj_id)
  exp3_objects.csv            （--per-object）每物件一筆 (avg_score, mean_iou)
  exp3_confidence_bands.csv   LOW / UNCERTAIN / HIGH 三組的平均 IoU、error rate
  exp3_summary.json           Spearman ρ/p、漏標(FN)率、FP、mean IoU 等
  exp3_scatter.png            散點圖（每單位一點）＋三組平均＋ρ（需 matplotlib）

回答 RQ3 的四樣證據：散點圖、Spearman ρ/p、LOW/UNC/HIGH 三組平均、漏標率。

單位
  預設：每 (幀, 偵測) 一筆。
  --per-object：每 (影片, 象限, obj_id) 聚合（score=跨幀平均=路由用的 avg_score，
                iou=跨幀平均）。影片建議用這個。

配對：遮罩 IoU 幾何最佳指派（與論文 greedy IoU 0.5 一致，不看 label 名稱）。
  融合塊->配最接近的 GT（方法 A），另一艘 missed；誤判/非船->配不到->IoU=0(FP)。

路徑
  pred: <root>/tests/exp/exp3/Video<X>_Q<Y>/coco/annotations/instances_train2024.json
  GT  : <root>/tests/exp/GT/V<X>Q<Y>GT/annotations/instances_default.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from evaluate_video_gt import (  # noqa: E402
    _load_json, _parse_frame_index, _image_maps, _annotation_to_mask, _mask_iou,
)

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

VIDEOS_DEFAULT = ["A", "B", "C", "D"]
QUARTILES_DEFAULT = [1, 2, 3, 4]


def pred_path(exp3_dir, video, q):
    return exp3_dir / f"Video{video}_Q{q}" / "coco" / "annotations" / "instances_train2024.json"


def gt_path(gt_dir, video, q):
    return gt_dir / f"V{video}Q{q}GT" / "annotations" / "instances_default.json"


def gt_masks_by_frame(coco):
    images, anns = _image_maps(coco)
    out = {}
    for image_id, image in images.items():
        f = _parse_frame_index(image)
        w, h = int(image["width"]), int(image["height"])
        out.setdefault(f, []).extend(_annotation_to_mask(a, w, h) for a in anns.get(image_id, []))
    return out


def pred_items_by_frame(coco):
    images, anns = _image_maps(coco)
    out = {}
    for image_id, image in images.items():
        f = _parse_frame_index(image)
        w, h = int(image["width"]), int(image["height"])
        lst = out.setdefault(f, [])
        for a in anns.get(image_id, []):
            mask = _annotation_to_mask(a, w, h)
            score = a.get("score", None)
            score = float(score) if score not in (None, "") else None
            obj_id = a.get("obj_id", a.get("id", None))
            lst.append((mask, score, obj_id))
    return out


def _iou_any_shape(a, b):
    if a.shape != b.shape:
        if cv2 is None:
            raise RuntimeError("GT 與 pred 解析度不同，需要 opencv 對齊尺寸。")
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return _mask_iou(a, b)


def match_frame(gt_masks, pred_items):
    rows = []
    G, P = len(gt_masks), len(pred_items)
    if P == 0:
        return rows, G, G
    if G == 0:
        for _m, score, obj_id in pred_items:
            rows.append((score, 0.0, "false_positive", obj_id))
        return rows, 0, 0
    iou = np.zeros((G, P))
    for gi, gm in enumerate(gt_masks):
        for pj, (pm, _s, _o) in enumerate(pred_items):
            iou[gi, pj] = _iou_any_shape(gm, pm)
    matched, used_gt = {}, set()
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(-iou)
        for gi, pj in zip(r, c):
            if iou[gi, pj] > 0:
                matched[int(pj)] = float(iou[gi, pj]); used_gt.add(int(gi))
    else:
        pairs = sorted(((iou[gi, pj], gi, pj) for gi in range(G) for pj in range(P)
                        if iou[gi, pj] > 0), reverse=True)
        ug, up = set(), set()
        for v, gi, pj in pairs:
            if gi in ug or pj in up:
                continue
            ug.add(gi); up.add(pj); matched[pj] = float(v); used_gt.add(gi)
    for pj, (_pm, score, obj_id) in enumerate(pred_items):
        rows.append((score, matched.get(pj, 0.0),
                     "matched" if pj in matched else "false_positive", obj_id))
    return rows, G, G - len(used_gt)


def aggregate_objects(detections):
    groups = defaultdict(list)
    for d in detections:
        groups[(d["video"], d["quartile"], d["obj_id"])].append(d)
    rows = []
    for (video, quartile, obj_id), items in groups.items():
        scores = [it["score"] for it in items if it["score"] != ""]
        ious = [it["iou"] for it in items]
        rows.append({
            "video": video, "quartile": quartile, "obj_id": obj_id, "n_frames": len(items),
            "score": round(float(np.mean(scores)), 6) if scores else "",
            "iou": round(float(np.mean(ious)), 6) if ious else 0.0,
            "status": "matched" if any(it["status"] == "matched" for it in items) else "false_positive",
        })
    return rows


def spearman(scores, ious):
    if len(scores) < 3:
        return None, None
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(scores, ious)
        return (None if np.isnan(rho) else float(rho)), (None if np.isnan(p) else float(p))
    except Exception:
        rx, ry = _rank(scores), _rank(ious)
        if np.std(rx) == 0 or np.std(ry) == 0:
            return None, None
        return float(np.corrcoef(rx, ry)[0, 1]), None


def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def confidence_bands(pool, low, high):
    buckets = {f"LOW (<{low})": [], f"UNCERTAIN ({low}-{high})": [], f"HIGH (>={high})": []}
    names = list(buckets)
    for s, v in pool:
        buckets[names[0 if s < low else (2 if s >= high else 1)]].append(v)
    out = []
    for name in names:
        vs = np.asarray(buckets[name], float)
        n = len(vs)
        out.append({
            "band": name, "n": n,
            "mean_iou": round(float(vs.mean()), 4) if n else None,
            "std_iou": round(float(vs.std(ddof=1)), 4) if n > 1 else (0.0 if n == 1 else None),
            "error_rate": round(float((vs < 0.5).mean()), 4) if n else None,
        })
    return out


def write_rows(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_main(pool_rows, bands, out_png, rho, p, unit, low, high):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    mx = [r["score"] for r in pool_rows if r["score"] != "" and r["status"] == "matched"]
    my = [r["iou"] for r in pool_rows if r["score"] != "" and r["status"] == "matched"]
    fx = [r["score"] for r in pool_rows if r["score"] != "" and r["status"] == "false_positive"]
    fy = [r["iou"] for r in pool_rows if r["score"] != "" and r["status"] == "false_positive"]
    plt.figure(figsize=(6.5, 4.5))
    if mx:
        plt.scatter(mx, my, s=18, alpha=0.5, label="matched")
    if fx:
        plt.scatter(fx, fy, s=22, alpha=0.6, marker="x", color="red", label="false positive")
    # 三組平均
    centers = {f"LOW (<{low})": low / 2, f"UNCERTAIN ({low}-{high})": (low + high) / 2,
               f"HIGH (>={high})": (high + 1) / 2}
    bx = [centers[b["band"]] for b in bands if b["n"] > 0]
    by = [b["mean_iou"] for b in bands if b["n"] > 0]
    if bx:
        plt.plot(bx, by, "s-", color="black", ms=10, label="band mean IoU")
    plt.axvline(low, ls="--", color="gray", lw=1)
    plt.axvline(high, ls="--", color="gray", lw=1)
    title = f"Exp3 calibration ({unit})"
    if rho is not None:
        title += f"  Spearman ρ={rho:.3f}" + (f", p={p:.3g}" if p is not None else "")
    plt.title(title)
    plt.xlabel("Presence score"); plt.ylabel("IoU vs GT")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return str(out_png)


def load_ignore(path):
    """讀忽略清單 CSV(欄位:video,quartile,frame,obj_id),回傳 set。"""
    if not path:
        return set()
    out = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.add((str(row.get("video", "")).strip(),
                     str(row.get("quartile", "")).strip(),
                     str(row.get("frame", "")).strip(),
                     str(row.get("obj_id", "")).strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description="實驗三 calibration（散點+Spearman+三組+漏標）。")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--exp3-dir", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--videos", nargs="+", default=VIDEOS_DEFAULT)
    ap.add_argument("--quartiles", type=int, nargs="+", default=QUARTILES_DEFAULT)
    ap.add_argument("--per-object", action="store_true")
    ap.add_argument("--low-threshold", type=float, default=0.5)
    ap.add_argument("--high-threshold", type=float, default=0.8)
    ap.add_argument("--exclude-fp", action="store_true")
    ap.add_argument("--ignore-list", default=None,
                    help="CSV(欄位 video,quartile,frame,obj_id):要忽略的偵測,等同 GUI ignore。")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ignore_set = load_ignore(args.ignore_list)

    root = Path(args.root)
    exp3_dir = Path(args.exp3_dir) if args.exp3_dir else root / "tests" / "exp" / "exp3"
    gt_dir = Path(args.gt_dir) if args.gt_dir else root / "tests" / "exp" / "GT"
    out_dir = Path(args.output_dir) if args.output_dir else exp3_dir / "_results"

    pairs, skipped = [], []
    for video in args.videos:
        for q in args.quartiles:
            p, g = pred_path(exp3_dir, video, q), gt_path(gt_dir, video, q)
            if p.exists() and g.exists():
                pairs.append((video, q, p, g))
            else:
                miss = []
                if not p.exists():
                    miss.append("pred")
                if not g.exists():
                    miss.append("GT")
                skipped.append(f"Video{video}_Q{q} (缺 {'+'.join(miss)})")

    print(f"找到 {len(pairs)} 組可比對；略過 {len(skipped)} 組。")
    for line in skipped:
        print("  - 略過", line)
    if args.dry_run:
        for video, q, p, g in pairs:
            print(f"  使用 Video{video}_Q{q}")
        return 0
    if not pairs:
        print("沒有可比對的組合。"); return 1

    detections = []
    per_cond = {}
    total_gt = total_missed = total_matched = total_fp = 0
    for video, q, p, g in pairs:
        gtf = gt_masks_by_frame(_load_json(g))
        prf = pred_items_by_frame(_load_json(p))
        c_gt = c_missed = c_matched = c_fp = 0
        c_ious = []
        for fr in sorted(set(gtf) | set(prf)):
            items = prf.get(fr, [])
            if ignore_set:
                items = [it for it in items
                         if (video, str(q), str(fr), str(it[2])) not in ignore_set]
            rows, n_gt, missed = match_frame(gtf.get(fr, []), items)
            c_gt += n_gt; c_missed += missed
            for score, iou, status, obj_id in rows:
                detections.append({
                    "video": video, "quartile": q, "frame": fr,
                    "obj_id": "" if obj_id is None else obj_id,
                    "score": "" if score is None else round(score, 6),
                    "iou": round(iou, 6), "status": status,
                })
                if status == "matched":
                    c_matched += 1; c_ious.append(iou)
                else:
                    c_fp += 1
        per_cond[f"Video{video}_Q{q}"] = {
            "gt_instances": c_gt, "matched": c_matched, "false_positive": c_fp,
            "missed_gt": c_missed, "fn_rate": round(c_missed / c_gt, 4) if c_gt else None,
            "mean_iou_matched": round(float(np.mean(c_ious)), 4) if c_ious else None,
        }
        total_gt += c_gt; total_missed += c_missed; total_matched += c_matched; total_fp += c_fp

    unit = "object" if args.per_object else "detection"
    objects = aggregate_objects(detections) if args.per_object else []
    pool_rows = objects if args.per_object else detections
    if args.exclude_fp:
        usable = [r for r in pool_rows if r["score"] != "" and r["status"] == "matched"]
    else:
        usable = [r for r in pool_rows if r["score"] != ""]
    pool = [(r["score"], r["iou"]) for r in usable]
    scores = [s for s, _ in pool]; ious = [v for _, v in pool]

    rho, pval = spearman(scores, ious)
    bands = confidence_bands(pool, args.low_threshold, args.high_threshold)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "exp3_detections.csv",
               ["video", "quartile", "frame", "obj_id", "score", "iou", "status"], detections)
    if args.per_object:
        write_rows(out_dir / "exp3_objects.csv",
                   ["video", "quartile", "obj_id", "n_frames", "score", "iou", "status"], objects)
    write_rows(out_dir / "exp3_confidence_bands.csv",
               ["band", "n", "mean_iou", "std_iou", "error_rate"], bands)
    png = plot_main(usable, bands, out_dir / "exp3_scatter.png", rho, pval, unit,
                    args.low_threshold, args.high_threshold)

    summary = {
        "calibration_unit": unit,
        "conditions_used": list(per_cond.keys()),
        "n_detections_total": len(detections),
        "n_objects_total": len(objects) if args.per_object else None,
        "n_in_calibration_pool": len(pool),
        "spearman_rho": None if rho is None else round(rho, 4),
        "spearman_p": None if pval is None else float(f"{pval:.3g}"),
        "confidence_bands": bands,
        "total_gt_instances": total_gt, "total_matched": total_matched,
        "total_false_positive": total_fp, "total_missed_gt": total_missed,
        "overall_fn_rate": round(total_missed / total_gt, 4) if total_gt else None,
        "mean_iou_matched": round(
            float(np.mean([d["iou"] for d in detections if d["status"] == "matched"])), 4
        ) if total_matched else None,
        "low_threshold": args.low_threshold, "high_threshold": args.high_threshold,
        "per_condition": per_cond,
    }
    (out_dir / "exp3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 實驗三完成 ===  單位：" + unit)
    print(json.dumps({k: summary[k] for k in (
        "n_in_calibration_pool", "spearman_rho", "spearman_p",
        "total_false_positive", "total_missed_gt", "overall_fn_rate", "mean_iou_matched")},
        ensure_ascii=False, indent=2))
    print("三組平均 IoU：")
    for b in bands:
        print(f"  {b['band']}: n={b['n']}, mean_iou={b['mean_iou']}, error_rate={b['error_rate']}")
    print(f"\n輸出：{out_dir}")
    print(f"  散點圖：{png}" if png else "  （未裝 matplotlib，無圖；CSV 仍可自行作圖）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
