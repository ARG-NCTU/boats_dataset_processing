#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""實驗三：Presence Score 路由有效性分析（calibration）。

對「STAMP 純自動匯出」與 GT 做逐偵測比對，產生 (presence_score, IoU vs GT)
配對，輸出 calibration 所需的明細、分桶、總結與圖。

單位（--per-object）
    預設：每個 (幀, 偵測) 一筆 (score, IoU)。
    --per-object：每個 (影片, 象限, obj_id) 聚合成一筆——
        score = 該物件跨幀平均（= STAMP 物件清單路由所用的 avg_score）
        IoU   = 該物件跨幀平均 IoU
    影片模式下 presence score 每物件一個，逐幀只是同分數重複；per-object 才是
    「校準系統實際路由的分數」，且 CI 不會被相關幀灌水。

配對：遮罩 IoU 幾何最佳指派（與論文 greedy IoU 0.5 一致，不依賴 label 名稱）。
    融合塊 -> 配最接近的 GT（方法 A），另一艘 missed；誤判 -> 配不到 -> IoU=0。
IoU 計算重用 evaluate_video_gt.py 的函式，與實驗一定義相同。

路徑（可覆蓋）
    pred: <root>/tests/exp/exp3/Video<X>_Q<Y>/coco/annotations/instances_train2024.json
    GT  : <root>/tests/exp/GT/V<X>Q<Y>GT/annotations/instances_default.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from evaluate_video_gt import (  # noqa: E402
    _load_json,
    _parse_frame_index,
    _image_maps,
    _annotation_to_mask,
    _mask_iou,
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


def pred_path(exp3_dir, video, quartile):
    return exp3_dir / f"Video{video}_Q{quartile}" / "coco" / "annotations" / "instances_train2024.json"


def gt_path(gt_dir, video, quartile):
    return gt_dir / f"V{video}Q{quartile}GT" / "annotations" / "instances_default.json"


def gt_masks_by_frame(coco):
    images, anns_by_image = _image_maps(coco)
    out = {}
    for image_id, image in images.items():
        f = _parse_frame_index(image)
        w, h = int(image["width"]), int(image["height"])
        masks = [_annotation_to_mask(a, w, h) for a in anns_by_image.get(image_id, [])]
        out.setdefault(f, []).extend(masks)
    return out


def pred_items_by_frame(coco):
    images, anns_by_image = _image_maps(coco)
    out = {}
    for image_id, image in images.items():
        f = _parse_frame_index(image)
        w, h = int(image["width"]), int(image["height"])
        lst = out.setdefault(f, [])
        for a in anns_by_image.get(image_id, []):
            mask = _annotation_to_mask(a, w, h)
            score = a.get("score", None)
            score = float(score) if score not in (None, "") else None
            obj_id = a.get("obj_id", a.get("id", None))
            lst.append((mask, score, obj_id))
    return out


def _iou_any_shape(a, b):
    if a.shape != b.shape:
        if cv2 is None:
            raise RuntimeError("GT 與 pred 解析度不同，需要 opencv 對齊遮罩尺寸。")
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return _mask_iou(a, b)


def match_frame(gt_masks, pred_items):
    rows = []
    G, P = len(gt_masks), len(pred_items)
    if P == 0:
        return rows, G, G
    if G == 0:
        for _mask, score, obj_id in pred_items:
            rows.append((score, 0.0, "false_positive", obj_id))
        return rows, 0, 0

    iou = np.zeros((G, P), dtype=float)
    for gi, gm in enumerate(gt_masks):
        for pj, (pm, _s, _o) in enumerate(pred_items):
            iou[gi, pj] = _iou_any_shape(gm, pm)

    matched_pred = {}
    used_gt = set()
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(-iou)
        for gi, pj in zip(r, c):
            if iou[gi, pj] > 0:
                matched_pred[int(pj)] = float(iou[gi, pj])
                used_gt.add(int(gi))
    else:
        pairs = sorted(((iou[gi, pj], gi, pj) for gi in range(G) for pj in range(P)
                        if iou[gi, pj] > 0), reverse=True)
        ug, up = set(), set()
        for v, gi, pj in pairs:
            if gi in ug or pj in up:
                continue
            ug.add(gi); up.add(pj)
            matched_pred[pj] = float(v); used_gt.add(gi)

    for pj, (_pm, score, obj_id) in enumerate(pred_items):
        if pj in matched_pred:
            rows.append((score, matched_pred[pj], "matched", obj_id))
        else:
            rows.append((score, 0.0, "false_positive", obj_id))
    return rows, G, G - len(used_gt)


def aggregate_objects(detections):
    groups = defaultdict(list)
    for d in detections:
        groups[(d["video"], d["quartile"], d["obj_id"])].append(d)
    rows = []
    for (video, quartile, obj_id), items in groups.items():
        scores = [it["score"] for it in items if it["score"] != ""]
        ious = [it["iou"] for it in items]
        any_matched = any(it["status"] == "matched" for it in items)
        rows.append({
            "video": video, "quartile": quartile, "obj_id": obj_id,
            "n_frames": len(items),
            "score": round(float(np.mean(scores)), 6) if scores else "",
            "iou": round(float(np.mean(ious)), 6) if ious else 0.0,
            "status": "matched" if any_matched else "false_positive",
        })
    return rows


def _bin_stats(scores, ious, score_lo=None, score_hi=None):
    n = int(len(scores))
    if n == 0:
        return {"n": 0, "score_min": score_lo, "score_max": score_hi,
                "score_median": None, "mean_iou": None, "std_iou": None,
                "ci95": None, "error_rate": None, "low_sample": True}
    mean = float(np.mean(ious))
    std = float(np.std(ious, ddof=1)) if n > 1 else 0.0
    ci = 1.96 * std / np.sqrt(n) if n > 0 else 0.0
    err = float(np.mean(ious < 0.5))
    return {
        "n": n,
        "score_min": float(np.min(scores)) if score_lo is None else float(score_lo),
        "score_max": float(np.max(scores)) if score_hi is None else float(score_hi),
        "score_median": round(float(np.median(scores)), 4),
        "mean_iou": round(mean, 4), "std_iou": round(std, 4),
        "ci95": round(ci, 4), "error_rate": round(err, 4), "low_sample": n < 30,
    }


def quantile_bins(scores, ious, min_per_bin=30, max_bins=10):
    s = np.asarray(scores, dtype=float); v = np.asarray(ious, dtype=float)
    N = len(s)
    if N == 0:
        return []
    order = np.argsort(s, kind="mergesort")
    s, v = s[order], v[order]
    n_bins = max(1, min(max_bins, N // min_per_bin)) if N >= min_per_bin else 1
    bins = []
    for chunk in np.array_split(np.arange(N), n_bins):
        if len(chunk) == 0:
            continue
        bins.append(_bin_stats(s[chunk], v[chunk]))
    return bins


def fixed_bins(scores, ious, n=10):
    s = np.asarray(scores, dtype=float); v = np.asarray(ious, dtype=float)
    edges = np.linspace(0.0, 1.0, n + 1)
    bins = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        mask = (s >= lo) & (s <= hi) if i == n - 1 else (s >= lo) & (s < hi)
        bins.append(_bin_stats(s[mask], v[mask], score_lo=lo, score_hi=hi))
    return bins


def write_detections(path, detections):
    fields = ["video", "quartile", "frame", "obj_id", "score", "iou", "status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for d in detections:
            w.writerow(d)


def write_objects(path, objects):
    fields = ["video", "quartile", "obj_id", "n_frames", "score", "iou", "status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for o in objects:
            w.writerow(o)


def write_bins(path, bins):
    fields = ["bin", "n", "score_min", "score_max", "score_median",
              "mean_iou", "std_iou", "ci95", "error_rate", "low_sample"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, b in enumerate(bins, start=1):
            row = {"bin": i}; row.update(b); w.writerow(row)


def maybe_plot(quant_bins, out_png, unit):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    xs = [b["score_median"] for b in quant_bins if b["n"] > 0]
    ys = [b["mean_iou"] for b in quant_bins if b["n"] > 0]
    es = [b["ci95"] for b in quant_bins if b["n"] > 0]
    if not xs:
        return None
    plt.figure(figsize=(6, 4))
    plt.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4)
    plt.xlabel("Presence score (bin median)")
    plt.ylabel("Mean IoU vs GT")
    plt.title(f"Exp 3: Presence-score calibration ({unit})")
    plt.ylim(0, 1); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()
    return str(out_png)


def main():
    ap = argparse.ArgumentParser(description="實驗三：Presence Score 路由有效性 calibration。")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--exp3-dir", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--videos", nargs="+", default=VIDEOS_DEFAULT)
    ap.add_argument("--quartiles", type=int, nargs="+", default=QUARTILES_DEFAULT)
    ap.add_argument("--per-object", action="store_true")
    ap.add_argument("--min-per-bin", type=int, default=30)
    ap.add_argument("--max-bins", type=int, default=10)
    ap.add_argument("--exclude-fp", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

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
                why = []
                if not p.exists():
                    why.append(f"pred缺:{p}")
                if not g.exists():
                    why.append(f"GT缺:{g}")
                skipped.append(f"Video{video}_Q{q}  ({'; '.join(why)})")

    print(f"找到 {len(pairs)} 組可比對；略過 {len(skipped)} 組。")
    for line in skipped:
        print("  - 略過", line)
    if args.dry_run:
        for video, q, p, g in pairs:
            print(f"  使用 Video{video}_Q{q}\n      pred={p}\n      gt  ={g}")
        return 0
    if not pairs:
        print("沒有任何可比對的組合，結束。")
        return 1

    detections = []
    per_cond = {}
    total_gt = total_missed = total_matched = total_fp = 0

    for video, q, p, g in pairs:
        gtf = gt_masks_by_frame(_load_json(g))
        prf = pred_items_by_frame(_load_json(p))
        frames = sorted(set(gtf) | set(prf))
        c_gt = c_missed = c_matched = c_fp = 0
        c_ious = []
        for fr in frames:
            rows, n_gt, missed = match_frame(gtf.get(fr, []), prf.get(fr, []))
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
            "frames": len(frames), "gt_instances": c_gt, "matched": c_matched,
            "false_positive": c_fp, "missed_gt": c_missed,
            "fn_rate": round(c_missed / c_gt, 4) if c_gt else None,
            "mean_iou_matched": round(float(np.mean(c_ious)), 4) if c_ious else None,
        }
        total_gt += c_gt; total_missed += c_missed
        total_matched += c_matched; total_fp += c_fp

    unit = "object" if args.per_object else "detection"
    objects = aggregate_objects(detections) if args.per_object else []
    pool_rows = objects if args.per_object else detections
    if args.exclude_fp:
        pool = [(r["score"], r["iou"]) for r in pool_rows
                if r["score"] != "" and r["status"] == "matched"]
    else:
        pool = [(r["score"], r["iou"]) for r in pool_rows if r["score"] != ""]
    no_score = sum(1 for r in pool_rows if r["score"] == "")
    scores = [s for s, _ in pool]; ious = [v for _, v in pool]

    qbins = quantile_bins(scores, ious, min_per_bin=args.min_per_bin, max_bins=args.max_bins)
    fbins = fixed_bins(scores, ious, n=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_detections(out_dir / "exp3_detections.csv", detections)
    if args.per_object:
        write_objects(out_dir / "exp3_objects.csv", objects)
    write_bins(out_dir / "exp3_calibration_quantile.csv", qbins)
    write_bins(out_dir / "exp3_calibration_fixed10.csv", fbins)
    png = maybe_plot(qbins, out_dir / "exp3_calibration.png", unit)

    summary = {
        "calibration_unit": unit,
        "conditions_used": list(per_cond.keys()),
        "n_detections_total": len(detections),
        "n_objects_total": len(objects) if args.per_object else None,
        "n_in_calibration_pool": len(pool),
        "n_without_score": no_score,
        "calibration_excludes_fp": bool(args.exclude_fp),
        "total_gt_instances": total_gt, "total_matched": total_matched,
        "total_false_positive": total_fp, "total_missed_gt": total_missed,
        "overall_fn_rate": round(total_missed / total_gt, 4) if total_gt else None,
        "mean_iou_matched": round(
            float(np.mean([d["iou"] for d in detections if d["status"] == "matched"])), 4
        ) if total_matched else None,
        "min_per_bin": args.min_per_bin,
        "n_quantile_bins": len([b for b in qbins if b["n"] > 0]),
        "per_condition": per_cond,
    }
    (out_dir / "exp3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 實驗三完成 ===")
    print(f"calibration 單位：{unit}")
    print(json.dumps({k: summary[k] for k in (
        "n_detections_total", "n_objects_total", "n_in_calibration_pool",
        "total_matched", "total_false_positive", "total_missed_gt",
        "overall_fn_rate", "mean_iou_matched", "n_quantile_bins")},
        ensure_ascii=False, indent=2))
    if no_score:
        print(f"注意：{no_score} 個單位沒有 score（已排除於 calibration）。")
    print(f"\n輸出寫到：{out_dir}")
    print(f"  calibration 圖：{png}" if png else "  （未安裝 matplotlib，略過畫圖）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
