#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# extract_object.py

import os
import cv2
import json
import argparse
import sys
import glob
import numpy as np
from tqdm import tqdm

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]

def _find_image_by_basename(folder, basename):
    for ext in IMAGE_EXTS:
        p = os.path.join(folder, basename + ext)
        if os.path.isfile(p):
            return p
    return None

def _polygon_to_mask(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask

def extract_cutout(input_dir, output_dir, image_name, json_name, label, shape_type):
    os.makedirs(output_dir, exist_ok=True)

    # 推導路徑
    if image_name is None and json_name is None:
        print("[Error] 請至少提供 --image_name 或 --json_name（通常兩者同名）", file=sys.stderr)
        sys.exit(1)

    # 允許只給其中一個名稱，另一個自動補
    if json_name is None and image_name:
        json_name = image_name + ".json"
    if image_name is None and json_name:
        image_name = os.path.splitext(json_name)[0]

    # 找影像與 JSON
    image_path = _find_image_by_basename(input_dir, image_name)
    if image_path is None:
        print(f"[Error] 找不到影像：{image_name}.* 在 {input_dir}", file=sys.stderr)
        sys.exit(2)
    json_path = os.path.join(input_dir, json_name)
    if not os.path.isfile(json_path):
        print(f"[Error] 找不到 JSON：{json_path}", file=sys.stderr)
        sys.exit(3)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[Error] 無法讀取影像：{image_path}", file=sys.stderr)
        sys.exit(4)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    masks = []
    for shape in data.get("shapes", []):
        if shape_type != "any":
            if shape.get("shape_type", "polygon") != "polygon":
                continue
        if (label is None) or (shape.get("label") == label):
            pts = shape.get("points", [])
            if len(pts) >= 3:
                masks.append(_polygon_to_mask(img.shape, pts))

    if not masks:
        print("[Error] 在 JSON 內找不到符合條件的多邊形；請檢查 --label 或 shape_type。", file=sys.stderr)
        sys.exit(5)

    mask = np.clip(np.sum(masks, axis=0), 0, 255).astype(np.uint8)
    fg = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(fg)
    rgba = cv2.merge([b, g, r, mask])

    out_path = os.path.join(output_dir, "object_cutout2.png")
    ok = cv2.imwrite(out_path, rgba)
    if not ok:
        print(f"[Error] 寫檔失敗：{out_path}", file=sys.stderr)
        sys.exit(6)

    print(f"[OK] Saved cutout → {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="從 Labelme JSON + 影像抠出物件，輸出 RGBA PNG（帶透明度）。")
    parser.add_argument("--input_dir", type=str, default="sonar",
                        help="來源資料夾（含影像與對應的 Labelme JSON）")
    parser.add_argument("--output_dir", type=str, default="sonar/_artifacts",
                        help="輸出資料夾（會輸出 object_cutout.png）")
    parser.add_argument("--image_name", type=str, default=None,
                        help="來源影像『檔名不含副檔名』，例如 src01（會自動找 .png/.jpg...）")
    parser.add_argument("--json_name", type=str, default=None,
                        help="Labelme JSON 檔名（含 .json），例如 src01.json")
    parser.add_argument("--label", type=str, default=None,
                        help="只抠指定 label（可省略）")
    parser.add_argument("--shape_type", type=str, default="polygon",
                        choices=["polygon", "any"], help="只吃 polygon 或所有 shape")
    args = parser.parse_args()

    extract_cutout(args.input_dir, args.output_dir,
                   args.image_name, args.json_name,
                   args.label, args.shape_type)

if __name__ == "__main__":
    main()

# python3 extract_object.py \
#   --input_dir sonar/20251029062806 \
#   --output_dir sonar/_artifacts \
#   --image_name 20251029062806_669 \
#   --json_name 20251029062806_669.json \
#   --label boat