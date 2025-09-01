import os
import re
import cv2
import json
import argparse
from tqdm import tqdm
import numpy as np

# -----------------------------
# Utilities
# -----------------------------

def sort_files(file):
    return file.lower()

def expand_path(p):
    return os.path.expanduser(p)

def load_name_class_map(path="Boats1-22/name_cls.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_mask_ids(path="Boats1-22/mask_ids.json"):
    with open(path, "r") as f:
        return json.load(f)

def get_group_key(boats_dir, mask_ids_dict):
    """
    將資料夾名稱對應到 mask_ids.json / name_cls.json 的群組鍵：
    - 若 boats_dir 本身就是群組鍵 (e.g., 'boats1-13')：直接回傳
    - 否則嘗試從 'boatsXX' 擷取數字，1-13 -> 'boats1-13'，>=14 -> 'boats14-22'
    """
    if boats_dir in mask_ids_dict:
        return boats_dir
    m = re.search(r"boats(\d+)", boats_dir, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        return "boats1-13" if idx <= 13 else "boats14-22"
    raise KeyError(f"無法從資料夾名稱推斷群組鍵：{boats_dir}")

# -----------------------------
# Core
# -----------------------------

def merge_json_files(
    rgb_thermal='rgb',
    bbox_seg='bbox',
    boats_root_path='~/dataset_boat12/Images',
    boat_count_per_scene=3002,
    vis=False,
    vis_dir='Visualization',
    mask_ids_path="Boats1-22/mask_ids.json",
    name_cls_path="Boats1-22/name_cls.json",
    occlude_thresh=0.3  # 遮擋判定門檻（只在有 label.json 時使用）
):
    """
    合併 .main.json 與 seg mask 產生 COCO 格式；
    - 若無 .main.json：不做遮擋判斷，直接用 mask 產生 bbox/area。
    - 若有 .main.json：使用其中的 GT bbox 做遮擋判斷（小於 occlude_thresh 視為遮擋、不輸出該標註）。
    - 類別以 name_cls.json 映射後名稱為準，動態建立 categories。
    """
    boats_root_path = expand_path(boats_root_path)

    # 載入映射與 mask id
    mask_ids_all = load_mask_ids(mask_ids_path)
    name_cls_all = load_name_class_map(name_cls_path)

    images = []
    annotations = []

    image_id = 1
    annotation_id = 1

    # 映射後類別 → id
    cat_name_to_id = {}
    categories = []
    next_cat_id = 1

    # 遮擋統計（僅在有 GT 時才計數）
    obscured_num = {}

    # 列出所有 boats_* 目錄
    boats_dirs = sorted(
        [d for d in os.listdir(boats_root_path) if os.path.isdir(os.path.join(boats_root_path, d))],
        key=sort_files
    )

    for boats_dir in tqdm(boats_dirs, desc="Boats", leave=False):
        boats_dir_path = os.path.join(boats_root_path, boats_dir)

        # 對應群組鍵
        group_key = get_group_key(boats_dir, mask_ids_all)
        mask_ids = mask_ids_all[group_key]
        name_cls = name_cls_all[group_key]

        # 列出 Scene 目錄
        scene_dirs = sorted(
            [d for d in os.listdir(boats_dir_path) if os.path.isdir(os.path.join(boats_dir_path, d))],
            key=sort_files
        )

        for _, scene_dir in tqdm(list(enumerate(scene_dirs)), desc="Scenes", leave=False):
            # thermal 僅處理 Scene1*
            if rgb_thermal == 'thermal' and 'Scene1' not in scene_dir:
                continue

            scene_dir_path = os.path.join(boats_dir_path, scene_dir)

            # 影像索引從 2 到 boat_count_per_scene-1（沿用原設定）
            for i in tqdm(range(2, boat_count_per_scene - 1), desc="Images", leave=False):
                # 構造路徑
                if rgb_thermal == 'rgb':
                    image_path = os.path.join(scene_dir_path, f'{i}.png')
                    vis_name = f'{i}.png'
                elif rgb_thermal == 'thermal':
                    image_path = os.path.join(scene_dir_path, f'{i}_thermal.png')
                    vis_name = f'{i}_thermal.png'
                else:
                    raise ValueError("Invalid value for rgb_thermal")

                label_path = os.path.join(scene_dir_path, f'{i}.main.json')
                mask_path = os.path.join(scene_dir_path, f'{i}_seg.png')

                # 必要檔案：影像、mask
                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    continue

                # 讀影像
                image = cv2.imread(image_path)
                if image is None:
                    continue
                image_height, image_width = image.shape[:2]

                file_name = (
                    f'{boats_dir}_{scene_dir}_{i}.png'
                    if rgb_thermal == 'rgb'
                    else f'{boats_dir}_{scene_dir}_{i}_thermal.png'
                )

                images.append({
                    "id": image_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": file_name,
                })

                # 嘗試讀 label json；不存在就用空 objects（→ 不做遮擋判斷）
                has_label = os.path.exists(label_path)
                if has_label:
                    try:
                        with open(label_path, 'r') as f:
                            data = json.load(f)
                        objects = data.get('objects', [])
                    except Exception:
                        data = {}
                        objects = []
                        has_label = False
                else:
                    data = {}
                    objects = []

                # 讀 seg mask
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    image_id += 1
                    continue
                mask = np.array(mask_img)

                source_img = image.copy()

                # 逐一處理本群組所有「原始類別名稱」與其 mask_id
                for orig_class_name, m_id in mask_ids.items():
                    # 原始類別 → 映射後類別
                    mapped_class = name_cls.get(orig_class_name, orig_class_name)

                    # 類別 ID 建立
                    if mapped_class not in cat_name_to_id:
                        cat_name_to_id[mapped_class] = next_cat_id
                        categories.append({"id": next_cat_id, "name": mapped_class})
                        obscured_num[mapped_class] = 0
                        next_cat_id += 1
                    class_id = cat_name_to_id[mapped_class]

                    # 從 mask 抽取該物件
                    current_obj_mask = np.where(mask == m_id, 255, 0).astype(np.uint8)
                    contours, _ = cv2.findContours(current_obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    largest_contour = max(contours, key=cv2.contourArea)
                    current_obj_mask = np.zeros_like(current_obj_mask)
                    cv2.drawContours(current_obj_mask, [largest_contour], -1, 255, cv2.FILLED)

                    # segmentation（可選）
                    if bbox_seg == 'both':
                        polygon = largest_contour.flatten().tolist()
                        if len(polygon) % 2 == 1:
                            polygon = polygon[:-1]
                    else:
                        polygon = []

                    # 由 mask 算 bbox 與可見面積
                    x, y, w, h = cv2.boundingRect(current_obj_mask)
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    current_visible_bbox = [x, y, w, h]
                    current_visible_area = w * h

                    obscured = False
                    gt_bbox = None

                    # === 只有在有 label.json 時才進行遮擋判斷 ===
                    if has_label and bbox_seg in ['bbox', 'both']:
                        # 找對應原始類別的 GT bbox
                        obj = None
                        for tmp_obj in objects:
                            if tmp_obj.get('class') == orig_class_name:
                                obj = tmp_obj
                                break

                        if obj is not None:
                            point = obj.get('bounding_box', {})
                            x1, y1 = point.get("top_left", [x, y])
                            x2, y2 = point.get("bottom_right", [x + w, y + h])
                            gt_w, gt_h = x2 - x1, y2 - y1
                            gt_bbox = [int(x1), int(y1), int(gt_w), int(gt_h)]
                            gt_area = max(1, gt_w * gt_h)
                            if current_visible_area < occlude_thresh * gt_area:
                                obscured_num[mapped_class] += 1
                                obscured = True

                    # 視覺化
                    if vis:
                        # 藍框：GT（若存在且有 label）
                        if gt_bbox is not None:
                            cv2.rectangle(source_img, (gt_bbox[0], gt_bbox[1]),
                                          (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]),
                                          (255, 0, 0), 2)
                        # 粉紅：遮擋；紅色：可見
                        color = (255, 0, 255) if obscured else (0, 0, 255)
                        cv2.rectangle(source_img, (current_visible_bbox[0], current_visible_bbox[1]),
                                      (current_visible_bbox[0]+current_visible_bbox[2], current_visible_bbox[1]+current_visible_bbox[3]),
                                      color, 2)

                    # 遮擋則跳過輸出（僅在 has_label=True 的情況下可能為 True）
                    if obscured:
                        continue

                    anno = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "area": int(current_visible_area),
                        "iscrowd": 0,
                        "bbox": current_visible_bbox,
                    }
                    if bbox_seg == 'both':
                        anno["segmentation"] = [polygon] if polygon else [[]]

                    annotations.append(anno)
                    annotation_id += 1

                # 只有在 vis=True 才輸出可視化圖
                if vis:
                    out_dir = os.path.join(vis_dir, boats_dir, scene_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(out_dir, vis_name), source_img)

                image_id += 1

    # 粗估理論張數（避免用到已離開 scope 的變數）
    blur_level = 1  # 0: no blur, 1: light blur, 2: heavy blur
    try:
        if boats_dirs:
            sample_boats_dir_path = os.path.join(boats_root_path, boats_dirs[0])
            scenes_cnt = len([d for d in os.listdir(sample_boats_dir_path) if os.path.isdir(os.path.join(sample_boats_dir_path, d))])
        else:
            scenes_cnt = 0
        images_num_theory = len(boats_dirs) * (scenes_cnt // (blur_level + 1)) * (blur_level + 1) * (boat_count_per_scene - 1)
    except Exception:
        images_num_theory = 0

    total_ann = max(1, len(annotations))
    obscured_rate = {k: f"{int(round(v / total_ann, 2) * 100)}%" for k, v in obscured_num.items()}

    info = f"""
    Data processing completed.

    RGB or Thermal: {rgb_thermal}
    Bbox or Both: {bbox_seg}
    Boats root path: {boats_root_path}
    Boats dirs * Scenes * (Source + Blur) * Images ≈ {len(boats_dirs)} * ? * {blur_level + 1} * {boat_count_per_scene - 1} ≈ {images_num_theory}
    Actual number of images: {image_id - 1}
    Number of annotations: {len(annotations)}
    Number of obscured objects: {obscured_num}
    Obscured rate (Number of obscured objects / Number of annotations): {obscured_rate}


    """
    print(info)
    with open('process_info.txt', 'a' if os.path.exists('process_info.txt') else 'w') as f:
        f.write(info)

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories  # 映射後類別
    }

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Merge Unity boat dataset json + seg masks into COCO format.")

    # 單一模式
    parser.add_argument("--rgb_thermal", type=str, choices=["rgb", "thermal"], default="rgb",
                        help="Use RGB or thermal images when not using --both")
    parser.add_argument("--bbox_seg", type=str, choices=["bbox", "both"], default="bbox",
                        help="Annotation mode: bbox only or bbox+seg")
    parser.add_argument("--boats_root_path", type=str, default="Images", help="Root path to boats dataset")
    parser.add_argument("--boat_count_per_scene", type=int, default=2002, help="Images per scene")
    parser.add_argument("--vis", action="store_true", help="Enable visualization and save vis images")
    parser.add_argument("--vis_dir", type=str, default="Visualization", help="Directory for visualization images")
    parser.add_argument("--mask_ids_path", type=str, default="Boats1-22/mask_ids.json", help="Path to mask_ids.json")
    parser.add_argument("--name_cls_path", type=str, default="Boats1-22/name_cls.json", help="Path to name_cls.json")
    parser.add_argument("--occlude_thresh", type=float, default=0.3, help="Threshold for occlusion (used only if label.json exists)")

    # 輸出
    parser.add_argument("--output", type=str, help="Output COCO json file (single mode)")

    # 雙模式
    parser.add_argument("--both", action="store_true", help="Run both rgb and thermal in one go")
    parser.add_argument("--output_rgb", type=str, help="Output path for RGB when --both")
    parser.add_argument("--output_thermal", type=str, help="Output path for thermal when --both")
    parser.add_argument("--output_prefix", type=str, help="If --both and per-mode outputs not given, use this prefix to form files: {prefix}_rgb.json and {prefix}_thermal.json")

    args = parser.parse_args()

    # 參數檢查/推導
    if args.both:
        if not args.output_rgb or not args.output_thermal:
            if not args.output_prefix:
                raise SystemExit("當使用 --both 時，請提供 --output_rgb 與 --output_thermal，或提供 --output_prefix 讓我自動生成檔名。")
            if not args.output_rgb:
                args.output_rgb = f"{args.output_prefix}_rgb.json"
            if not args.output_thermal:
                args.output_thermal = f"{args.output_prefix}_thermal.json"
    else:
        if not args.output:
            raise SystemExit("未使用 --both 時，請提供 --output。")

    return args

def main():
    args = parse_args()

    coco_data = merge_json_files(
        rgb_thermal=args.rgb_thermal,
        bbox_seg=args.bbox_seg,
        boats_root_path=args.boats_root_path,
        boat_count_per_scene=args.boat_count_per_scene,
        vis=args.vis,
        vis_dir=args.vis_dir,
        mask_ids_path=args.mask_ids_path,
        name_cls_path=args.name_cls_path,
        occlude_thresh=args.occlude_thresh
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"✅ Saved merged COCO json to {args.output}")

if __name__ == "__main__":
    main()


# Usage examples:

# RGB only:
"""
python3 merge_json_unity.py \
  --rgb_thermal rgb \
  --boats_root_path Images \
  --boat_count_per_scene 2002 \
  --output Boats1-22/coco_formatted_unity_rgb_data.json
"""


# Thermal only:
"""
python3 merge_json_unity.py \
  --rgb_thermal thermal \
  --boats_root_path Images \
  --boat_count_per_scene 2002 \
  --output Boats1-22/coco_formatted_unity_thermal_data.json
"""
