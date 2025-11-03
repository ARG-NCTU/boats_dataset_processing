#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# paste_and_label.py - 針對聲納圖像的物件貼上與自動標註

import os
import cv2
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import glob

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]

def parse_region(region_str):
    """解析區域字串，格式：x1,y1,x2,y2"""
    if region_str is None:
        return None
    parts = [int(x.strip()) for x in region_str.split(',')]
    if len(parts) != 4:
        raise ValueError("區域格式錯誤，應為：x1,y1,x2,y2")
    return tuple(parts)

def read_original_json(json_path):
    """讀取原始Labelme JSON，獲取標籤資訊"""
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 獲取第一個shape的標籤
    if data.get('shapes') and len(data['shapes']) > 0:
        label = data['shapes'][0].get('label', 'object')
        print(f"從原始JSON讀取標籤：{label}")
        return {
            'label': label,
            'version': data.get('version', '5.0.1')
        }
    return None

def get_polygon_from_mask(mask, simplify=True):
    """從mask提取polygon輪廓"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    if simplify:
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        polygon = approx.reshape(-1, 2).tolist()
    else:
        polygon = largest_contour.reshape(-1, 2).tolist()
    
    return polygon

def random_transform_object(obj_img, max_rotation=360, scale_range=(0.8, 1.2), 
                            flip_h=True, flip_v=True):
    """隨機變換物件"""
    h, w = obj_img.shape[:2]
    
    # 隨機縮放
    scale = random.uniform(*scale_range)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w > 0 and new_h > 0:
        obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 隨機翻轉
    if flip_h and random.random() > 0.5:
        obj_img = cv2.flip(obj_img, 1)
    if flip_v and random.random() > 0.5:
        obj_img = cv2.flip(obj_img, 0)
    
    # 隨機旋轉
    angle = random.uniform(0, max_rotation)
    h, w = obj_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    obj_img = cv2.warpAffine(obj_img, M, (new_w, new_h), 
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    
    return obj_img

def paste_object_in_region(bg_img, obj_img, region=None, margin=10, debug=False):
    """
    將物件貼到背景圖的指定區域內
    region: (x1, y1, x2, y2) 定義可貼上的區域
    debug: 是否顯示調試資訊（繪製邊界框）
    """
    bg_h, bg_w = bg_img.shape[:2]
    obj_h, obj_w = obj_img.shape[:2]
    
    # 確定可貼上的區域
    if region is None:
        region = (0, 0, bg_w, bg_h)
    
    x1, y1, x2, y2 = region
    region_w = x2 - x1
    region_h = y2 - y1
    
    print(f"  物件原始尺寸：{obj_w}x{obj_h}")
    print(f"  可用區域：{region_w}x{region_h}")
    
    # 確保物件不會超出區域邊界
    max_x = x2 - obj_w - margin
    max_y = y2 - obj_h - margin
    
    if max_x <= x1 + margin or max_y <= y1 + margin:
        # 物件太大，縮小以適應區域
        scale = min((region_w - 2*margin) / obj_w, (region_h - 2*margin) / obj_h) * 0.85
        if scale < 0.1:
            scale = 0.1
        new_w, new_h = int(obj_w * scale), int(obj_h * scale)
        print(f"  物件太大，縮放至：{new_w}x{new_h} (scale={scale:.2f})")
        obj_img = cv2.resize(obj_img, (new_w, new_h))
        obj_h, obj_w = new_h, new_w
        max_x = x2 - obj_w - margin
        max_y = y2 - obj_h - margin
    
    # 在區域內隨機位置
    x = random.randint(x1 + margin, max(x1 + margin, max_x))
    y = random.randint(y1 + margin, max(y1 + margin, max_y))
    
    print(f"  貼上位置：({x}, {y})")
    
    # 提取alpha通道
    if obj_img.shape[2] == 4:
        obj_rgb = obj_img[:, :, :3]
        alpha = obj_img[:, :, 3] / 255.0
        alpha_mask_binary = (obj_img[:, :, 3] > 10).astype(np.uint8) * 255
    else:
        obj_rgb = obj_img
        gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        alpha = (gray > 10).astype(np.float32)
        alpha_mask_binary = (gray > 10).astype(np.uint8) * 255
    
    # 檢查alpha通道
    non_zero_pixels = np.count_nonzero(alpha > 0.1)
    print(f"  物件非透明像素數：{non_zero_pixels}")
    
    if non_zero_pixels == 0:
        print("  [警告] 物件完全透明！")
        return bg_img, [], (x, y, obj_w, obj_h)
    
    # 貼上物件 - 使用更明顯的混合方式
    result = bg_img.copy()
    
    # 確保不會超出邊界
    paste_h = min(obj_h, bg_h - y)
    paste_w = min(obj_w, bg_w - x)
    
    if paste_h <= 0 or paste_w <= 0:
        print("  [警告] 貼上區域無效！")
        return bg_img, [], (x, y, obj_w, obj_h)
    
    obj_rgb_crop = obj_rgb[:paste_h, :paste_w]
    alpha_crop = alpha[:paste_h, :paste_w]
    alpha_mask_crop = alpha_mask_binary[:paste_h, :paste_w]
    
    alpha_3ch = np.stack([alpha_crop] * 3, axis=2)
    
    roi = result[y:y+paste_h, x:x+paste_w]
    
    # 使用alpha混合
    blended = (alpha_3ch * obj_rgb_crop + (1 - alpha_3ch) * roi).astype(np.uint8)
    result[y:y+paste_h, x:x+paste_w] = blended
    
    # Debug模式：繪製邊界框
    if debug:
        cv2.rectangle(result, (x, y), (x+paste_w, y+paste_h), (0, 255, 0), 2)
        cv2.putText(result, f"Object", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 計算polygon
    polygon = get_polygon_from_mask(alpha_mask_crop)
    polygon = [[p[0] + x, p[1] + y] for p in polygon]
    
    print(f"  Polygon點數：{len(polygon)}")
    
    return result, polygon, (x, y, paste_w, paste_h)

def save_labelme_json(image_path, polygons, labels, output_path, version="5.0.1"):
    """保存Labelme格式JSON"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    shapes = []
    for polygon, label in zip(polygons, labels):
        shapes.append({
            "label": label,
            "points": polygon,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        })
    
    data = {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_yolo_txt(image_shape, polygons, class_id, output_path):
    """保存YOLO格式標註"""
    h, w = image_shape[:2]
    
    with open(output_path, 'w') as f:
        for polygon in polygons:
            normalized = []
            for x, y in polygon:
                normalized.append(f"{x/w:.6f}")
                normalized.append(f"{y/h:.6f}")
            
            line = f"{class_id} " + " ".join(normalized) + "\n"
            f.write(line)

def save_coco_json(images_info, output_path, category_name="object"):
    """保存COCO格式JSON"""
    categories = [{"id": 1, "name": category_name, "supercategory": "none"}]
    
    images = []
    annotations = []
    ann_id = 1
    
    for img_id, info in enumerate(images_info, 1):
        images.append({
            "id": img_id,
            "file_name": os.path.basename(info["image_path"]),
            "width": info["width"],
            "height": info["height"]
        })
        
        for polygon in info["polygons"]:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            segmentation = []
            for p in polygon:
                segmentation.extend([p[0], p[1]])
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [segmentation],
                "area": (x_max - x_min) * (y_max - y_min),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "iscrowd": 0
            })
            ann_id += 1
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)

def visualize_region(image_path, region, output_path):
    """視覺化顯示貼上區域"""
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = region
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img, f"Paste Region: {x2-x1}x{y2-y1}", (x1+10, y1+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)
    print(f"區域視覺化已保存至：{output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="將物件隨機貼到聲納圖片的指定區域（如HF區域），並自動生成標註")
    parser.add_argument("--object_png", type=str, required=True,
                        help="物件PNG檔路徑（RGBA，透明背景）")
    parser.add_argument("--original_json", type=str, default=None,
                        help="原始Labelme JSON檔路徑（用於讀取標籤名稱）")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="目標圖片資料夾")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="輸出資料夾")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="要生成的樣本數量")
    parser.add_argument("--label", type=str, default=None,
                        help="物件標籤名稱（若不指定則從JSON讀取）")
    parser.add_argument("--format", type=str, default="labelme",
                        choices=["labelme", "yolo", "coco", "all"],
                        help="標註格式")
    
    # 區域設定
    parser.add_argument("--region", type=str, default=None,
                        help="限制貼上區域，格式：x1,y1,x2,y2（像素坐標）。例如：40,105,570,255")
    parser.add_argument("--region_percent", type=str, default=None,
                        help="限制貼上區域（百分比），格式：x1,y1,x2,y2（0-100）")
    parser.add_argument("--visualize_region", action="store_true",
                        help="視覺化顯示貼上區域")
    
    # 變換參數
    parser.add_argument("--max_rotation", type=float, default=30,
                        help="最大旋轉角度（預設30度適合聲納圖）")
    parser.add_argument("--scale_min", type=float, default=0.9,
                        help="最小縮放比例")
    parser.add_argument("--scale_max", type=float, default=1.1,
                        help="最大縮放比例")
    parser.add_argument("--no_flip", action="store_true",
                        help="不使用翻轉")
    parser.add_argument("--seed", type=int, default=None,
                        help="隨機種子")
    parser.add_argument("--debug", action="store_true",
                        help="Debug模式：在圖片上顯示貼上位置")
    parser.add_argument("--copy_remaining", action="store_true",
                        help="將沒有被使用的圖片也複製到輸出資料夾（作為負樣本）")
    parser.add_argument("--create_empty_labels", action="store_true",
                        help="為負樣本創建空標註檔案（配合copy_remaining使用）")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # 讀取物件
    print(f"\n正在讀取物件PNG：{args.object_png}")
    obj_img = cv2.imread(args.object_png, cv2.IMREAD_UNCHANGED)
    if obj_img is None:
        print(f"[Error] 無法讀取物件圖片：{args.object_png}")
        return
    
    print(f"原始圖片尺寸：{obj_img.shape[1]}x{obj_img.shape[0]}")
    print(f"物件通道數：{obj_img.shape[2]}")
    
    # 裁切透明區域，只保留物件本體
    if obj_img.shape[2] == 4:
        alpha = obj_img[:, :, 3]
        coords = cv2.findNonZero(alpha)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            obj_img = obj_img[y:y+h, x:x+w]
            print(f"裁切後物件尺寸：{obj_img.shape[1]}x{obj_img.shape[0]} (移除了透明邊緣)")
        else:
            print("[Warning] 物件完全透明！")
    else:
        print("[Warning] 物件沒有alpha通道，無法自動裁切")
    
    # 讀取原始JSON獲取標籤
    label_info = None
    if args.original_json:
        label_info = read_original_json(args.original_json)
    
    # 決定使用的標籤名稱
    if args.label:
        label_name = args.label
    elif label_info:
        label_name = label_info['label']
    else:
        label_name = "object"
    
    print(f"使用標籤名稱：{label_name}")
    
    # 找出所有目標圖片
    images = []
    for ext in IMAGE_EXTS:
        images.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
    
    if not images:
        print(f"[Error] 在 {args.image_dir} 找不到圖片")
        return
    
    print(f"找到 {len(images)} 張圖片")
    
    # 確定貼上區域
    region = None
    sample_img = cv2.imread(images[0])
    img_h, img_w = sample_img.shape[:2]
    print(f"圖片尺寸：{img_w}x{img_h}")
    
    if args.region:
        region = parse_region(args.region)
        print(f"使用指定區域：{region}")
    elif args.region_percent:
        x1, y1, x2, y2 = [float(x.strip()) for x in args.region_percent.split(',')]
        region = (
            int(img_w * x1 / 100),
            int(img_h * y1 / 100),
            int(img_w * x2 / 100),
            int(img_h * y2 / 100)
        )
        print(f"使用百分比區域：{args.region_percent} -> {region}")
    else:
        region = (0, 0, img_w, img_h)
        print(f"使用整張圖片：{region}")
    
    # 視覺化區域
    if args.visualize_region:
        os.makedirs(args.output_dir, exist_ok=True)
        visualize_region(images[0], region, 
                        os.path.join(args.output_dir, "region_visualization.png"))
    
    # 創建輸出資料夾
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels"), exist_ok=True)
    
    # 生成樣本
    coco_images_info = []
    success_count = 0
    used_images = set()  # 記錄已使用的圖片
    
    print(f"\n開始生成 {args.num_samples} 個樣本...\n")
    
    for i in tqdm(range(args.num_samples), desc="生成樣本"):
        # 隨機選擇背景圖
        bg_path = random.choice(images)
        used_images.add(bg_path)  # 記錄使用過的圖片
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue
        
        try:
            print(f"\n樣本 {i}:")
            
            # 變換物件
            transformed_obj = random_transform_object(
                obj_img.copy(),
                max_rotation=args.max_rotation,
                scale_range=(args.scale_min, args.scale_max),
                flip_h=not args.no_flip,
                flip_v=not args.no_flip
            )
            
            # 貼上物件到指定區域
            result_img, polygon, bbox = paste_object_in_region(
                bg_img, transformed_obj, region=region, debug=args.debug)
            
            if len(polygon) == 0:
                print("  [跳過] 未能生成有效polygon")
                continue
            
            # 保存圖片
            output_img_name = f"augmented_{i:04d}.png"
            output_img_path = os.path.join(args.output_dir, "images", output_img_name)
            cv2.imwrite(output_img_path, result_img)
            
            # 保存標註
            version = label_info['version'] if label_info else "5.0.1"
            
            if args.format in ["labelme", "all"]:
                json_path = os.path.join(args.output_dir, "labels", 
                                        f"augmented_{i:04d}.json")
                save_labelme_json(output_img_path, [polygon], [label_name], 
                                json_path, version=version)
            
            if args.format in ["yolo", "all"]:
                txt_path = os.path.join(args.output_dir, "labels", 
                                       f"augmented_{i:04d}.txt")
                save_yolo_txt(result_img.shape, [polygon], 0, txt_path)
            
            if args.format in ["coco", "all"]:
                coco_images_info.append({
                    "image_path": output_img_path,
                    "width": result_img.shape[1],
                    "height": result_img.shape[0],
                    "polygons": [polygon]
                })
            
            success_count += 1
            
        except Exception as e:
            print(f"\n[Warning] 樣本 {i} 生成失敗：{e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存COCO格式
    if args.format in ["coco", "all"] and coco_images_info:
        coco_path = os.path.join(args.output_dir, "annotations.json")
        save_coco_json(coco_images_info, coco_path, category_name=label_name)
    
    # 複製剩餘未使用的圖片（負樣本）
    negative_count = 0
    if args.copy_remaining:
        print(f"\n正在複製剩餘的 {len(images) - len(used_images)} 張圖片...")
        
        remaining_images = [img for img in images if img not in used_images]
        
        for idx, img_path in enumerate(tqdm(remaining_images, desc="複製負樣本")):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # 保存圖片
                output_img_name = f"negative_{idx:04d}{os.path.splitext(img_path)[1]}"
                output_img_path = os.path.join(args.output_dir, "images", output_img_name)
                cv2.imwrite(output_img_path, img)
                
                # 創建空標註檔案（如果需要）
                if args.create_empty_labels:
                    if args.format in ["labelme", "all"]:
                        json_path = os.path.join(args.output_dir, "labels", 
                                                f"negative_{idx:04d}.json")
                        save_labelme_json(output_img_path, [], [], json_path, 
                                        version=label_info['version'] if label_info else "5.0.1")
                    
                    if args.format in ["yolo", "all"]:
                        txt_path = os.path.join(args.output_dir, "labels", 
                                               f"negative_{idx:04d}.txt")
                        # YOLO格式：空檔案表示沒有物件
                        open(txt_path, 'w').close()
                    
                    if args.format in ["coco", "all"]:
                        # COCO會在最後統一處理
                        coco_images_info.append({
                            "image_path": output_img_path,
                            "width": img.shape[1],
                            "height": img.shape[0],
                            "polygons": []  # 空列表表示沒有物件
                        })
                
                negative_count += 1
                
            except Exception as e:
                print(f"\n[Warning] 複製 {img_path} 失敗：{e}")
                continue
        
        # 如果有負樣本且使用COCO格式，重新生成完整的annotations.json
        if args.format in ["coco", "all"] and args.create_empty_labels and negative_count > 0:
            coco_path = os.path.join(args.output_dir, "annotations.json")
            save_coco_json(coco_images_info, coco_path, category_name=label_name)
    
    print(f"\n{'='*60}")
    print(f"[完成] 成功生成了 {success_count}/{args.num_samples} 個增強樣本")
    if args.copy_remaining:
        print(f"[完成] 複製了 {negative_count} 張負樣本")
        print(f"總計：{success_count + negative_count} 張圖片")
    print(f"輸出位置：{args.output_dir}")
    print(f"- images/: 貼上物件後的圖片")
    if negative_count > 0:
        print(f"  - augmented_XXXX.*: 有物件的增強圖片 ({success_count}張)")
        print(f"  - negative_XXXX.*: 無物件的負樣本 ({negative_count}張)")
    print(f"- labels/: 標註檔案 ({args.format})")
    print(f"- 標籤名稱：{label_name}")
    if region:
        print(f"- 貼上區域：x={region[0]}-{region[2]}, y={region[1]}-{region[3]}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

# === 針對你的使用情況 ===

# python3 paste_and_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --original_json sonar/20251029062806/20251029062806_670.json \
#   --image_dir sonar/20251029062806 \
#   --output_dir sonar/augmented \
#   --num_samples 50 \
#   --region 40,105,570,255 \
#   --format all \
#   --max_rotation 30 \
#   --scale_min 0.9 \
#   --scale_max 1.1 \
#   --debug \
#   --visualize_region

# === 針對你的使用情況 ===

# python3 paste_and_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --original_json sonar/20251029062806/20251029062806_670.json \
#   --image_dir sonar/20251029062806 \
#   --output_dir sonar/augmented \
#   --num_samples 50 \
#   --region 40,105,570,255 \
#   --format all \
#   --max_rotation 30 \
#   --scale_min 0.9 \
#   --scale_max 1.1 \
#   --debug \
#   --visualize_region

# === 使用範例 ===

# 1. 手動指定HF區域（推薦，最精確）
# python3 paste_and_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --image_dir sonar/20251029062806 \
#   --output_dir sonar/augmented \
#   --num_samples 50 \
#   --label boat \
#   --region 40,105,570,255 \
#   --visualize_region

# python3 paste_n_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --image_dir sonar/20251029062806 \
#   --output_dir sonar/augmented \
#   --num_samples 1 \
#   --region 40,105,570,255 \
#   --visualize_region


# 2. 使用百分比指定區域（例如中間60%的區域）
# python3 paste_and_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --image_dir sonar/20251029062806 \
#   --output_dir sonar/augmented \
#   --num_samples 50 \
#   --label boat \
#   --region_percent 20,10,80,90 \
#   --format all

# 3. 自動偵測HF區域（尋找紅色垂直線）
# python3 paste_and_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --image_dir sonar/20251029062806 \
#   --output_dir sonar/augmented \
#   --num_samples 50 \
#   --label boat \
#   --auto_detect_region \
#   --visualize_region

# python3 paste_and_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --original_json sonar/20251029062806/20251029062806_670.json \
#   --image_dir sonar/20251029062806 \
#   --output_dir sonar/augmented \
#   --num_samples 50 \
#   --region 40,105,570,255 \
#   --format all \
#   --max_rotation 30 \
#   --scale_min 0.9 \
#   --scale_max 1.1 \
#   --debug \
#   --visualize_region

# python3 paste_n_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --original_json sonar/20251029062806/20251029062806_669.json \
#   --image_dir sonar/all2 \
#   --output_dir sonar/augmented2 \
#   --num_samples 250 \
#   --region 40,105,560,255 \
#   --format all \
#   --max_rotation 60 \
#   --scale_min 0.9 \
#   --scale_max 1.1
#   --copy_remaining \
#   --create_empty_labels

# python3 paste_and_label.py \
#   --object_png sonar/_artifacts/object_cutout.png \
#   --original_json sonar/20251029062806/20251029062806_670.json \
#   --image_dir sonar/all \
#   --output_dir sonar/augmented \
#   --num_samples 250 \
#   --region 40,105,560,255 \
#   --format all \
#   --max_rotation 60 \
#   --scale_min 0.9 \
#   --scale_max 1.1 \
#   --copy_remaining \
#   --create_empty_labels

# 1. 生成250張有物件的圖片，並保留剩餘171張作為負樣本
# python3 paste_and_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --original_json sonar/20251029062806/20251029062806_670.json \
#   --image_dir sonar/20251029062806 \
#   --output_dir sonar/augmented \
#   --num_samples 250 \
#   --region 40,105,570,255 \
#   --format all \
#   --max_rotation 30 \
#   --scale_min 0.9 \
#   --scale_max 1.1 \
#   --copy_remaining \
#   --create_empty_labels

# 2. 如果只想複製圖片，不需要空標註
# python3 paste_and_label.py \
#   --object_png sonar/_artifacts/object_cutout2.png \
#   --original_json sonar/20251029062806/20251029062806_670.json \
#   --image_dir sonar/20251029062806 \
#   --output_dir sonar/augmented \
#   --num_samples 250 \
#   --region 40,105,570,255 \
#   --format labelme \
#   --copy_remaining
