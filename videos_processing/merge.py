#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# merge_augmented.py - 合併多個augmented資料夾，自動重新編號

import os
import json
import shutil
import re
from pathlib import Path
from tqdm import tqdm

def parse_filename(filename):
    """
    解析檔名，提取前綴和編號
    例如：augmented_0123.png -> ('augmented', 123, '.png')
         negative_0045.json -> ('negative', 45, '.json')
    """
    match = re.match(r'^(augmented|negative)_(\d+)(\.\w+)$', filename)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        ext = match.group(3)
        return prefix, number, ext
    return None, None, None

def get_max_number(folder, prefix):
    """找出資料夾中指定前綴的最大編號"""
    max_num = -1
    
    if not os.path.exists(folder):
        return max_num
    
    for filename in os.listdir(folder):
        file_prefix, number, _ = parse_filename(filename)
        if file_prefix == prefix and number is not None:
            max_num = max(max_num, number)
    
    return max_num

def update_json_imagepath(json_path, new_image_name):
    """更新JSON檔案中的imagePath欄位"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新imagePath
        data['imagePath'] = new_image_name
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"[Warning] 更新JSON失敗 {json_path}: {e}")
        return False

def merge_augmented_folders(source_dir, target_dir, dry_run=False):
    """
    將source_dir的內容合併到target_dir，自動重新編號避免衝突
    
    Args:
        source_dir: 來源資料夾路徑
        target_dir: 目標資料夾路徑
        dry_run: 是否只顯示操作而不實際執行
    """
    
    # 檢查資料夾是否存在
    if not os.path.exists(source_dir):
        print(f"[Error] 來源資料夾不存在：{source_dir}")
        return False
    
    if not os.path.exists(target_dir):
        print(f"[Error] 目標資料夾不存在：{target_dir}")
        return False
    
    source_images = os.path.join(source_dir, "images")
    source_labels = os.path.join(source_dir, "labels")
    target_images = os.path.join(target_dir, "images")
    target_labels = os.path.join(target_dir, "labels")
    
    # 確保子資料夾存在
    for folder in [source_images, source_labels, target_images, target_labels]:
        if not os.path.exists(folder):
            print(f"[Warning] 資料夾不存在：{folder}")
    
    # 找出目標資料夾中每個前綴的最大編號
    aug_max = get_max_number(target_images, 'augmented')
    neg_max = get_max_number(target_images, 'negative')
    
    print(f"\n目標資料夾現有編號：")
    print(f"  augmented: 0 ~ {aug_max} ({aug_max + 1}張)")
    print(f"  negative: 0 ~ {neg_max} ({neg_max + 1}張)")
    
    # 下一個可用編號
    next_aug_num = aug_max + 1
    next_neg_num = neg_max + 1
    
    print(f"\n來源資料夾將從以下編號開始：")
    print(f"  augmented: {next_aug_num}")
    print(f"  negative: {next_neg_num}")
    
    if dry_run:
        print(f"\n[DRY RUN模式] 以下是將要執行的操作：\n")
    else:
        print(f"\n開始合併...\n")
    
    # 收集所有需要處理的檔案
    operations = []
    stats = {'augmented': 0, 'negative': 0}
    
    # 處理images資料夾
    if os.path.exists(source_images):
        image_files = sorted(os.listdir(source_images))
        
        for filename in image_files:
            prefix, old_num, ext = parse_filename(filename)
            
            if prefix is None:
                print(f"[Skip] 跳過非標準檔名：{filename}")
                continue
            
            # 決定新編號
            if prefix == 'augmented':
                new_num = next_aug_num
                next_aug_num += 1
            elif prefix == 'negative':
                new_num = next_neg_num
                next_neg_num += 1
            else:
                continue
            
            new_filename = f"{prefix}_{new_num:04d}{ext}"
            
            operations.append({
                'type': 'image',
                'old_path': os.path.join(source_images, filename),
                'new_path': os.path.join(target_images, new_filename),
                'old_name': filename,
                'new_name': new_filename,
                'prefix': prefix
            })
            
            stats[prefix] += 1
    
    # 建立舊檔名到新檔名的映射表（從images）
    image_name_mapping = {}
    for op in operations:
        if op['type'] == 'image':
            # 取得不含副檔名的基礎名稱
            old_base = os.path.splitext(op['old_name'])[0]
            new_base = os.path.splitext(op['new_name'])[0]
            image_name_mapping[old_base] = new_base
    
    # 處理labels資料夾
    if os.path.exists(source_labels):
        label_files = sorted(os.listdir(source_labels))
        
        for filename in label_files:
            prefix, old_num, ext = parse_filename(filename)
            
            if prefix is None:
                continue
            
            # 取得對應的基礎名稱
            old_base = os.path.splitext(filename)[0]
            
            # 從映射表中找到新的編號
            if old_base not in image_name_mapping:
                print(f"[Warning] 找不到對應的圖片：{filename}")
                continue
            
            new_base = image_name_mapping[old_base]
            new_num = int(new_base.split('_')[1])  # 從新基礎名稱提取編號
            
            new_filename = f"{prefix}_{new_num:04d}{ext}"
            
            # 找到對應的圖片檔名（用於更新JSON）
            img_exts = ['.png', '.jpg', '.jpeg', '.bmp']
            corresponding_image = None
            for img_ext in img_exts:
                img_name = f"{prefix}_{new_num:04d}{img_ext}"
                if any(op['new_name'] == img_name for op in operations if op['type'] == 'image'):
                    corresponding_image = img_name
                    break
            
            operations.append({
                'type': 'label',
                'old_path': os.path.join(source_labels, filename),
                'new_path': os.path.join(target_labels, new_filename),
                'old_name': filename,
                'new_name': new_filename,
                'prefix': prefix,
                'is_json': ext == '.json',
                'corresponding_image': corresponding_image
            })
    
    # 顯示統計
    print(f"統計資訊：")
    print(f"  augmented: {stats['augmented']}張")
    print(f"  negative: {stats['negative']}張")
    print(f"  總計: {stats['augmented'] + stats['negative']}張\n")
    
    if dry_run:
        print("範例操作（前10個）：")
        for i, op in enumerate(operations[:10]):
            print(f"  {op['old_name']} -> {op['new_name']}")
        if len(operations) > 10:
            print(f"  ... 還有 {len(operations) - 10} 個操作")
        print("\n使用 --execute 參數來實際執行合併")
        return True
    
    # 實際執行複製和重命名
    success_count = 0
    failed_count = 0
    
    for op in tqdm(operations, desc="複製檔案"):
        try:
            # 複製檔案
            shutil.copy2(op['old_path'], op['new_path'])
            
            # 如果是JSON，更新imagePath
            if op['type'] == 'label' and op['is_json'] and op['corresponding_image']:
                update_json_imagepath(op['new_path'], op['corresponding_image'])
            
            success_count += 1
            
        except Exception as e:
            print(f"\n[Error] 複製失敗：{op['old_name']} -> {e}")
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"[完成] 合併結果：")
    print(f"  成功：{success_count} 個檔案")
    print(f"  失敗：{failed_count} 個檔案")
    print(f"  目標資料夾：{target_dir}")
    print(f"{'='*60}")
    
    return failed_count == 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="合併兩個augmented資料夾，自動重新編號避免衝突")
    parser.add_argument("--source", type=str, required=True,
                        help="來源資料夾（要合併進去的）")
    parser.add_argument("--target", type=str, required=True,
                        help="目標資料夾（主要的）")
    parser.add_argument("--dry-run", action="store_true",
                        help="預覽模式：只顯示將要執行的操作，不實際執行")
    parser.add_argument("--execute", action="store_true",
                        help="執行合併（需要明確指定此參數）")
    
    args = parser.parse_args()
    
    if not args.execute and not args.dry_run:
        print("[提示] 請使用 --dry-run 預覽操作，或使用 --execute 執行合併")
        print("範例：")
        print("  預覽：python3 merge_augmented.py --source augmented3 --target augmented --dry-run")
        print("  執行：python3 merge_augmented.py --source augmented3 --target augmented --execute")
        return
    
    merge_augmented_folders(args.source, args.target, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

# === 使用範例 ===

# 1. 先預覽操作（強烈建議！）
# python3 merge_augmented.py \
#   --source sonar/augmented3 \
#   --target sonar/augmented \
#   --dry-run

# 2. 確認無誤後執行合併
# python3 merge_augmented.py \
#   --source sonar/augmented3 \
#   --target sonar/augmented \
#   --execute