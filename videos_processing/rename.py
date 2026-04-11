#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# rename_to_image.py - 將 negative 和 augmented 都重命名為 image

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

def rename_to_image(target_dir, dry_run=False):
    """
    將資料夾中的 negative_xxxx 和 augmented_xxxx 都改名為 image_xxxx
    
    Args:
        target_dir: 目標資料夾路徑
        dry_run: 是否只顯示操作而不實際執行
    """
    
    # 檢查資料夾是否存在
    if not os.path.exists(target_dir):
        print(f"[Error] 目標資料夾不存在：{target_dir}")
        return False
    
    images_dir = os.path.join(target_dir, "images")
    labels_dir = os.path.join(target_dir, "labels")
    
    # 確保子資料夾存在
    if not os.path.exists(images_dir):
        print(f"[Error] images 資料夾不存在：{images_dir}")
        return False
    
    if not os.path.exists(labels_dir):
        print(f"[Warning] labels 資料夾不存在：{labels_dir}")
    
    if dry_run:
        print(f"\n[DRY RUN模式] 以下是將要執行的操作：\n")
    else:
        print(f"\n開始重命名...\n")
    
    # 收集所有需要重命名的檔案
    rename_operations = []
    stats = {'augmented': 0, 'negative': 0}
    
    # 第一步：收集並排序所有圖片檔案
    if os.path.exists(images_dir):
        image_files = []
        for filename in os.listdir(images_dir):
            prefix, number, ext = parse_filename(filename)
            if prefix in ['augmented', 'negative']:
                image_files.append({
                    'filename': filename,
                    'prefix': prefix,
                    'number': number,
                    'ext': ext
                })
                stats[prefix] += 1
        
        # 先按 prefix 排序（augmented 在前），再按編號排序
        # 這樣可以保持 augmented 和 negative 的相對順序
        image_files.sort(key=lambda x: (x['prefix'] == 'negative', x['number']))
        
        # 分配新編號
        for new_num, file_info in enumerate(image_files):
            old_filename = file_info['filename']
            new_filename = f"image_{new_num:04d}{file_info['ext']}"
            
            rename_operations.append({
                'type': 'image',
                'old_path': os.path.join(images_dir, old_filename),
                'new_path': os.path.join(images_dir, new_filename),
                'old_name': old_filename,
                'new_name': new_filename,
                'old_prefix': file_info['prefix']
            })
    
    # 建立舊檔名到新檔名的映射表
    image_name_mapping = {}
    for op in rename_operations:
        if op['type'] == 'image':
            old_base = os.path.splitext(op['old_name'])[0]
            new_base = os.path.splitext(op['new_name'])[0]
            image_name_mapping[old_base] = new_base
    
    # 第二步：處理labels資料夾
    if os.path.exists(labels_dir):
        for filename in os.listdir(labels_dir):
            prefix, number, ext = parse_filename(filename)
            
            if prefix not in ['augmented', 'negative']:
                continue
            
            # 取得對應的基礎名稱
            old_base = os.path.splitext(filename)[0]
            
            # 從映射表中找到新的基礎名稱
            if old_base not in image_name_mapping:
                print(f"[Warning] 找不到對應的圖片：{filename}")
                continue
            
            new_base = image_name_mapping[old_base]
            new_filename = f"{new_base}{ext}"
            
            # 找到對應的圖片檔名（用於更新JSON）
            img_exts = ['.png', '.jpg', '.jpeg', '.bmp']
            corresponding_image = None
            for img_ext in img_exts:
                img_name = f"{new_base}{img_ext}"
                if any(op['new_name'] == img_name for op in rename_operations if op['type'] == 'image'):
                    corresponding_image = img_name
                    break
            
            rename_operations.append({
                'type': 'label',
                'old_path': os.path.join(labels_dir, filename),
                'new_path': os.path.join(labels_dir, new_filename),
                'old_name': filename,
                'new_name': new_filename,
                'old_prefix': prefix,
                'is_json': ext == '.json',
                'corresponding_image': corresponding_image
            })
    
    # 顯示統計
    print(f"統計資訊：")
    print(f"  augmented: {stats['augmented']}張")
    print(f"  negative: {stats['negative']}張")
    print(f"  總計: {stats['augmented'] + stats['negative']}張")
    print(f"  將全部重命名為: image_0000 ~ image_{stats['augmented'] + stats['negative'] - 1:04d}\n")
    
    if dry_run:
        print("範例操作（前20個）：")
        shown = 0
        for op in rename_operations:
            if op['type'] == 'image':
                print(f"  {op['old_name']} -> {op['new_name']}")
                shown += 1
                if shown >= 20:
                    break
        
        if len([op for op in rename_operations if op['type'] == 'image']) > 20:
            remaining = len([op for op in rename_operations if op['type'] == 'image']) - 20
            print(f"  ... 還有 {remaining} 個圖片")
        
        print(f"\n對應的 labels 也會一起重命名")
        print("\n使用 --execute 參數來實際執行重命名")
        return True
    
    # 使用臨時檔名避免衝突
    # 策略：先全部改成臨時名稱，再改成最終名稱
    print("階段 1/3: 重命名為臨時檔名...")
    temp_operations = []
    
    for op in tqdm(rename_operations, desc="建立臨時檔名"):
        try:
            # 使用 .tmp_ 前綴作為臨時檔名
            temp_name = f".tmp_{op['new_name']}"
            temp_path = os.path.join(os.path.dirname(op['new_path']), temp_name)
            
            shutil.move(op['old_path'], temp_path)
            
            temp_operations.append({
                'temp_path': temp_path,
                'final_path': op['new_path'],
                'temp_name': temp_name,
                'final_name': op['new_name'],
                'is_json': op.get('is_json', False),
                'corresponding_image': op.get('corresponding_image')
            })
            
        except Exception as e:
            print(f"\n[Error] 重命名失敗：{op['old_name']} -> {e}")
            return False
    
    print("\n階段 2/3: 重命名為最終檔名...")
    
    for op in tqdm(temp_operations, desc="完成重命名"):
        try:
            shutil.move(op['temp_path'], op['final_path'])
            
            # 如果是JSON，更新imagePath
            if op['is_json'] and op['corresponding_image']:
                update_json_imagepath(op['final_path'], op['corresponding_image'])
            
        except Exception as e:
            print(f"\n[Error] 最終重命名失敗：{op['temp_name']} -> {e}")
            return False
    
    print("\n階段 3/3: 驗證結果...")
    
    # 驗證結果
    success = True
    for op in temp_operations:
        if not os.path.exists(op['final_path']):
            print(f"[Error] 檔案不存在：{op['final_name']}")
            success = False
    
    if success:
        print(f"\n{'='*60}")
        print(f"[完成] 重命名成功！")
        print(f"  處理檔案數：{len(rename_operations)}")
        print(f"  目標資料夾：{target_dir}")
        print(f"{'='*60}")
    
    return success

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="將 negative 和 augmented 前綴都改為 image，並重新編號")
    parser.add_argument("--target", type=str, required=True,
                        help="目標資料夾路徑")
    parser.add_argument("--dry-run", action="store_true",
                        help="預覽模式：只顯示將要執行的操作，不實際執行")
    parser.add_argument("--execute", action="store_true",
                        help="執行重命名（需要明確指定此參數）")
    
    args = parser.parse_args()
    
    if not args.execute and not args.dry_run:
        print("[提示] 請使用 --dry-run 預覽操作，或使用 --execute 執行重命名")
        print("範例：")
        print("  預覽：python3 rename_to_image.py --target augmented --dry-run")
        print("  執行：python3 rename_to_image.py --target augmented --execute")
        return
    
    rename_to_image(args.target, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

# === 使用範例 ===

# 1. 先預覽操作（強烈建議！）
# python3 rename.py --target augmented --dry-run

# 2. 確認無誤後執行重命名
# python3 rename.py --target augmented --execute