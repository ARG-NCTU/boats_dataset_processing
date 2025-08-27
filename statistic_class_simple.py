import os
import json
import csv
import argparse
from collections import OrderedDict, defaultdict

def load_categories_id_to_name(*json_paths):
    """
    從一個或多個 COCO JSON 檔案收集 categories，建立
    - id_to_name: {category_id: name}
    - ordered_ids: 依 category_id 遞增排序的唯一 id 清單
    """
    id_to_name = {}
    for p in json_paths:
        if p is None:
            continue
        with open(p, 'r') as f:
            data = json.load(f)
        for c in data.get('categories', []):
            cid = int(c['id'])
            cname = str(c['name'])
            id_to_name[cid] = cname  # 若重複 id 出現，以最後一次為準（通常一致）
    ordered_ids = sorted(id_to_name.keys())
    return id_to_name, ordered_ids

def statistic_classes_by_id(label_path):
    """回傳 {category_id: count} 與影像數量。"""
    with open(label_path, 'r') as f:
        data = json.load(f)
    counts = defaultdict(int)
    for ann in data.get('annotations', []):
        cid = int(ann['category_id'])
        counts[cid] += 1
    num_images = len(data.get('images', []))
    return counts, num_images

def main(args):
    # 從 COCO JSON 收集 categories（建議以 train 為主，這裡把 val/test 也併進去以防分割不一致）
    id_to_name, ordered_ids = load_categories_id_to_name(args.train_json, args.val_json, args.test_json)

    # 各 split 的計數與影像數
    train_counts, n_train_img = statistic_classes_by_id(args.train_json)
    val_counts,   n_val_img   = statistic_classes_by_id(args.val_json)
    if args.test_json:
        test_counts,  n_test_img  = statistic_classes_by_id(args.test_json)
    else:
        test_counts,  n_test_img  = defaultdict(int), 0

    # 影像總數 & 百分比（避免除以 0）
    total_images = n_train_img + n_val_img + n_test_img
    train_img_pct = (n_train_img / total_images * 100) if total_images else 0.0
    val_img_pct   = (n_val_img   / total_images * 100) if total_images else 0.0
    test_img_pct  = (n_test_img  / total_images * 100) if total_images else 0.0

    # 標註總數 & 百分比
    sum_train_ann = sum(train_counts.values())
    sum_val_ann   = sum(val_counts.values())
    sum_test_ann  = sum(test_counts.values()) if args.test_json else 0
    total_ann     = sum_train_ann + sum_val_ann + sum_test_ann
    train_ann_pct = (sum_train_ann / total_ann * 100) if total_ann else 0.0
    val_ann_pct   = (sum_val_ann   / total_ann * 100) if total_ann else 0.0
    test_ann_pct  = (sum_test_ann  / total_ann * 100) if total_ann else 0.0

    # 平均每張圖的標註數
    avg_train = (sum_train_ann / n_train_img) if n_train_img else 0.0
    avg_val   = (sum_val_ann   / n_val_img)   if n_val_img   else 0.0
    avg_test  = (sum_test_ann  / n_test_img)  if n_test_img  else 0.0

    # --- Summary ---
    print('-' * 50)
    print('Total number of classes:', len(ordered_ids))
    print('Total number of images:', total_images)
    print(f'Training images: {n_train_img} ({train_img_pct:.2f}%)')
    print(f'Validation images: {n_val_img} ({val_img_pct:.2f}%)')
    if args.test_json:
        print(f'Test images: {n_test_img} ({test_img_pct:.2f}%)')
    print('Total number of annotations:', total_ann)
    print(f'Training annotations: {sum_train_ann} ({train_ann_pct:.2f}%)')
    print(f'Validation annotations: {sum_val_ann} ({val_ann_pct:.2f}%)')
    if args.test_json:
        print(f'Test annotations: {sum_test_ann} ({test_ann_pct:.2f}%)')
    print('Average annotations per training image:', avg_train)
    print('Average annotations per validation image:', avg_val)
    if args.test_json:
        print('Average annotations per test image:', avg_test)

    # --- Class-wise table ---
    print('-' * 50)
    if args.test_json:
        print('Class Name            Train Annotations    Validation Annotations    Test Annotations')
    else:
        print('Class Name            Train Annotations    Validation Annotations')
    print('-' * 50)

    rows = []
    for cid in ordered_ids:
        cname = id_to_name[cid]
        tr = train_counts.get(cid, 0)
        va = val_counts.get(cid, 0)
        if args.test_json:
            te = test_counts.get(cid, 0)
            print(f'{cname:<20} {tr:<20} {va:<20} {te:<20}')
            rows.append([cname, tr, va, te])
        else:
            print(f'{cname:<20} {tr:<20} {va:<20}')
            rows.append([cname, tr, va])

    # --- Save CSV ---
    out_csv = args.output_csv
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        if args.test_json:
            writer.writerow(['Class Name', 'Train Annotations', 'Validation Annotations', 'Test Annotations'])
        else:
            writer.writerow(['Class Name', 'Train Annotations', 'Validation Annotations'])
        writer.writerows(rows)

    print(f'Class statistics saved to {out_csv}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate class statistics directly from COCO categories.")
    parser.add_argument('--train_json', type=str, required=True, help="Path to training COCO JSON.")
    parser.add_argument('--val_json',   type=str, required=True, help="Path to validation COCO JSON.")
    parser.add_argument('--test_json',  type=str, default=None, help="Path to test COCO JSON (optional).")
    parser.add_argument('--output_csv', type=str, default='class_statistics.csv', help="Output CSV path.")
    args = parser.parse_args()
    main(args)

# python3 statistic_class_simple.py --train_json Boat_unity_dataset/annotations/instances_train2024.json --val_json Boat_unity_dataset/annotations/instances_val2024.json --test_json Boat_unity_dataset/annotations/instances_test2024.json