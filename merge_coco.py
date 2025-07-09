import argparse
import os
import json
import shutil
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved: {file_path} ({len(data['images'])} images, {len(data['annotations'])} annotations)")

def merge_two_coco_sets(set_a, set_b):
    # 建立 category name 到 ID 的映射
    name_to_id = {cat['name']: cat['id'] for cat in set_a['categories']}
    next_cat_id = max(name_to_id.values()) + 1

    # 建立 B 的舊 ID 到新 ID 的 mapping
    b_id_map = {}
    for cat in set_b['categories']:
        name = cat['name']
        if name in name_to_id:
            b_id_map[cat['id']] = name_to_id[name]
        else:
            name_to_id[name] = next_cat_id
            b_id_map[cat['id']] = next_cat_id
            set_a['categories'].append({'id': next_cat_id, 'name': name})
            next_cat_id += 1

    # shift B 的 image_id 和 annotation_id
    image_id_shift = max(img['id'] for img in set_a['images']) + 1
    anno_id_shift = max(anno['id'] for anno in set_a['annotations']) + 1

    for img in set_b['images']:
        img['id'] += image_id_shift
    for anno in set_b['annotations']:
        anno['id'] += anno_id_shift
        anno['image_id'] += image_id_shift
        anno['category_id'] = b_id_map[anno['category_id']]  # 重新對應新的 category_id

    return {
        'images': set_a['images'] + set_b['images'],
        'annotations': set_a['annotations'] + set_b['annotations'],
        'categories': set_a['categories']
    }


def copy_images(image_list, source_dirs, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    copied, missing = 0, []
    for img in tqdm(image_list, desc="📦 Copying images"):
        file_name = img['file_name']
        found = False
        for src in source_dirs:
            src_path = os.path.join(src, file_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(dest_dir, file_name))
                copied += 1
                found = True
                break
        if not found:
            missing.append(file_name)
    print(f"✅ Copied {copied} images.")
    if missing:
        print(f"⚠️ {len(missing)} images not found:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print("  ...")

def main():
    parser = argparse.ArgumentParser(description="Merge two COCO-format datasets from directories.")
    parser.add_argument('--dataset1', type=str, default='KS_Buoy_dataset', help='First dataset directory (contains annotations/ and images/)')
    parser.add_argument('--dataset2', type=str, default='GuardBoat_dataset', help='Second dataset directory (contains annotations/ and images/)')
    parser.add_argument('--output_dir', type=str, default='TW_Marine_2cls_dataset', help='Output directory')
    # parser.add_argument('--copy_images', action='store_true', help='Copy images to output_dir/images/')

    args = parser.parse_args()

    # Paths
    ann1_dir = os.path.join(args.dataset1, 'annotations')
    ann2_dir = os.path.join(args.dataset2, 'annotations')

    train1 = load_json(os.path.join(ann1_dir, 'instances_train2024.json'))
    train2 = load_json(os.path.join(ann2_dir, 'instances_train2024.json'))
    val1 = load_json(os.path.join(ann1_dir, 'instances_val2024.json'))
    val2 = load_json(os.path.join(ann2_dir, 'instances_val2024.json'))
    test1 = load_json(os.path.join(ann1_dir, 'instances_test2024.json'))
    test2 = load_json(os.path.join(ann2_dir, 'instances_test2024.json'))

    merged_train = merge_two_coco_sets(train1, train2)
    merged_val = merge_two_coco_sets(val1, val2)
    merged_test = merge_two_coco_sets(test1, test2)

    save_json(merged_train, os.path.join(args.output_dir, 'annotations', 'instances_train2024.json'))
    save_json(merged_val, os.path.join(args.output_dir, 'annotations', 'instances_val2024.json'))
    save_json(merged_test, os.path.join(args.output_dir, 'annotations', 'instances_test2024.json'))

    # if args.copy_images:
    copy_images(merged_train['images'] + merged_val['images'] + merged_test['images'],
                source_dirs=[os.path.join(args.dataset1, 'images'),
                                os.path.join(args.dataset2, 'images')],
                dest_dir=os.path.join(args.output_dir, 'images'))

if __name__ == "__main__":
    main()
