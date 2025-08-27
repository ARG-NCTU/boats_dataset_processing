import json
import random
import shutil
import os
from pathlib import Path

random.seed(42)

src_root = Path("TW_Marine_5cls_dataset")
dst_root = Path("TW_Marine_5cls_dataset_split")
dst_root.mkdir(exist_ok=True)

src_ann = src_root / "annotations"
src_images = src_root / "images"
src_train_imgs = src_root / "train2024"
src_val_imgs = src_root / "val2024"

dst_ann = dst_root / "annotations"
dst_images = dst_root / "images"
dst_train_imgs = dst_root / "train2024"
dst_val_imgs = dst_root / "val2024"
dst_test_imgs = dst_root / "test2024"

# 建立目錄
dst_ann.mkdir(exist_ok=True)
dst_images.mkdir(exist_ok=True)
dst_train_imgs.mkdir(exist_ok=True)
dst_val_imgs.mkdir(exist_ok=True)
dst_test_imgs.mkdir(exist_ok=True)

# 複製 images/ 和 val2024/ 原封不動
print("Copying images/ ...")
shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
print("Copying val2024/ ...")
shutil.copytree(src_val_imgs, dst_val_imgs, dirs_exist_ok=True)

# 處理 annotations
with open(src_ann / "instances_train2024.json") as f:
    train_data = json.load(f)
with open(src_ann / "instances_val2024.json") as f:
    val_data = json.load(f)

images = train_data["images"]
annotations = train_data["annotations"]
categories = train_data["categories"]

num_test = len(images) // 8
test_image_ids = set(img["id"] for img in random.sample(images, num_test))

train_images = [img for img in images if img["id"] not in test_image_ids]
test_images = [img for img in images if img["id"] in test_image_ids]

train_annotations = [ann for ann in annotations if ann["image_id"] not in test_image_ids]
test_annotations = [ann for ann in annotations if ann["image_id"] in test_image_ids]

new_train_data = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories
}

new_test_data = {
    "images": test_images,
    "annotations": test_annotations,
    "categories": categories
}

with open(dst_ann / "instances_train2024.json", "w") as f:
    json.dump(new_train_data, f, indent=4)
with open(dst_ann / "instances_test2024.json", "w") as f:
    json.dump(new_test_data, f, indent=4)
with open(dst_ann / "instances_val2024.json", "w") as f:
    json.dump(val_data, f, indent=4)

print(f"New train: {len(train_images)} images")
print(f"New test: {len(test_images)} images")

# 複製 train2024 裡的圖片到 train2024/ 和 test2024/
src_train_img_dict = {img.name: img for img in src_train_imgs.iterdir() if img.is_file()}

train_img_filenames = {img["file_name"] for img in train_images}
test_img_filenames = {img["file_name"] for img in test_images}

print("Copying train2024/ images ...")
for fname in train_img_filenames:
    shutil.copy2(src_train_img_dict[fname], dst_train_imgs / fname)

print("Copying test2024/ images ...")
for fname in test_img_filenames:
    shutil.copy2(src_train_img_dict[fname], dst_test_imgs / fname)

print("✅ Done. Output in", dst_root)
