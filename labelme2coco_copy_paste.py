import os
import cv2
import json
import random
from tqdm import tqdm
import argparse
from merge_real_virtual import split_dataset, copy_images

# Load class names
def load_classes(class_file):
    with open(class_file, "r") as f:
        return [cname.strip() for cname in f.readlines()]

# Split labeled and background images
def split_images_by_json(labelme_dir):
    labeled_images = []
    background_images = []
    for file_name in os.listdir(labelme_dir):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            json_file = file_name.replace(".png", ".json").replace(".jpg", ".json")
            image_path = os.path.join(labelme_dir, file_name)
            json_path = os.path.join(labelme_dir, json_file)

            if os.path.exists(json_path):
                labeled_images.append((image_path, json_path))
            else:
                background_images.append(image_path)
    return labeled_images, background_images

# Extract labeled objects from images
def extract_labeled_objects(labeled_images):
    labeled_objects = []
    for image_path, json_path in tqdm(labeled_images, desc="Extracting labeled objects"):
        image = cv2.imread(image_path)
        with open(json_path, "r") as f:
            label_data = json.load(f)

        for shape in label_data["shapes"]:
            label = shape["label"]
            points = shape["points"]

            # Extract bbox
            x1, y1 = map(int, map(round, points[0]))
            x2, y2 = map(int, map(round, points[1]))
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            w, h = x2 - x1, y2 - y1

            # Validate bbox
            if w <= 0 or h <= 0 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                print(f"Invalid bbox in {json_path}: {x1, y1, x2, y2}")
                continue

            # Add labeled object
            labeled_objects.append({
                "file_name": os.path.basename(image_path),
                "label": label,
                "bbox": [x1, y1, w, h]
            })
    return labeled_objects



# Extract the largest labeled object from a specific JSON file
def extract_largest_from_json(json_path, image_dir):
    with open(json_path, "r") as f:
        label_data = json.load(f)

    image_name = os.path.basename(json_path).replace(".json", ".png")
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or invalid: {image_path}")

    largest_bbox = None
    largest_area = 0

    for shape in label_data["shapes"]:
        label = shape["label"]
        points = shape["points"]

        # Extract bbox
        x1, y1 = map(int, map(round, points[0]))
        x2, y2 = map(int, map(round, points[1]))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        w, h = x2 - x1, y2 - y1
        area = w * h

        # Validate bbox
        if w <= 0 or h <= 0 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            continue

        # Check if this bbox is the largest so far
        if area > largest_area:
            largest_area = area
            largest_bbox = {
                "label": label,
                "bbox": [x1, y1, w, h],
                "area": area
            }

    if largest_bbox:
        x1, y1, w, h = largest_bbox["bbox"]
        cropped_obj = image[y1:y1 + h, x1:x1 + w]
        return {
            "image": cropped_obj,
            "label": largest_bbox["label"],
            "bbox": [x1, y1, w, h],
            "area": largest_bbox["area"]
        }
    else:
        raise ValueError(f"No valid bounding box found in: {json_path}")


# Augment and paste the specific object onto background images
def augment_and_paste(labeled_object, background_images, output_image_dir, output_annotation_dir, class_list):
    augmented_images = []
    annotation_id = 1
    image_id = 1

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)

    for bg_image_path in tqdm(background_images, desc="Augmenting and pasting objects"):
        bg_image = cv2.imread(bg_image_path)
        if bg_image is None:
            print(f"Invalid background image: {bg_image_path}")
            continue
        bg_height, bg_width = bg_image.shape[:2]

        bg_filename = os.path.basename(bg_image_path)
        new_image_name = f"aug_{bg_filename}"
        new_image_path = os.path.join(output_image_dir, new_image_name)

        # 70% chance to copy-paste, 30% chance to do nothing
        if random.random() < 0.7:
            cropped_obj = labeled_object["image"]
            obj_label = labeled_object["label"]
            bbox_w, bbox_h = labeled_object["bbox"][2], labeled_object["bbox"][3]

            # Apply random scaling
            scale = random.uniform(0.7, 1.0)
            new_w, new_h = int(bbox_w * scale), int(bbox_h * scale)
            cropped_obj = cv2.resize(cropped_obj, (new_w, new_h))

            # Decide position based on size
            margin_x = bg_width // 10  # Leave margin on the sides
            margin_y = bg_height // 2  # Place in the upper 2/3 region

            new_x = random.randint(margin_x, bg_width - margin_x - new_w)
            new_y = random.randint(margin_y, bg_height - new_h)

            # Paste the object onto the background
            try:
                bg_image[new_y:new_y + new_h, new_x:new_x + new_w] = cropped_obj
            except ValueError:
                print(f"Object does not fit on the background image: {bg_image_path}")
                continue

            # Create new annotation
            new_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_list.index(obj_label),
                "bbox": [new_x, new_y, new_w, new_h],
                "area": new_w * new_h,
                "iscrowd": 0
            }

            # Save the new image
            cv2.imwrite(new_image_path, bg_image)

            # Append to augmented images
            augmented_images.append({
                "file_name": new_image_name,
                "width": bg_width,
                "height": bg_height,
                "id": image_id,
                "annotations": [new_annotation]
            })
            annotation_id += 1
        else:
            # Do nothing augmentation
            cv2.imwrite(new_image_path, bg_image)
            augmented_images.append({
                "file_name": new_image_name,
                "width": bg_width,
                "height": bg_height,
                "id": image_id,
                "annotations": []
            })

        image_id += 1

    return augmented_images

# Merge multiple COCO datasets
def merge_coco(coco_data_list):
    merged_data = coco_data_list[0]
    image_id_offset = 0
    annotation_id_offset = 0
    for coco_data in coco_data_list[1:]:
        image_id_offset += len(merged_data["images"])
        annotation_id_offset += len(merged_data["annotations"])
        for image in coco_data["images"]:
            image["id"] += image_id_offset
            merged_data["images"].append(image)
        for annotation in coco_data["annotations"]:
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
            merged_data["annotations"].append(annotation)
    return merged_data

# Save COCO format dataset
def save_coco(data, output_json):
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process and augment dataset.")
    parser.add_argument("--labelme_dir", type=str, default="bags_processing/label/2024-11-23-15-28-41-stitched", help="Path to labelme directory")
    parser.add_argument("--json_path", type=str, default="bags_processing/label/2024-11-23-15-28-41-stitched/544.json", help="Path to JSON file")
    parser.add_argument("--output_dir", type=str, default="Tainan_Lifebuoy_dataset", help="Output directory")
    parser.add_argument("--classes", type=str, default="bags_processing/classes.txt", help="Path to class names file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(output_image_dir, exist_ok=True)

    class_list = load_classes(args.classes)

    # Split labeled and background images
    labeled_images, background_images = split_images_by_json(args.labelme_dir)

    # Extract labeled data into COCO format
    labeled_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": cname} for i, cname in enumerate(class_list)]
    }
    image_id = 1
    annotation_id = 1

    labeled_objects = extract_labeled_objects(labeled_images)
    for obj in labeled_objects:
        # Copy labeled images to the output directory
        src_path = os.path.join(args.labelme_dir, obj["file_name"])
        dst_path = os.path.join(output_image_dir, obj["file_name"])
        if not os.path.exists(dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            os.system(f"cp {src_path} {dst_path}")

        # Add labeled images and annotations into labeled_data
        labeled_data["images"].append({
            "id": image_id,
            "file_name": obj["file_name"],
            "width": obj["bbox"][2],
            "height": obj["bbox"][3]
        })

        labeled_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_list.index(obj["label"]),
            "bbox": obj["bbox"],
            "area": obj["bbox"][2] * obj["bbox"][3],
            "iscrowd": 0
        })
        image_id += 1
        annotation_id += 1

    # Process the largest labeled object from the specific JSON
    labeled_object = extract_largest_from_json(args.json_path, args.labelme_dir)

    # Perform augmentation
    augmented_images = augment_and_paste(
        labeled_object,
        background_images,
        output_image_dir,
        os.path.join(args.output_dir, "annotations"),
        class_list
    )

    # Create COCO dataset for augmented images
    augmented_data = {
        "images": [img for img in augmented_images],
        "annotations": [ann for img in augmented_images for ann in img["annotations"]],
        "categories": [{"id": i, "name": cname} for i, cname in enumerate(class_list)]
    }

    # Merge labeled and augmented data
    merged_data = merge_coco([labeled_data, augmented_data])

    # Split data into training and validation sets
    train_data, val_data = split_dataset(merged_data)
    annotations_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # Save COCO format datasets
    save_coco(train_data, os.path.join(annotations_dir, "instances_train2024.json"))
    save_coco(val_data, os.path.join(annotations_dir, "instances_val2024.json"))

    # Copy images to output directories
    os.makedirs(output_image_dir, exist_ok=True)
    output_image_dir = f"{args.output_dir}/train2024"
    copy_images(train_data['images'], [f"{args.output_dir}/images"], output_image_dir)
    output_image_dir = f"{args.output_dir}/val2024"
    copy_images(val_data['images'], [f"{args.output_dir}/images"], output_image_dir)

    # Copy class file
    os.system(f"cp {args.classes} {args.output_dir}/annotations/classes.txt")

    print(f"Processing and augmentation completed. Output saved to {args.output_dir}")



if __name__ == "__main__":
    main()