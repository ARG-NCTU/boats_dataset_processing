# labelme file structure:
# bags_processing/d435_images/2020-11-17-15-00-00_mid/1.png

import os
import cv2
import numpy as np
from labelme import utils
import json
from PIL import Image
from tqdm import tqdm
import argparse
from merge_real_virtual import split_dataset, copy_images

def load_classes():
    class_list = []
    with open("bags_processing/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

# def labelme2real(labelme_dir, output_image_dir):

#     coco_data = {
#         "images": [],
#         "annotations": [],
#         "categories": [{"id": i, "name": cname} for i, cname in enumerate(class_list)]
#     }

#     image_id = 1
#     annotation_id = 1

#     for labelme_file in tqdm(os.listdir(labelme_dir), desc="Converting labelme to coco", leave=False):
#         if labelme_file.endswith(".json"):
#             with open(os.path.join(labelme_dir, labelme_file)) as f:
#                 data = json.load(f)
            
#             ############ image ############
#             image_name = labelme_file.replace(".json", ".png")
#             dir_name = labelme_dir.split("/")[-1]  
#             out_image_name = f"{dir_name}_{image_name}"
#             image = cv2.imread(os.path.join(labelme_dir, image_name))
#             height, width = image.shape[:2]
#             image_info = {
#                 "id": image_id,
#                 "file_name": out_image_name,
#                 "width": width,
#                 "height": height
#             }
#             coco_data["images"].append(image_info)

#             # copy image to output_dir
#             cv2.imwrite(os.path.join(output_image_dir, out_image_name), image)
            
#             ############ annotations ############
            
#             # Convert labelme format to coco format
#             for shape in data["shapes"]:
#                 points = shape["points"]
#                 label = shape["label"]
#                 x1, y1 = points[0]
#                 x1, y1 = int(round(x1)), int(round(y1))
#                 x2, y2 = points[1]
#                 x2, y2 = int(round(x2)), int(round(y2))
#                 if x1 > x2:
#                     x1, x2 = x2, x1
#                 if y1 > y2:
#                     y1, y2 = y2, y1
#                 w = x2 - x1
#                 h = y2 - y1
#                 if x1 < 0 or y1 < 0 or w <= 0 or h <= 0 or x2 > width or y2 > height:
#                     print("====================================")
#                     print(f"Invalid json file {labelme_file}, bbox: {x1, y1, x2, y2}")
#                     print("====================================")
#                     continue
#                 area = w * h
#                 bbox = [x1, y1, w, h]
#                 category_id = class_list.index(label)
#                 annotation = {
#                     "id": annotation_id,
#                     "image_id": image_id,
#                     "category_id": category_id,
#                     "bbox": bbox,
#                     "area": area,
#                     "iscrowd": 0
#                 }

#                 coco_data["annotations"].append(annotation)
#                 annotation_id += 1

#             image_id += 1

#     print(f"Converted {labelme_dir} to coco format")
#     return coco_data

def labelme2real(labelme_dir, output_image_dir):

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": cname} for i, cname in enumerate(class_list)]
    }

    image_id = 1
    annotation_id = 1

    for file_name in tqdm(os.listdir(labelme_dir), desc="Converting labelme to coco", leave=False):
        if file_name.endswith(".png"):  # Check all PNG files
            image_name = file_name
            dir_name = labelme_dir.split("/")[-1]
            out_image_name = f"{dir_name}_{image_name}"
            image_path = os.path.join(labelme_dir, image_name)
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping invalid image {image_path}")
                continue

            height, width = image.shape[:2]
            image_info = {
                "id": image_id,
                "file_name": out_image_name,
                "width": width,
                "height": height
            }
            coco_data["images"].append(image_info)

            # Copy image to output directory
            cv2.imwrite(os.path.join(output_image_dir, out_image_name), image)
            
            # Check if JSON file exists
            json_file = file_name.replace(".png", ".json")
            json_path = os.path.join(labelme_dir, json_file)
            if os.path.exists(json_path):
                with open(json_path) as f:
                    data = json.load(f)

                # Process annotations
                for shape in data["shapes"]:
                    points = shape["points"]
                    label = shape["label"]
                    x1, y1 = points[0]
                    x1, y1 = int(round(x1)), int(round(y1))
                    x2, y2 = points[1]
                    x2, y2 = int(round(x2)), int(round(y2))
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    w = x2 - x1
                    h = y2 - y1
                    if x1 < 0 or y1 < 0 or w <= 0 or h <= 0 or x2 > width or y2 > height:
                        print("====================================")
                        print(f"Invalid json file {json_file}, bbox: {x1, y1, x2, y2}")
                        print("====================================")
                        continue
                    area = w * h
                    bbox = [x1, y1, w, h]
                    category_id = class_list.index(label)
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    }

                    coco_data["annotations"].append(annotation)
                    annotation_id += 1
            else:
                # Add empty annotations for images without JSON
                print(f"No annotations for {image_name}. Adding empty annotations.")
            
            image_id += 1

    print(f"Converted {labelme_dir} to coco format")
    return coco_data


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

def save_coco(data, output_json):
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process real dataset.")
    
    # Define arguments
    parser.add_argument("--labelme_dir", type=str, default="bags_processing/d435_images")
    parser.add_argument("--output_dir", type=str, default="real_dataset")
    parser.add_argument("--classes", type=str, default="Boat_dataset_unity/Lifebuoy/classes.txt")
    
    args = parser.parse_args()

    os.makedirs(f"{args.output_dir}", exist_ok=True)
    output_image_dir = f"{args.output_dir}/images"
    os.makedirs(output_image_dir, exist_ok=True)

    coco_data_list = []
    for labelme_dir in tqdm(os.listdir(args.labelme_dir), desc="Processing labelme", leave=False):
        labelme_dir_path = os.path.join(args.labelme_dir, labelme_dir)
        coco_data = labelme2real(labelme_dir_path, output_image_dir)
        coco_data_list.append(coco_data)
    
    merged_data = merge_coco(coco_data_list)
    train_data, val_data = split_dataset(merged_data)
    os.makedirs(f"{args.output_dir}/annotations", exist_ok=True)
    save_coco(train_data, f"{args.output_dir}/annotations/instances_train2024.json")
    save_coco(val_data, f"{args.output_dir}/annotations/instances_val2024.json")

    os.makedirs(output_image_dir, exist_ok=True)
    output_image_dir = f"{args.output_dir}/train2024"
    copy_images(train_data['images'], [f"{args.output_dir}/images"], output_image_dir)
    output_image_dir = f"{args.output_dir}/val2024"
    copy_images(val_data['images'], [f"{args.output_dir}/images"], output_image_dir)

    os.system(f"cp bags_processing/classes.txt {args.output_dir}/annotations/classes.txt")

if __name__ == "__main__":
    main()

# python3 labelme2real.py --labelme_dir bags_processing/d435_images --output_dir Kaohsiung_Port_dataset
# python3 labelme2real.py --labelme_dir bags_processing/label --output_dir Tainan_Lifebuoy_dataset
