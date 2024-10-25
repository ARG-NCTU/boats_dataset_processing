# Filter unused json files that image path does not exist

import os
import json
import cv2

root_dir = "d435_images"
for dir in os.listdir(root_dir):
    print(dir)
    json_dir = os.path.join(root_dir, dir)
    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue
        with open(os.path.join(json_dir, json_file), "r") as f:
            data = json.load(f)
        image_name = data["imagePath"]
        image_path = os.path.join(json_dir, image_name)
        if not os.path.exists(image_path):
            print("delete", json_file)
            os.remove(os.path.join(json_dir, json_file))