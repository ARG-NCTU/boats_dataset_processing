import json
import os

# Read the JSON file
with open('Lifebuoy/coco_formatted_unity_rgb_data.json', 'r') as file:
    json_data = json.load(file)

# Modify "area" of each annotation to be a bbox w * h
for annotation in json_data["annotations"]:
    annotation["area"] = int(annotation["bbox"][2] * annotation["bbox"][3])

# Write the modified JSON data to a new file
with open('Lifebuoy/coco_formatted_unity_rgb_data_modified.json', 'w') as file:
    json.dump(json_data, file, indent=4)