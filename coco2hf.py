import json
from tqdm import tqdm
import os

# Function to convert JSON data to JSONL format
def convert_json_to_jsonl(json_data):
    jsonl_output = []

    # Mapping from image id to file name
    image_id_to_path = {image["id"]: image["file_name"] for image in json_data["images"]}

    # Mapping from category id to category name
    category_id_to_name = {category["id"]: category["name"] for category in json_data["categories"]}

    # Process annotations to create jsonl lines with tqdm for progress tracking
    for annotation in tqdm(json_data["annotations"], desc="Converting Annotations"):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        jsonl_line = {
            "image_id": image_id,
            "image_path": f"images/{image_id_to_path[image_id]}",
            "width": next(image["width"] for image in json_data["images"] if image["id"] == image_id),
            "height": next(image["height"] for image in json_data["images"] if image["id"] == image_id),
            "objects": {
                "id": [annotation["id"]],
                "area": [annotation["area"]],
                "bbox": [annotation["bbox"]],
                "category": [category_id_to_name[category_id]]
            }
        }
        jsonl_output.append(json.dumps(jsonl_line))
    return jsonl_output

def transform_json_to_jsonl(input_json_path, output_jsonl_path):
    # Reading the JSON file
    with open(input_json_path, 'r') as file:
        json_data = json.load(file)

    # Converting JSON to JSONL format with tqdm for progress tracking
    jsonl_data = convert_json_to_jsonl(json_data)

    # Writing the JSONL output to a file
    with open(output_jsonl_path, 'w') as outfile:
        for line in tqdm(jsonl_data, desc="Writing JSONL"):
            outfile.write(line + '\n')

    print(f"Conversion complete. The JSONL file is saved as {output_jsonl_path}.")

os.makedirs("Boat_dataset_hf/annotations/", exist_ok=True)
transform_json_to_jsonl("Boat_dataset/annotations/instances_train2024.json", "Boat_dataset_hf/annotations/instances_train2024.jsonl")
transform_json_to_jsonl("Boat_dataset/annotations/instances_val2024.json", "Boat_dataset_hf/annotations/instances_val2024.jsonl")
