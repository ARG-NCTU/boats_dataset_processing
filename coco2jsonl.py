import argparse
import json
from tqdm import tqdm
import os
import shutil
from PIL import Image
import numpy as np
from base64 import b64encode

# Function to convert JSON data to JSONL format
def convert_json_to_jsonl(json_data, image_dir, output_jsonl_path, batch_size=10000):
    
    # Create a dictionary to map image IDs to image objects
    images = sorted(json_data["images"], key=lambda x: x['id'])
    image_id_to_image = {image["id"]: image for image in images}
    
    anno_num = 0
    batch_count = 0

    # Process annotations in batches
    annotations = json_data["annotations"]

    # Sort annotations by image_id to ensure the order is maintained
    annotations = sorted(annotations, key=lambda x: x['image_id'])

    for i in tqdm(range(0, len(annotations), batch_size), desc="Converting Annotations in Batches"):
        batch_annotations = annotations[i:i + batch_size]

        # Initialize a variable to store the previous JSONL line
        prev_jsonl_line = None
        prev_image_id = None

        # Process each annotation in the batch
        with open(output_jsonl_path, 'a') as outfile:
            for annotation in batch_annotations:
                image_not_found = 0
                if annotation is None:
                    continue
                image_id = annotation["image_id"]

                # Skip if the image ID is not found in the images dictionary
                if image_id not in image_id_to_image:
                    # print(f"Warning: Image ID {image_id} not found in images section. Skipping annotation.")
                    image_not_found += 1
                    continue
                
                if image_id != prev_image_id:
                    # If there was a previous jsonl line, write it to file
                    if prev_jsonl_line:
                        outfile.write(json.dumps(prev_jsonl_line) + '\n')

                    image_name = image_id_to_image[image_id]["file_name"]
                    # image = Image.open(f'{image_dir}/{image_name}')
                    # image = b64encode(np.array(image).tobytes()).decode('utf-8')
                    print(f"Image ID: {image_id}, Image Name: {image_name}")
                    image_width = int(image_id_to_image[image_id]["width"])
                    image_height = int(image_id_to_image[image_id]["height"])
                    # Create a new jsonl line for the new image
                    jsonl_line = {
                        "image_id": image_id,
                        # "image": image,
                        "image_path": f'images/{image_name}',
                        "width": image_width,
                        "height": image_height,
                        "objects": {
                            "id": [annotation["id"]],
                            "area": [annotation["area"]],
                            "bbox": [annotation["bbox"]],
                            "category": [annotation["category_id"]]
                        }
                    }
                else:
                    # Append to the existing jsonl line for the same image
                    jsonl_line["objects"]["id"].append(annotation["id"])
                    jsonl_line["objects"]["area"].append(annotation["area"])
                    jsonl_line["objects"]["bbox"].append(annotation["bbox"])
                    jsonl_line["objects"]["category"].append(annotation["category_id"])

                # Update previous image ID and line
                prev_image_id = image_id
                prev_jsonl_line = jsonl_line
                anno_num += 1

            print(f"Image not found: {image_not_found}")
            print(f"Processed {len(batch_annotations) - image_not_found} annotations in batch {batch_count}.")

            # Write the last jsonl line after finishing the batch
            if prev_jsonl_line:
                outfile.write(json.dumps(prev_jsonl_line) + '\n')
            
        batch_count += 1
        

    print(f"Converted {anno_num} annotations in total.")

def transform_json_to_jsonl(input_json_path, image_dir, output_jsonl_path):
    # Reading the JSON file
    with open(input_json_path, 'r') as file:
        json_data = json.load(file)

    # Converting JSON to JSONL format with tqdm for progress tracking
    if os.path.exists(output_jsonl_path):
        os.remove(output_jsonl_path)  # Delete the file if it already exists
    convert_json_to_jsonl(json_data, image_dir, output_jsonl_path)

    print(f"Conversion complete. The JSONL file is saved as {output_jsonl_path}.")

def convert_all_annotations(input_dir, image_dir, output_dir):
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert all JSON files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_json_path = os.path.join(input_dir, filename)
            output_jsonl_path = os.path.join(output_dir, filename.replace(".json", ".jsonl"))
            print(f"Converting {filename} to {output_jsonl_path}...")
            transform_json_to_jsonl(input_json_path, image_dir, output_jsonl_path)

# Main function for argument parsing
def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotation files to HuggingFace JSONL format.")
    parser.add_argument('--input_dir', default='Boat_dataset/annotations', help="Directory containing COCO JSON files.")
    parser.add_argument('--image_dir', default='Boat_dataset/images', help="Directory containing COCO images.")
    parser.add_argument('--output_dir', default='Boat_dataset_hf/annotations', help="Directory to save HuggingFace JSONL files.")
    
    args = parser.parse_args()

    # Call the function to convert all annotation files
    convert_all_annotations(args.input_dir, args.image_dir, args.output_dir)

if __name__ == '__main__':
    main()

# Usage:
# python3 coco2jsonl.py --input_dir Boat_dataset/annotations --output_dir Boat_dataset_hf/annotations
# python3 coco2jsonl.py --input_dir Lifebuoy_dataset/annotations --image_dir Lifebuoy_dataset/images --output_dir Lifebuoy_dataset_hf/annotations
# python3 coco2jsonl.py --input_dir real_lifebuoy_dataset/annotations --output_dir real_lifebuoy_dataset_hf/annotations

# python3 coco2jsonl.py --input_dir Boat_unity_dataset/annotations --image_dir Boat_unity_dataset/images --output_dir Boat_unity_dataset_hf/annotations
