from datasets import Dataset, Features, Value, Image, Sequence
import argparse
import json
from tqdm import tqdm
import os
import shutil
from PIL import Image as PILImage
import pandas as pd

# Function to convert JSON data to JSONL format
def convert_json_to_parquet(json_data, image_dir, output_parquet_path, batch_size=10000):
    
    # Create a dictionary to map image IDs to image objects
    images = sorted(json_data["images"], key=lambda x: x['id'])
    image_id_to_image = {image["id"]: image for image in images}
    
    anno_num = 0
    batch_count = 0

    # Process annotations in batches
    annotations = json_data["annotations"]

    # Sort annotations by image_id to ensure the order is maintained
    annotations = sorted(annotations, key=lambda x: x['image_id'])

    # Prepare lists to collect data for each annotation
    records = []

    for i in tqdm(range(0, len(annotations), batch_size), desc="Converting Annotations in Batches"):
        batch_annotations = annotations[i:i + batch_size]

        # Initialize a variable to store the previous JSONL line
        prev_record = None
        prev_image_id = None

        # Process each annotation in the batch
        with open(output_parquet_path, 'a') as outfile:
            for annotation in tqdm(batch_annotations, desc="Processing Annotations", leave=False):
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
                    # If there was a previous record, append it to the list
                    if prev_record:
                        records.append(prev_record)

                    image_name = image_id_to_image[image_id]["file_name"]
                    image = PILImage.open(f'{image_dir}/{image_name}').convert("RGB") # Convert to CMYK if required
                    # print(f"Image ID: {image_id}, Image Name: {image_name}")
                    image_width = int(image_id_to_image[image_id]["width"])
                    image_height = int(image_id_to_image[image_id]["height"])
                    # Create a new jsonl line for the new image
                    record = {
                        "image_id": image_id,
                        "image": image,
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
                    # Append to the existing record for the same image
                    record["objects"]["id"].append(annotation["id"])
                    record["objects"]["area"].append(annotation["area"])
                    record["objects"]["bbox"].append(annotation["bbox"])
                    record["objects"]["category"].append(annotation["category_id"])

                # Update previous image ID and line
                prev_image_id = image_id
                prev_record = record
                anno_num += 1

            print(f"Image not found: {image_not_found}")
            print(f"Processed {len(batch_annotations) - image_not_found} annotations in batch {batch_count}.")

            # Write the last record to the file
            if prev_record:
                records.append(prev_record)
            
        batch_count += 1
        
    # Specify the correct schema for your dataset
    features = Features({
        'image_id': Value('int32'),
        "image": Image(),
        'image_path': Value('string'),
        'width': Value('int32'),
        'height': Value('int32'),
        'objects': Features({
            'id': Sequence(Value('int32')),
            'area': Sequence(Value('float32')),  
            'bbox': Sequence(Sequence(Value('float32'), length=4)), 
            'category': Sequence(Value('int32'))
        })
    })

    # Convert list of records to Dataset with image data
    dataset = Dataset.from_list(records, features=features)
    
    # Save as Parquet file
    dataset.to_parquet(output_parquet_path)
    print(f"Conversion complete. The Parquet file is saved as {output_parquet_path}.")



def transform_json_to_parquet(input_json_path, image_dir, output_parquet_path):
    # Reading the JSON file
    with open(input_json_path, 'r') as file:
        json_data = json.load(file)

    # Converting JSON to JSONL format with tqdm for progress tracking
    if os.path.exists(output_parquet_path):
        os.remove(output_parquet_path)  # Delete the file if it already exists
    convert_json_to_parquet(json_data, image_dir, output_parquet_path)

    print(f"Conversion complete. The JSONL file is saved as {output_parquet_path}.")

def convert_all_annotations(input_dir, image_dir, output_dir):
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert all JSON files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_json_path = os.path.join(input_dir, filename)
            output_parquet_path = os.path.join(output_dir, filename.replace(".json", ".parquet"))
            print(f"Converting {filename} to {output_parquet_path}...")
            transform_json_to_parquet(input_json_path, image_dir, output_parquet_path)

# Main function for argument parsing
def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotation files to HuggingFace Parquet format.")
    parser.add_argument('--input_dir', default='Boat_dataset/annotations', help="Directory containing COCO JSON files.")
    parser.add_argument('--image_dir', default='Boat_dataset/images', help="Directory containing COCO images.")
    parser.add_argument('--output_dir', default='Boat_dataset_hf/annotations', help="Directory to save HuggingFace Parquet files.")
    
    args = parser.parse_args()

    # Call the function to convert all annotation files
    convert_all_annotations(args.input_dir, args.image_dir, args.output_dir)

if __name__ == '__main__':
    main()

# Usage:
# python3 coco2parquet.py --input_dir Boat_dataset/annotations --image_dir Boat_dataset/images --output_dir Boat_dataset_hf/annotations

# python3 coco2parquet.py --input_dir Lifebuoy_dataset/annotations --image_dir Lifebuoy_dataset/images --output_dir Lifebuoy_dataset_hf/annotations
# python3 coco2parquet.py --input_dir Lifebuoy_underwater_dataset/annotations --image_dir Lifebuoy_underwater_dataset/images --output_dir Lifebuoy_underwater_dataset_hf/annotations
# python3 coco2parquet.py --input_dir real_lifebuoy_dataset/annotations --image_dir real_lifebuoy_dataset/images --output_dir real_lifebuoy_dataset_hf/annotations
# python3 coco2parquet.py --input_dir Kaohsiung_Port_dataset/annotations --image_dir Kaohsiung_Port_dataset/images --output_dir Kaohsiung_Port_dataset_hf/annotations
