import json
import argparse
import os

def convert_jsonl_to_coco(input_dir, output_dir):
    # Category mapping (Example)
    category_mapping = {0: "category_name"}  # Update with actual category ids and names
    category_id_map = {cat_id: idx + 1 for idx, cat_id in enumerate(category_mapping.keys())}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each JSONL file in the input directory
    for jsonl_filename in os.listdir(input_dir):
        if not jsonl_filename.endswith('.jsonl'):
            continue
        
        jsonl_path = os.path.join(input_dir, jsonl_filename)
        output_file = os.path.join(output_dir, jsonl_filename.replace('.jsonl', '.json'))
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": []
        }

        # Add categories to COCO format
        for cat_id, cat_name in category_mapping.items():
            coco_data["categories"].append({
                "id": category_id_map[cat_id],
                "name": cat_name,
                "supercategory": "none"
            })

        annotation_id = 1

        # Process each line in JSONL
        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line)

                # Add image information from JSONL file directly
                image_id = record['image_id']
                image_name = record['image_path'].split('/')[-1]
                width = record['width']
                height = record['height']

                coco_data["images"].append({
                    "id": image_id,
                    "file_name": image_name,
                    "width": width,
                    "height": height
                })

                # Add annotations
                for obj_id, area, bbox, category in zip(
                        record['objects']['id'], 
                        record['objects']['area'], 
                        record['objects']['bbox'], 
                        record['objects']['category']):
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id_map[category],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        # Save to COCO JSON
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=4)
        
        print(f"Conversion complete for {jsonl_filename}. The COCO JSON file is saved as {output_file}.")

# Main function for argument parsing
def main():
    parser = argparse.ArgumentParser(description="Convert JSONL files in a directory to COCO JSON format.")
    parser.add_argument('--input_dir', default="real_lifebuoy_dataset_hf/annotations", help="Directory containing JSONL files.")
    parser.add_argument('--output_dir', default="real_lifebuoy_dataset/annotations", help="Directory to save COCO JSON files.")

    args = parser.parse_args()

    # Call the conversion function
    convert_jsonl_to_coco(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()

# python3 jsonl2json.py --input_dir real_lifebuoy_dataset_hf/annotations --output_dir real_lifebuoy_dataset/annotations