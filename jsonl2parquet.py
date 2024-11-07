import os
import json
import argparse
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Sequence, Image

# Function to process a single annotation file
def convert_annotation_file(annotation_file, image_dir, output_file):
    with open(annotation_file, 'r') as f:
        records = [json.loads(line) for line in f]

    # Add image data to each record
    for record in records:
        image_name = record["image_path"].split('/')[-1]
        record["image"] = PILImage.open(os.path.join(image_dir, image_name)).convert("RGB")

    # Define the schema
    features = Features({
        'image_id': Value('int32'),
        "image": Image(),  # Automatically handles image data
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
    dataset.to_parquet(output_file)
    print(f"Conversion complete. The Parquet file is saved as {output_file}.")

# Function to convert all annotation files in a directory
def convert_all_annotations(input_dir, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            annotation_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename.replace(".jsonl", ".parquet"))
            convert_annotation_file(annotation_file, image_dir, output_file)

# Main function for argument parsing
def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotation files to HuggingFace Parquet format.")
    parser.add_argument('--input_dir', default='real_lifebuoy_dataset_hf/annotations', help="Directory containing COCO JSONL files.")
    parser.add_argument('--image_dir', default='real_lifebuoy_dataset/images ', help="Directory containing COCO images.")
    parser.add_argument('--output_dir', default='real_lifebuoy_dataset_hf/annotations', help="Directory to save HuggingFace Parquet files.")
    
    args = parser.parse_args()

    # Call the function to convert all annotation files
    convert_all_annotations(args.input_dir, args.image_dir, args.output_dir)

if __name__ == '__main__':
    main()

# python3 jsonl2parquet.py --input_dir real_lifebuoy_dataset_hf/annotations --image_dir real_lifebuoy_dataset/images --output_dir real_lifebuoy_dataset_hf/annotations
