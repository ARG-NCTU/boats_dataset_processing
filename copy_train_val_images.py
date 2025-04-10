import os
import shutil
import argparse
import json
from merge_real_virtual import copy_images

def main():
    parser = argparse.ArgumentParser(description="Copy train and val images to separate directories.")
    
    # Define arguments
    parser.add_argument("--dir", type=str, default="real_lifebuoy_dataset")
    
    args = parser.parse_args()

    train_images = []
    val_images = []
    with open(f"{args.dir}/annotations/instances_train2024.json", 'r') as f:
        train_data = json.load(f)
        train_images = train_data['images']
    with open(f"{args.dir}/annotations/instances_val2024.json", 'r') as f:
        val_data = json.load(f)
        val_images = val_data['images']

    os.makedirs(f"{args.dir}/train2024", exist_ok=True)
    copy_images(train_images, [f"{args.dir}/images"], f"{args.dir}/train2024")
    os.makedirs(f"{args.dir}/val2024", exist_ok=True)
    copy_images(val_images, [f"{args.dir}/images"], f"{args.dir}/val2024")

if __name__ == '__main__':
    main()

# Usage:
# python3 copy_train_val_images.py --dir real_lifebuoy_dataset
# python3 copy_train_val_images.py --dir Lifebuoy_dataset