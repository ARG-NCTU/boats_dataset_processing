# Split image folder to N folders with folders name as folder_1, folder_2, folder_3, ..., folder_N

import os
import shutil
import argparse

def split_image_folder(input_dir, output_dir, num_folders):
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_dir_name = os.path.basename(output_dir)

    # Create subdirectories
    for i in range(num_folders):
        os.makedirs(os.path.join(output_dir, f"{output_dir_name}_{i + 1}"), exist_ok=True)

    # Split images into subdirectories
    for i, file in enumerate(sorted(os.listdir(input_dir), key=lambda x: int(x.split(".")[0]))):
        if file.endswith(".png"):
            shutil.move(os.path.join(input_dir, file), os.path.join(output_dir, f"{output_dir_name}_{(i % num_folders) + 1}", file))

    print(f"Split complete. Images are saved in {output_dir}.")

def main(args):
    split_image_folder(args.input_dir, args.output_dir, args.num_folders)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split images in a directory into N subdirectories.")
    parser.add_argument("--input_dir", type=str, default="~/boats_dataset_processing/bags_processing/d435_images/2024-11-01-15-23-17_left", help="Directory containing images to split.")
    parser.add_argument("--output_dir", type=str, default="~/boats_dataset_processing/videos_processing/2024-11-01-15-23-17_left/2024-11-01-15-23-17_left", help="Directory to save split images.")
    parser.add_argument("--num_folders", type=int, default=100, help="Number of subdirectories to split images into.")
    args = parser.parse_args()
    main(args)

# Example usage:
# python3 split_image_folder.py --input_dir ~/boats_dataset_processing/bags_processing/d435_images/2024-11-01-15-23-17_left --output_dir ~/boats_dataset_processing/videos_processing/2024-11-01-15-23-17_left/2024-11-01-15-23-17_left --num_folders 100