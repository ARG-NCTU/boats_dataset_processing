# Convert mp4 to png and padding, resizing to 800x800

import cv2
import os
import shutil
from tqdm import tqdm
import argparse

def convert_mp4_to_png(input_dir, output_dir):
    # Create output directory if it doesn't exist
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    
    # Convert all MP4 files in the input directory
    for file in tqdm(os.listdir(input_dir), desc="Converting MP4 to PNG", leave=False):
        if file.endswith(".MOV") or file.endswith(".mp4"):
            file_name = file.split(".")[0]
            os.makedirs(f"{output_dir}/{file_name}", exist_ok=True)
            video = cv2.VideoCapture(os.path.join(input_dir, file))
            success, image = video.read()
            count = 0
            img_count = 1
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            dismiss = 0
            max_dismiss = 7.5
            while success:
                if dismiss < max_dismiss:
                    dismiss += 1
                    success, image = video.read()
                    continue
                print(f"Processing {file_name}: {count}/{total_frames}", end="\r")
                dismiss = 0
                # Save frame as PNG file
                cv2.imwrite(f"{output_dir}/{file_name}/{file_name}_{img_count}.png", image)
                success, image = video.read()
                count += max_dismiss
                img_count += 1

    print(f"Conversion complete. PNG files are saved in {output_dir}.")

def main(args):
    convert_mp4_to_png(args.input_dir, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images and from videos files.")
    parser.add_argument("--input_dir", type=str, default="d435_videos", help="Directory containing video files.")
    parser.add_argument("--output_dir", type=str, default="d435_images", help="Directory to save extracted images.")
    args = parser.parse_args()
    main(args)

# Usage
# python3 mp42png.py --input_dir k180_0610/used --output_dir k180_0610/used_images