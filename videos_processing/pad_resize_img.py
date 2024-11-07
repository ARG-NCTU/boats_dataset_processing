import os
import cv2
from tqdm import tqdm

def pad_resize_image(input_dir, output_dir, target_size):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Pad to square and resize all PNG files in the input directory
    for folder in tqdm(os.listdir(input_dir), desc="Padding and Resizing", leave=False):
        for file in tqdm(os.listdir(f"{input_dir}/{folder}"), desc=f"Processing {folder}", leave=False):
            if file.endswith(".png"):
                image = cv2.imread(os.path.join(input_dir, folder, file))
                h, w, _ = image.shape
                max_dim = max(h, w)
                top = (max_dim - h) // 2
                bottom = max_dim - h - top
                left = (max_dim - w) // 2
                right = max_dim - w - left
                padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                resized_image = cv2.resize(padded_image, (target_size, target_size))
                cv2.imwrite(f"{output_dir}/{folder}/{file}", resized_image)

    print(f"Padding and resizing complete. PNG files are saved in {output_dir}.")

def main(args):
    pad_resize_image(args.input_dir, args.output_dir, args.target_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pad and resize images to square.")
    parser.add_argument("--input_dir", type=str, default="d435_images", help="Directory containing PNG files.")
    parser.add_argument("--output_dir", type=str, default="d435_images", help="Directory to save padded and resized images.")
    parser.add_argument("--target_size", type=int, default=800, help="Target size for padding and resizing.")
    args = parser.parse_args()
    main(args)