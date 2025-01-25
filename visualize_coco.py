import os
import json
import cv2
import random
import argparse
from tqdm import tqdm

def visualize_coco_annotations(coco_json, image_dir, output_dir=None):
    """
    Visualize COCO annotations on images.
    
    Args:
        coco_json (str): Path to COCO annotation JSON file.
        image_dir (str): Directory containing images referenced in the JSON.
        output_dir (str, optional): Directory to save visualized images. If None, images won't be saved.
        show (bool, optional): If True, display the images with annotations using OpenCV.
    """
    # Load COCO data
    with open(coco_json, "r") as f:
        coco_data = json.load(f)

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create a mapping from image_id to image metadata
    image_id_to_metadata = {img["id"]: img for img in coco_data["images"]}

    # Create a mapping from category_id to category name
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Visualize each annotation
    for annotation in tqdm(coco_data["annotations"], desc="Visualizing annotations"):
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]

        # Get image metadata
        image_metadata = image_id_to_metadata[image_id]
        image_path = os.path.join(image_dir, image_metadata["file_name"])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found or invalid: {image_path}")
            continue

        # Get bbox coordinates and category name
        x, y, w, h = map(int, bbox)
        category_name = category_id_to_name[category_id]

        # Draw the bbox and label on the image
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"{category_name}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save or display the image
        if output_dir:
            output_path = os.path.join(output_dir, image_metadata["file_name"])
            cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO annotations on images.")
    parser.add_argument("--coco_json", type=str, default="Tainan_Lifebuoy_dataset/annotations/instances_train2024.json", help="Path to COCO annotation JSON file.")
    parser.add_argument("--image_dir", type=str, default="Tainan_Lifebuoy_dataset/train2024", help="Directory containing images.")
    parser.add_argument("--output_dir", type=str, default="Tainan_Lifebuoy_dataset/Visualization", help="Directory to save visualized images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    visualize_coco_annotations(args.coco_json, args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()

# Example usage:
# python visualize_coco.py --coco_json Tainan_Lifebuoy_dataset/annotations/instances_train2024.json --image_dir Tainan_Lifebuoy_dataset/train2024 --output_dir Tainan_Lifebuoy_dataset/Visualization
# python visualize_coco.py --coco_json Tainan_Lifebuoy_dataset/annotations/instances_val2024.json --image_dir Tainan_Lifebuoy_dataset/val2024 --output_dir Tainan_Lifebuoy_dataset/Visualization

