import os
import json
import cv2
import numpy as np

import matplotlib.pyplot as plt

# Define colors for each class
COLORS = {
    'Civilian_ship': (0, 0, 255),  # Red
    'Buoy': (57, 255, 20),         # Neon Green
    'Warship': (0, 0, 0),          # Black
    'Coast_guard': (255, 0, 0)     # Blue
}

def load_labelme_annotation(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def draw_bboxes(image, annotations):
    for shape in annotations['shapes']:
        points = shape['points']
        points = np.array(points, dtype=np.int32)
        # Draw the bounding box with the corresponding color
        cv2.rectangle(image, tuple(points[0]), tuple(points[1]), COLORS[shape['label']], 2)

    return image

def visualize_bboxes(image_dir, annotation_dir):
    os.makedirs('Visualization', exist_ok=True)
    output_dir = os.path.join('Visualization', image_dir.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    # sort key is the json file name
    for filename in sorted(os.listdir(annotation_dir), key=lambda x: int(x.split('.')[0])):
        if filename.endswith('.json'):
            json_path = os.path.join(annotation_dir, filename)
            annotations = load_labelme_annotation(json_path)
            
            image_path = os.path.join(image_dir, annotations['imagePath'].split('/')[-1])
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image {annotations['imagePath']} not found.")
                continue
            
            image_with_bboxes = draw_bboxes(image, annotations)

            # Save the image with bboxes
            output_path = os.path.join(output_dir, annotations['imagePath'].split('/')[-1])
            cv2.imwrite(output_path, image_with_bboxes)
            
            # plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.cvtColor(image_with_bboxes, cv2.COLOR_BGR2RGB))
            # plt.title(annotations['imagePath'])
            # plt.axis('off')
            # plt.show()

def main(args):
    for sub_dir in os.listdir(args.root_dir):
        sub_dir = os.path.join(args.root_dir, sub_dir)
        visualize_bboxes(sub_dir, sub_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize bounding boxes on images")
    parser.add_argument("--root_dir", type=str, default="d435_images", help="Directory containing images and LabelMe annotations")
    args = parser.parse_args()
    main(args)

# Run the script
# python3 visualize_bbox.py --root_dir ~/boats_dataset_processing/bags_processing/Kaohsiung_Port_dataset_labelme/d435_images