import os
import cv2
import numpy as np
import argparse

def visualize_seg(seg_image_path):

    # Check if the file exists
    if not os.path.exists(seg_image_path):
        print(f"File not found: {seg_image_path}")
    else:
        # Read the segmentation image
        seg_image = cv2.imread(seg_image_path, cv2.IMREAD_GRAYSCALE)

        if seg_image is None:
            print("Error: Unable to read the image. Please check the file integrity.")
        else:
            # Display the segmentation image 0: background, not 0: boat
            # Convert cv2 image to numpy array
            seg_image = np.array(seg_image)
            
            # Collect unique pixel values
            unique_values = np.unique(seg_image)
            print("Unique pixel values:", unique_values)

            # Remove max pixel value / 2 element of unique values, if it exists
            remove_value = np.max(unique_values) // 2
            if remove_value in unique_values:
                unique_values = [val for val in unique_values if val != remove_value and val != 0]
            else:
                unique_values = [val for val in unique_values if val != 0]
            
            # Map unique pixel values to color indices, excluding 0 (background)
            color_map = {val: np.random.randint(0, 255, 3) for val in unique_values}
            print("Color mapping:", color_map)

            # Create a new image with different colors
            new_image = np.zeros((seg_image.shape[0], seg_image.shape[1], 3), dtype=np.uint8)
            
            for i in range(seg_image.shape[0]):
                for j in range(seg_image.shape[1]):
                    pixel_value = seg_image[i][j]
                    if pixel_value in color_map:
                        new_image[i][j] = color_map[pixel_value]

            # Save the new image
            cv2.imwrite("seg_image.png", new_image)
            print("Image saved as 'seg_image.png'.")

def main(args):
    visualize_seg(args.seg_image_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Visualize segmentation image")
    parser.add_argument("seg_image_path", type=str, default="./Images/boats7-13/Scene1/62_seg.png", help="Path to the segmentation image")

    args = parser.parse_args()
    main(args)
