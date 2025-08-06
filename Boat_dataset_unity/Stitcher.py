#!/usr/bin/env python3

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import time
import gc

class Setting():
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def check_and_make_dir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def load_images(self, start, end, suffix):
        images = []
        for i in range(start, end + 1):
            if args.suffix is None:
                img_path = os.path.join(self.input_dir, f"{i}.png")
            else:
                img_path = os.path.join(self.input_dir, f"{i}{suffix}.png")
            if not os.path.exists(img_path):
                break
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(image)
                else:
                    print(f"Failed to load image: {img_path}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                return None
        return images

    def split_image_into_thirds(self, image):
        height, width = image.shape[:2]
        third_width = width // 3

        left = image[:, :third_width]
        mid = image[:, third_width:2 * third_width]
        right = image[:, 2 * third_width:]

        return left, mid, right

    def file_setting(self, start, end, suffix=None):
        all_images = self.load_images(start, end, suffix=suffix)
        if all_images is None:
            return None, None, None, None

        if not all_images:
            print(f"No valid images loaded for batch {start} to {end}.")
            return [], [], [], None

        left_images, mid_images, right_images = [], [], []
        for idx, image in enumerate(all_images):
            try:
                left, mid, right = self.split_image_into_thirds(image)
                left_images.append(left)
                mid_images.append(mid)
                right_images.append(right)
            except Exception as e:
                print(f"Error processing image {start + idx}: {e}")

        if args.h1_path is None:
            output_images_dir = os.path.join(self.output_dir, 'images')
        else:
            output_images_dir = self.output_dir
        self.check_and_make_dir(output_images_dir)

        return left_images, mid_images, right_images, output_images_dir


class Stitcher():
    def __init__(self):
        pass

    def remove_black_border(self, img):
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold the image to create a binary mask
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # Find contours in the binary mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the largest contour which will be the stitched area
        max_area = 0
        best_rect = (0, 0, img.shape[1], img.shape[0])
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)
        x, y, w, h = best_rect
        return img[y:y+h, x:x+w]

    def linearBlending(self, img_left, img_right):
        # Find the dimensions of the final blended image
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]
        height = max(h1, h2)
        width = max(w1, w2)

        # Create new images with the size of the final image
        img_left_large = np.zeros((height, width, 3), dtype=np.uint8)
        img_right_large = np.zeros((height, width, 3), dtype=np.uint8)

        # Place the images onto the large canvases
        img_left_large[:h1, :w1] = img_left
        img_right_large[:h2, :w2] = img_right

        # Create an overlap mask
        overlap_mask = np.logical_and(
            np.any(img_left_large != 0, axis=2),
            np.any(img_right_large != 0, axis=2)
        )

        # Initialize the alpha mask
        alpha_mask = np.zeros((height, width), dtype=np.float32)

        # Compute the alpha mask for blending
        for i in range(height):
            overlap_indices = np.where(overlap_mask[i])[0]
            if len(overlap_indices) > 0:
                minIdx = overlap_indices[0]
                maxIdx = overlap_indices[-1]
                if maxIdx > minIdx:
                    alpha = np.linspace(1, 0, maxIdx - minIdx + 1)
                    alpha_mask[i, minIdx:maxIdx+1] = alpha

        # Convert alpha_mask to 3 channels
        alpha_mask_3c = np.dstack([alpha_mask]*3)

        # Perform linear blending
        blended = (img_left_large * alpha_mask_3c + img_right_large * (1 - alpha_mask_3c)).astype(np.uint8)

        # Handle non-overlapping regions
        blended[~overlap_mask] = img_left_large[~overlap_mask] + img_right_large[~overlap_mask]

        return blended

    def stitching(self, img_left, img_right, flip=False, H=None, save_H_path=None):
        try:
            # Use the provided homography matrix if available; otherwise, compute it
            if H is None:
                # print('SIFT Feature Detection and Matching...')
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(img_left, None)
                kp2, des2 = sift.detectAndCompute(img_right, None)

                # Use BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                # Apply ratio test as per Lowe's paper
                good_matches = []
                src_pts = []
                dst_pts = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                        src_pts.append(kp1[m.queryIdx].pt)
                        dst_pts.append(kp2[m.trainIdx].pt)

                src_pts = np.float32(src_pts)
                dst_pts = np.float32(dst_pts)

                # print('Estimating Homography...')
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                # Save the homography matrix if a path is provided and it was newly computed
                if save_H_path is not None and H is not None:
                    np.save(save_H_path, H)
                    print(f"Homography matrix saved at {save_H_path}")
            # else:
            #     print("Using provided homography matrix.")

            # print('Warping Images...')
            warping_start = time.time()

            # Warp the right image to align with the left image
            height_left, width_left = img_left.shape[:2]
            height_right, width_right = img_right.shape[:2]
            panorama_size = (width_left + width_right, max(height_left, height_right))
            img_right_warped = cv2.warpPerspective(img_right, H, panorama_size)

            # Place the left image onto the panorama canvas
            panorama = np.zeros_like(img_right_warped)
            panorama[0:height_left, 0:width_left] = img_left

            # print(f'Warping took {time.time() - warping_start:.2f} seconds.')

            # print('Blending Images...')
            blending_start = time.time()
            blended = self.linearBlending(panorama, img_right_warped)
            # print(f'Blending took {time.time() - blending_start:.2f} seconds.')

            if flip:
                blended = cv2.flip(blended, 1)

            # print('Cropping Result...')
            cropping_start = time.time()
            cropped_result = blended
            # cropped_result = self.remove_black_border(blended)
            # print(f'Cropping took {time.time() - cropping_start:.2f} seconds.')

            return cropped_result

        except Exception as e:
            print(f"Error in stitching: {e}")
            return None


def create_video_from_images(images_dir, video_path, fps=30):
    images = sorted([img for img in os.listdir(images_dir) if img.endswith(".png")], key=lambda x: int(x.split(".")[0]))
    if not images:
        print("No images found in the directory for video creation.")
        return
    
    # Read the first image to get the width and height
    frame = cv2.imread(os.path.join(images_dir, images[0]))
    height, width, _ = frame.shape

    # Define the codec and create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is a codec for .mp4
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(images_dir, image))
        out.write(frame)

    out.release()
    print(f"Video saved at {video_path}")

def process_in_batches(batch_size, total_images, setting, intermediate_dir, homography_dir, args, stitcher):
    for batch_start in tqdm(range(0, total_images, batch_size), desc="Processing batches", leave=False):
        batch_end = min(batch_start + batch_size, total_images)
        # print(f"Processing batch {batch_start + 1} to {batch_end}")

        left_images, base_images, right_images, output_images_dir = setting.file_setting(batch_start + 1, batch_end, args.suffix)
        if left_images is None or base_images is None or right_images is None or output_images_dir is None:
            continue

        num_images = min(len(left_images), len(base_images), len(right_images))
        if num_images == 0:
            print(f"No images found for batch {batch_start + 1} to {batch_end}. Skipping.")
            continue

        for i in range(num_images):
            try:
                # print(f"Processing image set {batch_start + i + 1}: "
                #       f"Left({left_images[i].shape}), "
                #       f"Base({base_images[i].shape}), "
                #       f"Right({right_images[i].shape})")
                
                H1 = np.load(args.h1_path) if args.h1_path is not None and os.path.exists(args.h1_path) else None
                H2 = np.load(args.h2_path) if args.h2_path is not None and os.path.exists(args.h2_path) else None

                img_left = cv2.flip(base_images[i], 1)
                img_right = cv2.flip(left_images[i], 1)

                LM_img = stitcher.stitching(
                    img_left, img_right, flip=True, H=H1, save_H_path=(None if H1 is not None else os.path.join(homography_dir, f"H1_{batch_start + i + 1}.npy"))
                )

                if LM_img is None:
                    print(f"Skipping stitching for image set {batch_start + i + 1}.")
                    continue

                intermediate_path = os.path.join(intermediate_dir, f"{batch_start + i + 1}.png")
                if args.h1_path is None:
                    cv2.imwrite(intermediate_path, LM_img)

                img_left = LM_img
                img_right = right_images[i]

                final_image = stitcher.stitching(
                    img_left, img_right, flip=False, H=H2, save_H_path=(None if H2 is not None else os.path.join(homography_dir, f"H2_{batch_start + i + 1}.npy"))
                )

                if final_image is None:
                    print(f"Skipping final stitching for image set {batch_start + i + 1}.")
                    continue
                
                if args.suffix is None:
                    final_path = os.path.join(output_images_dir, f"{batch_start + i + 1}.png")
                else:
                    final_path = os.path.join(output_images_dir, f"{batch_start + i + 1}{args.suffix}.png")
                cv2.imwrite(final_path, final_image)

            except Exception as e:
                print(f"Error processing image set {batch_start + i + 1}: {e}")

        del left_images, base_images, right_images
        gc.collect()

def main(args):
    setting = Setting(args.input_dir, args.output_dir)
    stitcher = Stitcher()
    
    total_images = 2001

    intermediate_dir = os.path.join(args.output_dir, 'intermediate')
    homography_dir = os.path.join(args.output_dir, 'homography')
    if args.h1_path is None:
        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(homography_dir, exist_ok=True)
    
    batch_size = 1
    process_in_batches(batch_size, total_images, setting, intermediate_dir, homography_dir, args, stitcher)
    
    # Create video from stitched images
    if args.h1_path is None:
        video_path = os.path.join(args.output_dir, 'video', 'stitched_result.mp4')
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        create_video_from_images(os.path.join(args.output_dir, 'images'), video_path, fps=30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stitch images from left, mid, and right folders.")
    parser.add_argument('--input_dir', type=str, default="Images/boats1-13/Scene1", help="Directory for input images")
    parser.add_argument('--suffix', type=str, default=None, help="Suffix of input images")
    parser.add_argument('--output_dir', type=str, default="stitched_images/boats1-13/Scene1", help="Output directory for images, video, and homographies")
    parser.add_argument('--h1_path', type=str, default=None, help="Path to save H1 homography matrix")
    parser.add_argument('--h2_path', type=str, default=None, help="Path to save H2 homography matrix")
    args = parser.parse_args()
    main(args)

# python3 Stitcher.py --input_dir Images/boats1-13/Scene1 --output_dir stiched_results/h1_h2

