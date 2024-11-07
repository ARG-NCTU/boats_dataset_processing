import os
import cv2
import argparse

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

def main(args):
    create_video_from_images(args.images_dir, args.video_path, args.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from images in a directory.")
    parser.add_argument("--images_dir", type=str, help="Directory containing images to create video from.")
    parser.add_argument("--video_path", type=str, help="Path to save the output video.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    args = parser.parse_args()
    main(args)

# python3 png2mp4.py --images_dir ~/boats_dataset_processing/stitching/stitched_images/images --video_path stitched_videos_accelerated/2024-11-01-14-33-57_stitched_result.mp4