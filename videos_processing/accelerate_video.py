# accelerate video by removing frames

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def accelerate_video(video_path, output_path, frame_interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()
    print(f"Accelerated video saved to {output_path}")

def main(args):
    for video_file in tqdm(os.listdir(args.video_dir), desc="Accelerating videos"):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(args.video_dir, video_file)
            output_path = os.path.join(args.output_dir, video_file)
            os.makedirs(args.output_dir, exist_ok=True)
            accelerate_video(video_path, output_path, args.frame_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="source_video", help="Directory containing videos to accelerate")
    parser.add_argument("--output_dir", default="acc_output_videos", help="Output directory for accelerated videos")
    parser.add_argument("--frame_interval", type=int, default=2, help="Interval to skip frames")
    args = parser.parse_args()
    main(args)

# python3 accelerate_video.py --video_dir source_video --output_dir d435_videos_accelerated --frame_interval 5
# python3 accelerate_video.py --video_dir video --output_dir stitched_videos_accelerated --frame_interval 1