#!/usr/bin/env python3
import cv2
import json
import argparse
import os
import numpy as np

# Initialize variables with predefined corner points for a rectangular area
rect_pts = [(100, 100), (400, 100), (400, 400), (100, 400)]  # Default rectangular points
dragging = -1  # Index of the point being dragged, -1 means no point is being dragged

def click_and_drag(event, x, y, flags, param):
    global rect_pts, dragging

    # Mouse down to start dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(rect_pts):
            if abs(pt[0] - x) < 10 and abs(pt[1] - y) < 10:  # Check if click is near a point
                dragging = i
                break

    # Mouse move to drag the selected point and adjust rectangle
    elif event == cv2.EVENT_MOUSEMOVE and dragging != -1:
        update_rectangle(x, y, dragging)
        update_frame()

    # Mouse up to release the point
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = -1

def update_rectangle(x, y, index):
    global rect_pts
    if index == 0:  # Top-left
        rect_pts[0] = (x, y)
        rect_pts[1] = (rect_pts[1][0], y)           # Adjust top-right y
        rect_pts[3] = (x, rect_pts[3][1])           # Adjust bottom-left x
    elif index == 1:  # Top-right
        rect_pts[1] = (x, y)
        rect_pts[0] = (rect_pts[0][0], y)           # Adjust top-left y
        rect_pts[2] = (x, rect_pts[2][1])           # Adjust bottom-right x
    elif index == 2:  # Bottom-right
        rect_pts[2] = (x, y)
        rect_pts[1] = (x, rect_pts[1][1])           # Adjust top-right x
        rect_pts[3] = (rect_pts[3][0], y)           # Adjust bottom-left y
    elif index == 3:  # Bottom-left
        rect_pts[3] = (x, y)
        rect_pts[0] = (x, rect_pts[0][1])           # Adjust top-left x
        rect_pts[2] = (rect_pts[2][0], y)           # Adjust bottom-right y

def update_frame():
    # Redraw the frame with circles at points and lines between them
    temp_frame = frame.copy()
    for i in range(len(rect_pts)):
        # Draw circles on each point
        cv2.circle(temp_frame, rect_pts[i], 5, (0, 255, 0), -1)
        # Draw lines between consecutive points
        if i > 0:
            cv2.line(temp_frame, rect_pts[i-1], rect_pts[i], (0, 255, 0), 2)
    # Draw line closing the rectangle
    cv2.line(temp_frame, rect_pts[3], rect_pts[0], (0, 255, 0), 2)
    cv2.imshow("Video", temp_frame)

def save_cropped_info(json_path, crop_info):
    with open(json_path, 'w') as json_file:
        json.dump(crop_info, json_file)
    print(f"Crop info saved to {json_path}: {crop_info}")

def crop_video(cap, output_path, json_path, rect_pts, fps):
    # Convert points to an ordered quadrilateral for perspective transformation
    pts_src = np.array(rect_pts, dtype="float32")
    width = int(abs(rect_pts[1][0] - rect_pts[0][0]))
    height = int(abs(rect_pts[3][1] - rect_pts[0][1]))

    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    transform_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    crop_info = {
        "points": rect_pts,
        "width": width,
        "height": height
    }
    save_cropped_info(json_path, crop_info)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        warped_frame = cv2.warpPerspective(frame, transform_matrix, (width, height))
        out.write(warped_frame)

    out.release()
    print("Cropped video has been saved.")

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    global frame
    cap = cv2.VideoCapture(args.input)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", click_and_drag)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Display the frame with current points and lines
        update_frame()
        key = cv2.waitKey(1) & 0xFF

        # If 'Enter' key is pressed and four points are set, crop the video
        if key == 13 and len(rect_pts) == 4:
            crop_video(cap, args.output, args.json, rect_pts, fps)
            break

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video to show various processing results in quadrants.")
    parser.add_argument('-i', '--input', type=str, default="stitched_videos_accelerated/2024-11-01-14-33-57_stitched_result.mp4", help='Path to the input video file')
    parser.add_argument('-o', '--output', type=str, default="stitched_videos_accelerated/2024-11-01-14-33-57_stitched_result_cropped.mp4", help='Path to the output cropped video file')
    parser.add_argument('-j', '--json', type=str, default="stitched_videos_accelerated/2024-11-01-14-33-57_stitched_result_cropped.json", help='Path to the output cropped info file')
    args = parser.parse_args()
    main(args)
