#!/usr/bin/env python3
"""
SAM 3 Video Demo Script
=======================

Demo SAM 3's text-prompt video segmentation capability.

Usage:
    python src/demo.py                           # Use default video & prompt
    python src/demo.py --video path/to/video.mp4 --prompt "person"
    python src/demo.py --prompt "bed, pillow"    # Multiple objects
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def check_environment():
    """Check CUDA and dependencies."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)


def run_image_demo(image_path: str, prompt: str, output_dir: str = "output"):
    """Run SAM3 on a single image with text prompt."""
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    print(f"\n🖼️  Image Demo")
    print(f"   Image: {image_path}")
    print(f"   Prompt: '{prompt}'")
    
    # Load model
    print("\n⏳ Loading SAM 3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("✅ Model loaded!")
    
    # Load image
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    
    # Run inference with text prompt
    print(f"\n⏳ Running inference with prompt: '{prompt}'...")
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]
    
    print(f"✅ Found {len(masks)} objects!")
    for i, score in enumerate(scores):
        print(f"   Object {i+1}: confidence = {score:.3f}")
    
    # Visualize results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    visualize_image_results(image_path, masks, boxes, scores, prompt, output_dir)
    
    return masks, boxes, scores


def run_video_demo(video_path: str, prompt: str, output_dir: str = "output", max_frames: int = 30):
    """Run SAM3 on video with text prompt."""
    from sam3.model_builder import build_sam3_video_predictor
    
    print(f"\n🎬 Video Demo")
    print(f"   Video: {video_path}")
    print(f"   Prompt: '{prompt}'")
    print(f"   Max frames: {max_frames}")
    
    # Load model
    print("\n⏳ Loading SAM 3 video predictor...")
    video_predictor = build_sam3_video_predictor()
    print("✅ Model loaded!")
    
    # Start session
    print("\n⏳ Starting video session...")
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    print(f"✅ Session started: {session_id}")
    
    # Add text prompt on first frame
    print(f"\n⏳ Adding prompt '{prompt}' on frame 0...")
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        )
    )
    
    outputs = response.get("outputs", {})
    print(f"✅ Prompt added!")
    
    # Get number of objects found
    if outputs:
        num_objects = len(outputs.get("obj_ids", []))
        print(f"   Found {num_objects} object(s) matching '{prompt}'")
        
        # Print confidence scores if available
        if "scores" in outputs:
            for i, score in enumerate(outputs["scores"]):
                print(f"   Object {i+1}: confidence = {score:.3f}")
    
    # Propagate through video
    print(f"\n⏳ Propagating masks through video...")
    response = video_predictor.handle_request(
        request=dict(
            type="propagate",
            session_id=session_id,
        )
    )
    
    all_results = response.get("outputs", {})
    print(f"✅ Propagation complete!")
    
    # Save visualization
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_video = Path(output_dir) / f"demo_{Path(video_path).stem}_{prompt.replace(' ', '_')}.mp4"
    
    visualize_video_results(video_path, all_results, str(output_video), max_frames)
    
    return all_results


def visualize_image_results(image_path, masks, boxes, scores, prompt, output_dir):
    """Save visualization of image segmentation results."""
    import cv2
    import numpy as np
    
    # Load image
    image = cv2.imread(image_path)
    overlay = image.copy()
    
    # Draw masks
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        color = colors[i % len(colors)]
        
        # Convert mask to numpy if needed
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        if mask.ndim == 3:
            mask = mask[0]
        
        # Apply mask overlay
        mask_bool = mask > 0.5
        overlay[mask_bool] = color
        
        # Draw bounding box
        if torch.is_tensor(box):
            box = box.cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{prompt}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Blend overlay
    result = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    
    # Save
    output_path = Path(output_dir) / f"demo_{Path(image_path).stem}_{prompt.replace(' ', '_')}.jpg"
    cv2.imwrite(str(output_path), result)
    print(f"\n📁 Saved result to: {output_path}")


def visualize_video_results(video_path, results, output_path, max_frames=30):
    """Save visualization of video segmentation results."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    frame_idx = 0
    frames_to_process = min(max_frames, total_frames)
    
    print(f"⏳ Processing {frames_to_process} frames...")
    
    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        overlay = frame.copy()
        
        # Get masks for this frame
        frame_results = results.get(frame_idx, {})
        masks = frame_results.get("masks", [])
        
        # Draw masks
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            
            if mask.ndim == 3:
                mask = mask[0]
            
            # Resize mask if needed
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height))
            
            mask_bool = mask > 0.5
            overlay[mask_bool] = color
        
        # Blend
        result = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        # Add frame counter
        cv2.putText(result, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(result)
        frame_idx += 1
        
        if frame_idx % 10 == 0:
            print(f"   Processed {frame_idx}/{frames_to_process} frames...")
    
    cap.release()
    out.release()
    
    print(f"\n📁 Saved result video to: {output_path}")
    print(f"   You can view it with: xdg-open {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM 3 Demo")
    parser.add_argument("--video", type=str, 
                        default="/app/third_party/sam3/assets/videos/bedroom.mp4",
                        help="Path to video file")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file (if provided, runs image demo instead)")
    parser.add_argument("--prompt", type=str, default="bed",
                        help="Text prompt for segmentation")
    parser.add_argument("--output", type=str, default="/app/output",
                        help="Output directory")
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Maximum frames to process for video")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check environment, don't run demo")
    
    args = parser.parse_args()
    
    check_environment()
    
    if args.check_only:
        return
    
    try:
        if args.image:
            run_image_demo(args.image, args.prompt, args.output)
        else:
            run_video_demo(args.video, args.prompt, args.output, args.max_frames)
        
        print("\n" + "=" * 60)
        print("✅ Demo complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
