#!/usr/bin/env python3
"""
SAM 3 Video Demo Script
=======================

Demo SAM 3's text-prompt video segmentation capability.

Usage:
    python src/demo.py                           # Use default video & prompt
    python src/demo.py --video path/to/video.mp4 --prompt "person"
    python src/demo.py --prompt "bed, pillow"    # Multiple objects
    python src/demo.py --prompt "bed" --max-frames 0  # Process ALL frames
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


def propagate_in_video(predictor, session_id):
    """Propagate masks through video using streaming API."""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def run_video_demo(video_path: str, prompt: str, output_dir: str = "output", max_frames: int = 30):
    """Run SAM3 on video with text prompt.
    
    Args:
        video_path: Path to video file
        prompt: Text prompt for segmentation
        output_dir: Output directory
        max_frames: Maximum frames to process. Use 0 or -1 for ALL frames.
    """
    from sam3.model_builder import build_sam3_video_predictor
    
    # Get total frames first
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Handle max_frames <= 0 as "all frames"
    if max_frames <= 0:
        max_frames = total_frames
        frames_info = f"ALL ({total_frames})"
    else:
        frames_info = str(max_frames)
    
    print(f"\n🎬 Video Demo")
    print(f"   Video: {video_path}")
    print(f"   Prompt: '{prompt}'")
    print(f"   Total frames: {total_frames}")
    print(f"   Max frames to process: {frames_info}")
    
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
        obj_ids = outputs.get("out_obj_ids", [])
        num_objects = len(obj_ids)
        print(f"   Found {num_objects} object(s) matching '{prompt}'")
        print(f"   Object IDs: {obj_ids}")
        
        # Print confidence scores if available
        if "out_probs" in outputs:
            for i, score in enumerate(outputs["out_probs"]):
                print(f"   Object {i+1}: confidence = {score:.3f}")
    
    # Propagate through video using streaming API
    print(f"\n⏳ Propagating masks through video...")
    outputs_per_frame = propagate_in_video(video_predictor, session_id)
    print(f"✅ Propagation complete! Got {len(outputs_per_frame)} frames")
    
    # Close session to free resources
    video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    
    # Save visualization
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_video = Path(output_dir) / f"demo_{Path(video_path).stem}_{prompt.replace(' ', '_').replace(',', '')}.mp4"
    
    visualize_video_results(video_path, outputs_per_frame, str(output_video), max_frames)
    
    # Shutdown predictor
    video_predictor.shutdown()
    
    return outputs_per_frame


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


def visualize_video_results(video_path, outputs_per_frame, output_path, max_frames=30):
    """Save visualization of video segmentation results.
    
    Args:
        video_path: Path to input video
        outputs_per_frame: Dict mapping frame_index -> outputs from SAM3
        output_path: Path for output video
        max_frames: Maximum frames to process. Use 0 or -1 for ALL frames.
    """
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
    
    # Handle max_frames <= 0 as "all frames"
    if max_frames <= 0:
        frames_to_process = total_frames
    else:
        frames_to_process = min(max_frames, total_frames)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate distinct colors for each object
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    frame_idx = 0
    
    print(f"⏳ Processing {frames_to_process} frames...")
    
    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        overlay = frame.copy()
        
        # Get outputs for this frame
        frame_outputs = outputs_per_frame.get(frame_idx, {})
        
        # Extract masks and object IDs
        masks = frame_outputs.get("out_binary_masks", [])
        obj_ids = frame_outputs.get("out_obj_ids", [])
        
        # Draw masks
        for i, mask in enumerate(masks):
            # Assign color based on object ID if available
            if i < len(obj_ids):
                color_idx = obj_ids[i] % len(colors)
            else:
                color_idx = i % len(colors)
            color = colors[color_idx]
            
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
        
        # Add frame counter and object count
        info_text = f"Frame: {frame_idx}/{frames_to_process} | Objects: {len(masks)}"
        cv2.putText(result, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
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
    parser.add_argument("--prompt", type=str, default="boy",
                        help="Text prompt for segmentation")
    parser.add_argument("--output", type=str, default="/app/data/output",
                        help="Output directory")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process. Use 0 or -1 for ALL frames.")
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
