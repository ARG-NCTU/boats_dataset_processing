# SAM3 Point Prompt API Reference

Based on: `sam3_for_sam1_task_example.ipynb`

## Overview

SAM3 supports interactive instance segmentation using point and box prompts, following the SAM1 task interface.

## Setup

```python
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# IMPORTANT: enable_inst_interactivity=True is required for point prompts
model = build_sam3_image_model(
    bpe_path=bpe_path,
    enable_inst_interactivity=True
)
processor = Sam3Processor(model)
```

## Basic Usage

### 1. Set Image

```python
from PIL import Image

image = Image.open("path/to/image.jpg")
inference_state = processor.set_image(image)
```

### 2. Point Prompt Prediction

```python
import numpy as np

# Define points: (N, 2) array of [x, y] coordinates
input_point = np.array([[520, 375]])  # Single point
# or multiple points:
# input_point = np.array([[500, 375], [1125, 625]])

# Define labels: (N,) array
# 1 = foreground (positive/include)
# 0 = background (negative/exclude)
input_label = np.array([1])  # Single positive point
# or multiple labels:
# input_label = np.array([1, 0])  # First positive, second negative

# Predict
masks, scores, logits = model.predict_inst(
    inference_state,
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # Return 3 masks, pick best by score
)

# Select best mask
best_idx = np.argmax(scores)
best_mask = masks[best_idx]  # Shape: (H, W)
```

### 3. Iterative Refinement with Mask Input

```python
# Use previous prediction's logits as hint for refinement
mask_input = logits[best_idx, :, :]  # Get logits from previous prediction

# Add more points
input_point = np.array([[500, 375], [600, 400]])
input_label = np.array([1, 1])  # Both positive

masks, scores, _ = model.predict_inst(
    inference_state,
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],  # Add batch dimension
    multimask_output=False,  # Single mask when confident
)
```

## Parameters

### `model.predict_inst()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `inference_state` | dict | State from `processor.set_image()` |
| `point_coords` | np.ndarray (N, 2) | Point coordinates [x, y] |
| `point_labels` | np.ndarray (N,) | 1=foreground, 0=background |
| `box` | np.ndarray (4,) or (B, 4) | Optional bounding box [x1, y1, x2, y2] |
| `mask_input` | np.ndarray (1, H, W) | Optional previous logits |
| `multimask_output` | bool | True=3 masks, False=1 mask |

### Returns

| Return | Type | Description |
|--------|------|-------------|
| `masks` | np.ndarray (N, H, W) | Binary masks |
| `scores` | np.ndarray (N,) | Quality scores |
| `logits` | np.ndarray (N, H, W) | Low-res logits for iteration |

## Batch Inference

```python
# Multiple images
img_batch = [image1, image2]
inference_state = processor.set_image_batch(img_batch)

# Points for each image
pts_batch = [
    np.array([[[500, 375]], [[650, 750]]]),  # Image 1: 2 objects, 1 point each
    np.array([[[400, 300]], [[630, 300]]]),  # Image 2: 2 objects, 1 point each
]
labels_batch = [
    np.array([[1], [1]]),
    np.array([[1], [1]]),
]

masks_batch, scores_batch, _ = model.predict_inst_batch(
    inference_state,
    pts_batch,
    labels_batch,
    box_batch=None,
    multimask_output=True
)
```

## Integration with HIL-AA

In our annotation tool, we use this API for interactive mask refinement:

```python
# From sam3_engine.py
def refine_mask(self, image, points, labels, mask_input=None):
    pil_image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inference_state = self._image_processor.set_image(pil_image)
    
    masks, scores, logits = self._image_model.predict_inst(
        inference_state,
        point_coords=points,
        point_labels=labels,
        mask_input=mask_input,
        multimask_output=True,
    )
    
    best_idx = np.argmax(scores)
    return masks[best_idx] > 0.5
```

## Tips

1. **Single Point Ambiguity**: With one point, SAM3 may return masks for different semantic levels (part, object, scene). Use `multimask_output=True` and select by score.

2. **Negative Points**: Use `label=0` to exclude regions. Useful for refining boundaries.

3. **Combine with Box**: Adding a bounding box can help constrain the prediction.

4. **Iterative Refinement**: Pass `logits` from previous prediction as `mask_input` for better results.
