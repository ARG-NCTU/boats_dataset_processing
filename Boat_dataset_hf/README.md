---
viewer: false
---

# Boat Dataset for Object Detection

## Overview
This dataset contains images of real & virtual boats for object detection tasks. It can be used to train and evaluate object detection models.

## Dataset Structure

### Data Instances

A data point comprises an image and its object annotations.

```

```

### Data Fields

- `image_id`: the image id
- `width`: the image width
- `height`: the image height
- `objects`: a dictionary containing bounding box metadata for the objects present on the image
  - `id`: the annotation id
  - `area`: the area of the bounding box
  - `bbox`: the object's bounding box (in the [coco](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco) format)
  - `category`: the object's category, with possible values including 
    - `BallonBoat` (0)
    - `BigBoat` (1)
    - `Boat` (2)
    - `JetSki` (3)
    - `Katamaran` (4)
    - `SailBoat` (5)
    - `SmallBoat` (6)
    - `SpeedBoat` (7)
    - `WAM_V` (8)

  
### Data Splits

- `Training dataset` (42833)
    - `Real`
        - `WAM_V` (2333)
    - `Virtual`
        - `BallonBoat` (4500)
        - `BigBoat` (4500)
        - `Boat` (4500)
        - `JetSki` (4500)
        - `Katamaran` (4500)
        - `SailBoat` (4500)
        - `SmallBoat` (4500)
        - `SpeedBoat` (4500)
        - `WAM_V` (4500)

- `Val dataset` (5400)
    - `Real` 
        - `WAM_V` (900)
    - `Virtual`
        - `BallonBoat` (500)
        - `BigBoat` (500)
        - `Boat` (500)
        - `JetSki` (500)
        - `Katamaran` (500)
        - `SailBoat` (500)
        - `SmallBoat` (500)
        - `SpeedBoat` (500)
        - `WAM_V` (500)


## Usage
```
from datasets import load_dataset
dataset = load_dataset("zhuchi76/Boat_dataset")
```

## Citation
If you use this dataset in your research, please cite the following paper:
