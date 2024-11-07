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
{
    'image_id': 1, 
    'image_path': 'images/boats7-13_Scene2_blur1_2.png', 
    'width': 640, 
    'height': 480, 
    'objects': 
    {
        'id': [1, 2, 3, 4, 5], 
        'area': [101.98473358154297, 209.18804931640625, 152.91551208496094, 1432.936279296875, 1135.2135009765625], 
        'bbox': [[352.0, 238.0, 10.0, 5.0], [302.0, 240.0, 29.0, 6.0], [296.0, 236.0, 24.0, 4.0], [303.0, 219.0, 67.0, 15.0], [369.0, 232.0, 32.0, 26.0]], 
        'category': [4, 6, 7, 9, 10]
    }
}
```

### Data Fields

- `image_id`: the image id
- `image_path`: the image path
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
    - `container_ship` (9)
    - `TugShip` (10)
    - `yacht` (11)
    - `blueboat` (12)

### Suffix of the jsonl files

- rtvrr: RGB & Thermal Virtual + RGB Real
- rvrr: RGB Virtual + RGB Real
- rtv: RGB & Thermal Virtual
- rv : RGB Virtual
- tv: Thermal Virtual
- rr: RGB Real
  
### Data Splits

```
--------------------------------------------------
Totol number of classes: 13
--------------------------------------------------
Total number of images: 111197
Total number / percentage of images in training set: 88872  / (79%)
Total number / percentage of images in validation set: 22325  / (20%)
--------------------------------------------------
Total number of annotations: 501313
Total number / percentage of annotations in training set: 400703  / (79%)
Total number / percentage of annotations in validation set: 100610  / (20%)
--------------------------------------------------
Average number of annotations per image in training set: 4.508765415428932
--------------------------------------------------
Dataset class & type statistics:
Class Name           Virtual Train RGB    Virtual Val RGB      Virtual Train Thermal Virtual Val Thermal  Real Train           Real Val             (100% / Avg Row%)   
----------------------------------------------------------------------------------------------------------------------------------------------------------------
BallonBoat           30816 (66% / 9%)     7716 (16% / 9%)      6191 (13% / 9%)      1537 (3% / 9%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
BigBoat              26980 (66% / 8%)     6761 (16% / 8%)      5347 (13% / 8%)      1361 (3% / 8%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
Boat                 30279 (66% / 9%)     7581 (16% / 9%)      6071 (13% / 9%)      1540 (3% / 9%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
JetSki               22646 (66% / 6%)     5683 (16% / 6%)      4516 (13% / 6%)      1124 (3% / 6%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
Katamaran            30122 (66% / 9%)     7540 (16% / 9%)      6045 (13% / 9%)      1497 (3% / 9%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
SailBoat             28848 (66% / 8%)     7269 (16% / 8%)      5804 (13% / 8%)      1453 (3% / 8%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
SmallBoat            30986 (66% / 9%)     7789 (16% / 9%)      6217 (13% / 9%)      1565 (3% / 9%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
SpeedBoat            31610 (66% / 9%)     7846 (16% / 9%)      6327 (13% / 9%)      1578 (3% / 9%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
WAM_V                33216 (62% / 10%)    8352 (15% / 10%)     6648 (12% / 9%)      1662 (3% / 10%)      2332 (4% / 100%)     900 (1% / 100%)      (100% / 16%)
container_ship       29328 (66% / 8%)     7281 (16% / 8%)      5861 (13% / 8%)      1477 (3% / 8%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
TugShip              23711 (66% / 7%)     5926 (16% / 7%)      4693 (13% / 7%)      1148 (3% / 6%)       0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
yacht                10103 (66% / 3%)     2542 (16% / 3%)      2038 (13% / 3%)      452 (2% / 2%)        0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
blueboat             3234 (64% / 0%)      822 (16% / 0%)       734 (14% / 1%)       208 (4% / 1%)        0 (0% / 0%)          0 (0% / 0%)          (100% / 16%)
(Avg Col% / 100%)    (64% / 100%)         (16% / 100%)         (14% / 100%)         (4 % / 100%)         (0 % / 100%)         (0 % / 100%)        
----------------------------------------------------------------------------------------------------------------------------------------------------------------
meaning of the items in the table:
1. The first number is the number of annotations in the class.
2. The second number in the bracket is the percentage of the class in the row.
3. The third number in the bracket is the percentage of the class in the column.
----------------------------------------------------------------------------------------------------------------------------------------------------------------
```


## Usage
```
from datasets import load_dataset
from datasets import Features, Value, Sequence

# Specify the correct schema for your dataset
features = Features({
    'image_id': Value('int32'),
    'image_path': Value('string'),
    'width': Value('int32'),
    'height': Value('int32'),
    'objects': {
        'id': Sequence(Value('int32')),
        'area': Sequence(Value('float32')),
        'bbox': Sequence(Sequence(Value('float32'), length=4)),
        'category': Sequence(Value('int32'))
    }
})

# Load the dataset with the correct features
dataset_rtvrr = load_dataset(
    'json', 
    data_files={'train': 'data/instances_train2024_rtvrr.jsonl', 
                'validation': 'data/instances_val2024_rtvrr.jsonl'},
    features=features  # Explicitly specify the schema
)
```

## Citation

