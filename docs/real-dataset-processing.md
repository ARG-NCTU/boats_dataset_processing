# Boats Dataset Processing
This repo is used for converting Robotx2022-Unity-dataset format to COCO format, and further to HugingFace dataset format.

## Setup
Clone this repo:
```bash
cd ~/ && git clone git@github.com:ARG-NCTU/boats_dataset_processing.git
```

Enter this repo:
```bash
cd ~/boats_dataset_processing
```

Enter / Pull Docker environment:
```bash
source cpu_run.sh
```

## Usage

### Extract ROS Bag

```bash
cd ~/boats_dataset_processing/bags_processing
```

Stitched cameras, 1.0x acceleration rate, compressed images
```bash
python3 extract_bags.py \
--bag_dir bags \
--output_image_dir images \
--output_video_dir videos \
--accelerate_rate 1.0 \
--topic /camera_pano_stitched/color/image_raw/compressed \
--output_names _stitched \
--compressed
```

Another example usage: 3 cameras, 5.0x acceleration rate, compressed images
```bash
python3 extract_bags.py \
--bag_dir bags \
--output_image_dir images \
--output_video_dir videos \
--accelerate_rate 5.0 \
--topic /camera1_fix/color/image_raw/compressed /camera2_fix/color/image_raw/compressed /camera3_fix/color/image_raw/compressed \
--output_names _left _mid _right \
--compressed
```

### Labelme

```bash
labelme
```

Visualize labelme annotations

![labelme-vis](example/labelme-vis.png)

```bash
cd ~/boats_dataset_processing/bags_processing
python3 visualize_bbox.py \
--root_dir ~/boats_dataset_processing/bags_processing/images
```

### Real Dataset processing

```bash
cd ~/boats_dataset_processing
```

Convert to COCO format
```bash
python3 labelme2coco.py \
--labelme_dir bags_processing/images \
--output_dir Ball_dataset \
--classes classes/Ball_classes.txt
```

Visualize COCO format annotations

![coco-vis](example/coco-vis.png)

```bash
python3 visualize_coco.py \
--coco_json Ball_dataset/annotations/instances_train2024.json \
--image_dir Ball_dataset/images \
--output_dir Ball_dataset/Visualization
```

Append Dataset
```bash
python3 merge_coco.py \
--dataset1 TW_Marine_2cls_dataset \
--dataset2 Ball_dataset \
--output_dir TW_Marine_5cls_dataset
```

Convert to HuggingFace parquet format
```bash
python3 coco2parquet.py \
--input_dir TW_Marine_5cls_dataset/annotations \
--image_dir TW_Marine_5cls_dataset/images \
--output_dir TW_Marine_5cls_dataset_hf/annotations
```

Add TW_Marine_5cls_dataset/annotations/classes.txt file and edit this like:
```bash
Buoy
GuardBoat
RedBall
YellowBall
GreenBall
```

Upload HuggingFace dataset
```bash
source upload_hf.sh ARG-NCTU/TW_Marine_5cls_dataset ARG-NCTU/TW_Marine_5cls_dataset_coco TW_Marine_5cls_dataset TW_Marine_5cls_dataset_hf
```
