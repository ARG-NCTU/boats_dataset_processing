# Boats Dataset Processing

## Setup
Clone this repo:
```bash
cd ~/ && git clone git@github.com:ARG-NCTU/boats_dataset_processing.git
```

Enter this repo:
```bash
cd ~/boats_dataset_processing
```

For first terminal to enter Docker environment:
```bash
source cpu_run.sh
```

For Second or more terminal to enter Docker environment:
```bash
source cpu_join.sh
```

Clean all ros1 packages
```bash
source clean_ros1_all.sh
```

Build all ros1 packages
```bash
source build_ros1_all.sh
```

## Usage

### Extract ROS Bag (if stitched image in bag)

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

### Play ROS Bag and Save images (if stitched image NOT in bag)

#### Terminal 1: ROSCORE

```bash
cd ~/boats_dataset_processing
source cpu_run.sh
source environment.sh
roscore
```

#### Terminal 2: Play bag

```bash
cd ~/boats_dataset_processing
source cpu_join.sh
source environment.sh
source rosbag/play_bag_dir.sh $HOME/boats_dataset_processing/bags/0610_JS5 $HOME/boats_dataset_processing/config/topics-raw-camera.txt
```

#### Terminal 3: Stitching

First Setup
```bash
cd ~/ && git clone git@github.com:JetSeaAI/opencv-cuda-docker.git
cd ~/opencv-cuda-docker
source docker_build.sh
exit
```

Launch cylindrical stitching
```bash
cd ~/opencv-cuda-docker
source docker_run.sh
source environment.sh 127.0.0.1 127.0.0.1
roslaunch cylindrical_processing cylindrical_stitching_JS5.launch
```

#### Terminal 4: Save Images

```bash
cd ~/boats_dataset_processing
source cpu_join.sh
source environment.sh
roslaunch image_processing save_images.launch
```

### Labelme

Use Labelme tools to label images

![labelme](example/labelme.gif)

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

Extend Dataset
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

Add TW_Marine_5cls_dataset/annotations/classes.txt file and edit this file like:
```bash
Buoy
GuardBoat
RedBall
YellowBall
GreenBall
```

Upload HuggingFace dataset

<img src="example/huggingface-dataset-example.png" alt="huggingface dataset example" width="800" height="auto" />

```bash
source upload_hf.sh ARG-NCTU/TW_Marine_5cls_dataset ARG-NCTU/TW_Marine_5cls_dataset_coco TW_Marine_5cls_dataset TW_Marine_5cls_dataset_hf
```