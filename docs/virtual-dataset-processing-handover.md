# Boats Dataset Processing
This repo is used for converting Robotx2022-Unity-dataset format to COCO format, and further to HugingFace dataset format.

## Clone repo

```bash
cd ~/ && git clone git@github.com:ARG-NCTU/boats_dataset_processing.git
```

## Setting HuggingFace token

```bash
vim ~/.bashrc
```

Go to HuggingFace Web page: this [link](https://huggingface.co/settings/tokens) to add your own token

Then add this line (Replace with your token):
```bash
export HUGGINGFACE_TOKEN=hf_...xxxx
```

## Enter the repo

```bash
cd ~/boats_dataset_processing
```

## Enter Docker Environment

For first terminal to enter Docker environment:
```bash
source cpu_run.sh
```

For Second or more terminal to enter Docker environment:
```bash
source cpu_join.sh
```

### 1. Robotx2022 Unity dataset Processing
Enter directory for unity dataset processing:
```bash
cd ~/boats_dataset_processing/Boat_dataset_unity
```

Download all example stitched images and segmentations:
```bash
mkdir -p Images
cd Images
wget ftp://140.113.148.83/arg-projectfile-download/unity_dataset/unity_stitched_data_scene1/boats1-13.zip
unzip boats1-13.zip
rm boats1-13.zip
wget ftp://140.113.148.83/arg-projectfile-download/unity_dataset/unity_stitched_data_scene1/boats14-22.zip
unzip boats14-22.zip
rm boats14-22.zip
cd ..
```

#### 1.1. Robotx2022 Unity dataset Visualization
Print segmentation (You can modify the segementation image path)
```bash
python3 visualize_seg.py --seg_image_path ./Images/boats1-13/Scene1/1969_seg.png
```

Example result

![image](example/seg_image_stitched.png)

#### 1.2. Gaussian Blur
Apply gaussian blur (You can modify the level of gaussian blur and images folders)
```bash
python3 gaussian_blur.py --datadir Images --blur_level 1
```

Example results
- Source Image

![image](example/boats1-13_scene1_1969.png)

- Blur Level 1 Image

![image](example/boats1-13_scene1_1969_blur1.png)


#### 1.3. Convert Robotx2022 Unity dataset format to COCO format
Convert to COCO format (You can first modify the mask_ids.json and classes.txt then modify obscure rate which is default to 30%)
```bash
python3 merge_json_unity.py \
--rgb_thermal rgb \
--boats_root_path Images \
--boat_count_per_scene 2002 \
--output Boats1-22/coco_formatted_unity_rgb_data.json
```
Labels should save in json files.

Example results
- Source Image with bbox annotations

![image](example/boats1-13_scene1_1969_anno.png)


Copy rgb & thermal images
```bash
python3 copy_images.py \
--boats_root_path Images \
--dest_rgb_folder Boats1-22/rgb_images \
--boat_count_per_scene 2002
```

### 2. Split COCO train/val/test annotations
Enter directory for unity dataset processing:
```bash
cd ~/boats_dataset_processing
```

Split COCO train/val/test annotations
```bash
python3 merge_real_virtual.py \
--unity_rgb_only \
--unity_dataset_dir Boat_dataset_unity/Boats1-22 \
--output_dir Boat_unity_example
```

Simple annotations for statistic analysis:
```bash
python3 statistic_class_simple.py \
--train_json Boat_unity_example/annotations/instances_train2024.json \
--val_json Boat_unity_example/annotations/instances_val2024.json \
--test_json Boat_unity_example/annotations/instances_test2024.json
```

Example output:
```bash
--------------------------------------------------
Total number of classes: 9
Total number of images: 7996
Training images: 5598 (70.01%)
Validation images: 1599 (20.00%)
Test images: 799 (9.99%)
Total number of annotations: 86318
Training annotations: 60517 (70.11%)
Validation annotations: 17114 (19.83%)
Test annotations: 8687 (10.06%)
Average annotations per training image: 10.81046802429439
Average annotations per validation image: 10.70293933708568
Average annotations per test image: 10.872340425531915
--------------------------------------------------
Class Name            Train Annotations    Validation Annotations    Test Annotations
--------------------------------------------------
WAM_V                2718                 734                  400                 
Hovercraft           7884                 2200                 1140                
Yacht                20798                5689                 3045                
CargoShip            2811                 764                  411                 
WorkBoat             2798                 761                  409                 
Blueboat             2316                 621                  337                 
MilitaryShip         13841                4153                 1932                
CoastGuardShip       5333                 1593                 740                 
Buoy                 2018                 599                  273                 
Class statistics saved to class_statistics.csv
```

Copy classes txt file:
```bash
cp Boat_dataset_unity/Boats1-22/classes.txt Boat_unity_example/annotations/
```

### 3. Convert COCO format to HuggingFace dataset format
Convert to HuggingFace dataset format:
```bash
python3 coco2jsonl.py \
--input_dir Boat_unity_example/annotations \
--image_dir Boat_unity_example/images \
--output_dir Boat_unity_example_hf/annotations
```


### 4. Upload HuggingFace dataset
```bash
source upload_hf.sh zhuchi76/Boat_unity_example zhuchi76/Boat_unity_example_coco Boat_unity_example Boat_unity_example_hf jsonl
```
Replace with your huggingface account name.