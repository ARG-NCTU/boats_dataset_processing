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

## Robotx2022 Unity dataset Processing
Enter directory for unity dataset processing:
```bash
cd ~/boats_dataset_processing/Boat_dataset_unity
```

First prepare your data or download all multiview images and segmentations:
```bash
mkdir -p Images
cd Images
wget ftp://140.113.148.83/arg-projectfile-download/unity_dataset/unity_multiview_data/boats1-13.zip
unzip boats1-13.zip
rm boats1-13.zip
wget ftp://140.113.148.83/arg-projectfile-download/unity_dataset/unity_multiview_data/boats14-22.zip
unzip boats14-22.zip
rm boats14-22.zip
cd ..
```

### Step1: Find Stitching Homagraphy 1 and 2 matrix

```bash
python3 step1_stitching_processing.py
```

### Step2: Apply to Stitching

```bash
python3 step2_stitching_processing.py
```

### Remove original "Images" folder and replaced with "stitched_results"

```bash
rm -rf Images
mv stitched_results Images
```


