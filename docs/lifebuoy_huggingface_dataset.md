
### 4. Upload HuggingFace dataset

#### 4.2. Lifebuoy dataset

Upload dataset
```bash
cd ~/boats_dataset_processing/Lifebuoy_dataset_hf/
huggingface-cli login

huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 README.md --repo-type=dataset --commit-message="Update README to hub"

huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 annotations/instances_train2024.parquet data/instances_train2024.parquet --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 annotations/instances_val2024.parquet data/instances_val2024.parquet --repo-type=dataset --commit-message="Upload val labels to hub"

huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 classes.txt data/classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_coco_2024 classes.txt classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"
```

Upload coco format dataset
```bash
cd ~/boats_dataset_processing/Lifebuoy_dataset
zip -r images.zip images/
huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 ./images.zip ./data/images.zip --repo-type=dataset --commit-message="Upload all images to hub"
zip -r train2024.zip train2024/
huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_coco_2024 ./annotations ./annotations --repo-type=dataset --commit-message="Upload training and val labels to hub"
zip -r train2024.zip train2024/
huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_coco_2024 ./train2024.zip ./train2024.zip --repo-type=dataset --commit-message="Upload training images to hub"
zip -r val2024.zip val2024/
huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_coco_2024 ./val2024.zip ./val2024.zip --repo-type=dataset --commit-message="Upload val images to hub"
```

#### 4.3. Lifebuoy underwater dataset

Upload dataset
```bash
cd ~/boats_dataset_processing/Lifebuoy_underwater_dataset_hf/
huggingface-cli login

huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 README.md --repo-type=dataset --commit-message="Update README to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 annotations/instances_train2024.parquet data/instances_train2024.parquet --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 annotations/instances_val2024.parquet data/instances_val2024.parquet --repo-type=dataset --commit-message="Upload val labels to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 classes.txt data/classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_coco_2024 classes.txt classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"
```

Upload coco format dataset
```bash
cd ~/boats_dataset_processing/Lifebuoy_underwater_dataset
zip -r images.zip images/
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 ./images.zip ./data/images.zip --repo-type=dataset --commit-message="Upload all images to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_coco_2024 ./annotations ./annotations --repo-type=dataset --commit-message="Upload training and val labels to hub"
zip -r train2024.zip train2024/
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_coco_2024 ./train2024.zip ./train2024.zip --repo-type=dataset --commit-message="Upload training images to hub"
zip -r val2024.zip val2024/
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_coco_2024 ./val2024.zip ./val2024.zip --repo-type=dataset --commit-message="Upload val images to hub"
```

#### 4.4. Real Lifebuoy dataset

Upload dataset
```bash
cd ~/boats_dataset_processing/real_lifebuoy_dataset_hf/
huggingface-cli login

huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_2024 README.md --repo-type=dataset --commit-message="Update README to hub"
huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_2024 annotations/instances_train2024.parquet data/instances_train2024.parquet --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_2024 annotations/instances_val2024.parquet data/instances_val2024.parquet --repo-type=dataset --commit-message="Upload val labels to hub"

huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_2024 classes.txt data/classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"
huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_coco_2024 classes.txt classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"
```

Upload coco format dataset
```bash
cd ~/boats_dataset_processing/real_lifebuoy_dataset
zip -r images.zip images/
huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_2024 ./images.zip ./data/images.zip --repo-type=dataset --commit-message="Upload all images to hub"
huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_coco_2024 ./annotations ./annotations --repo-type=dataset --commit-message="Upload training and val labels to hub"
zip -r train2024.zip train2024/
huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_coco_2024 ./train2024.zip ./train2024.zip --repo-type=dataset --commit-message="Upload training images to hub"
zip -r val2024.zip val2024/
huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_coco_2024 ./val2024.zip ./val2024.zip --repo-type=dataset --commit-message="Upload val images to hub"
```

#### 4.5. Kaohsiung Port dataset

Upload dataset
```bash
cd ~/boats_dataset_processing/Kaohsiung_Port_dataset_hf/
huggingface-cli login

huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_2024 README.md --repo-type=dataset --commit-message="Update README to hub"
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_2024 annotations/instances_train2024.parquet data/instances_train2024.parquet --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_2024 annotations/instances_val2024.parquet data/instances_val2024.parquet --repo-type=dataset --commit-message="Upload val labels to hub"

```

Upload coco format dataset
```bash
cd ~/boats_dataset_processing/Kaohsiung_Port_dataset
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_coco_2024 ./annotations ./ --repo-type=dataset --commit-message="Upload training and val labels and classes to hub"
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_coco_2024 ./train2024.zip ./train2024.zip --repo-type=dataset --commit-message="Upload training images to hub"
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_coco_2024 ./val2024.zip ./val2024.zip --repo-type=dataset --commit-message="Upload val images to hub"
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_coco_2024 ./images.zip ./images.zip --repo-type=dataset --commit-message="Upload training and val images to hub"
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_2024 ./images.zip ./data/images.zip --repo-type=dataset --commit-message="Upload training and val images to hub"
zip -r images.zip images/
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_2024 ./images.zip ./data/images.zip --repo-type=dataset --commit-message="Upload all images to hub"
zip -r train2024.zip train2024/
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_coco_2024 ./train2024.zip ./train2024.zip --repo-type=dataset --commit-message="Upload training images to hub"
zip -r val2024.zip val2024/
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_coco_2024 ./val2024.zip ./val2024.zip --repo-type=dataset --commit-message="Upload val images to hub"
```

#### 4.6. Kaohsiung Port Buoy dataset

Upload dataset
```bash
cd ~/boats_dataset_processing/KS_Buoy_dataset_hf/
huggingface-cli login

huggingface-cli upload ARG-NCTU/KS_Buoy_dataset_2025 README.md --repo-type=dataset --commit-message="Update README to hub"
huggingface-cli upload ARG-NCTU/KS_Buoy_dataset_2025 annotations/instances_train2024.parquet data/instances_train2024.parquet --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/KS_Buoy_dataset_2025 annotations/instances_val2024.parquet data/instances_val2024.parquet --repo-type=dataset --commit-message="Upload val labels to hub"

```

Upload coco format dataset
```bash
cd ~/boats_dataset_processing/KS_Buoy_dataset
huggingface-cli upload ARG-NCTU/KS_Buoy_dataset_coco_2025 ./annotations ./ --repo-type=dataset --commit-message="Upload training and val labels and classes to hub"
huggingface-cli upload ARG-NCTU/KS_Buoy_dataset_2025 ./annotations/classes.txt ./data/classes.txt --repo-type=dataset --commit-message="Upload classes to hub"
sudo apt update
sudo apt install zip
zip -r images.zip images/
huggingface-cli upload ARG-NCTU/KS_Buoy_dataset_2025 ./images.zip ./data/images.zip --repo-type=dataset --commit-message="Upload all images to hub"
zip -r train2024.zip train2024/
huggingface-cli upload ARG-NCTU/KS_Buoy_dataset_coco_2025 ./train2024.zip ./train2024.zip --repo-type=dataset --commit-message="Upload training images to hub"
zip -r val2024.zip val2024/
huggingface-cli upload ARG-NCTU/KS_Buoy_dataset_coco_2025 ./val2024.zip ./val2024.zip --repo-type=dataset --commit-message="Upload val images to hub"
```

#### 4.7. GuarBoat dataset

Upload dataset
```bash
cd ~/boats_dataset_processing/GuardBoat_dataset_hf/
huggingface-cli login

huggingface-cli upload ARG-NCTU/GuardBoat_dataset_2025 README.md --repo-type=dataset --commit-message="Update README to hub"
huggingface-cli upload ARG-NCTU/GuardBoat_dataset_2025 annotations/instances_train2025.parquet data/instances_train2025.parquet --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/GuardBoat_dataset_2025 annotations/instances_val2025.parquet data/instances_val2025.parquet --repo-type=dataset --commit-message="Upload val labels to hub"

```

Upload coco format dataset
```bash
cd ~/boats_dataset_processing/GuardBoat_dataset
huggingface-cli upload ARG-NCTU/GuardBoat_dataset_coco_2025 ./annotations ./ --repo-type=dataset --commit-message="Upload training and val labels and classes to hub"
huggingface-cli upload ARG-NCTU/GuardBoat_dataset_2025 ./annotations/classes.txt ./data/classes.txt --repo-type=dataset --commit-message="Upload classes to hub"
sudo apt update
sudo apt install zip
zip -r images.zip images/
huggingface-cli upload ARG-NCTU/GuardBoat_dataset_2025 ./images.zip ./data/images.zip --repo-type=dataset --commit-message="Upload all images to hub"
zip -r train2025.zip train2025/
huggingface-cli upload ARG-NCTU/GuardBoat_dataset_coco_2025 ./train2025.zip ./train2025.zip --repo-type=dataset --commit-message="Upload training images to hub"
zip -r val2025.zip val2025/
huggingface-cli upload ARG-NCTU/GuardBoat_dataset_coco_2025 ./val2025.zip ./val2025.zip --repo-type=dataset --commit-message="Upload val images to hub"
```

### 5. Download HuggingFace dataset

#### 5.2. Lifebuoy dataset

Download dataset
```bash
huggingface-cli download ARG-NCTU/Lifebuoy_dataset_2024 --include "*.parquet" --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
huggingface-cli download ARG-NCTU/Lifebuoy_dataset_2024 data/classes.txt --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
```

#### 5.3. Lifebuoy underwater dataset

Download dataset
```bash
huggingface-cli download ARG-NCTU/Lifebuoy_underwater_dataset_2024 --include "*.parquet" --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
huggingface-cli download ARG-NCTU/Lifebuoy_underwater_dataset_2024 data/classes.txt --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
```

### 6. Use HuggingFace dataset for training DETR

Clone or pull huggingface notebook repo
```bash
cd ~/
git clone git@github.com:ARG-NCTU/huggingface-notebooks.git
cd ~/huggingface-notebooks/
source gpu_run.sh
cd ~/huggingface-notebooks/transformers_doc/en/pytorch
```

Download HuggingFace dataset:
```bash
huggingface-cli login
huggingface-cli download ARG-NCTU/Boat_dataset_2024 --include "*.parquet" --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
huggingface-cli download ARG-NCTU/Boat_dataset_2024 data/classes.txt --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
huggingface-cli download ARG-NCTU/Boat_dataset_2024 data/images.zip --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
```

Use Boat Dataset for object detection model training example:
```bash
cd ~/huggingface-notebooks/
source jupyter_notebook.sh 
```

Ctrl + click the website link. 
You can start edit jupyter notebook in transformers_doc/en/pytorch.

Another way is run python script:
```bash
cd ~/huggingface-notebooks/transformers_doc/en/pytorch
huggingface-cli login
python3 python3 train_detr_lifebuoy.py
```

### 7. Upload testing video to hub

```bash
huggingface-cli login
cd ~/boats_dataset_processing/Lifebuoy_dataset_hf/
huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 ./lifebuoy_detection.mp4 ./video/lifebuoy_detection.mp4 --repo-type=dataset --commit-message="Upload testing video to hub"

cd ~/boats_dataset_processing/Lifebuoy_underwater_dataset_hf/
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 ./lifebuoy_underwater_detection.mp4 ./video/lifebuoy_underwater_detection.mp4 --repo-type=dataset --commit-message="Upload testing video to hub"

cd ~/boats_dataset_processing/real_lifebuoy_dataset_hf/
huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_2024 ./detr_finetuned_2.mp4 ./video/detr_finetuned_2.mp4 --repo-type=dataset --commit-message="Upload testing video to hub"
huggingface-cli upload ARG-NCTU/Real_Lifebuoy_dataset_2024 ./detr_finetuned_1vs2.mp4 ./video/detr_finetuned_1vs2.mp4 --repo-type=dataset --commit-message="Upload testing video to hub"

cd ~/boats_dataset_processing/Kaohsiung_Port_dataset_hf/
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 ./1107_kaohsiung_port1_out.mp4 ./video/1107_kaohsiung_port1_out.mp4 --repo-type=dataset --commit-message="Upload testing video to hub"
huggingface-cli upload ARG-NCTU/Kaohsiung_Port_dataset_2024 ./1107_kaohsiung_port2_out.mp4 ./video/1107_kaohsiung_port2_out.mp4 --repo-type=dataset --commit-message="Upload testing video to hub"

```

```bash
huggingface-cli login
cd ~/boats_dataset_processing/JetSeaAIBoat/
huggingface-cli upload JetSea-AI/JetSeaAIBoat ./images/20241212_120307 ./images/20241212_120307 --repo-type=dataset --commit-message="Upload images to hub"

huggingface-cli upload JetSea-AI/JetSeaAIBoat ./images/20241212_123729 ./images/20241212_123729 --repo-type=dataset --commit-message="Upload images to hub"

huggingface-cli upload JetSea-AI/JetSeaAIBoat ./images/20241212_130904 ./images/20241212_130904 --repo-type=dataset --commit-message="Upload images to hub"

huggingface-cli upload JetSea-AI/JetSeaAIBoat ./images/20241212_163302 ./images/20241212_163302 --repo-type=dataset --commit-message="Upload images to hub"
```
