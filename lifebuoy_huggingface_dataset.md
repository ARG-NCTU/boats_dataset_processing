### 4. Upload HuggingFace dataset

#### 4.2. Lifebuoy dataset

Upload dataset
```bash
cd Lifebuoy_dataset_hf/
huggingface-cli login

huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 annotations/instances_train2024.jsonl data/instances_train2024.jsonl --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 annotations/instances_val2024.jsonl data/instances_val2024.jsonl --repo-type=dataset --commit-message="Upload val labels to hub"
```

#### 4.3. Lifebuoy underwater dataset

Upload dataset
```bash
huggingface-cli login

huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 Lifebuoy_underwater_dataset_2024.py --repo-type=dataset --commit-message="Update script to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 README.md --repo-type=dataset --commit-message="Update README to hub"

huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 annotations/instances_train2024.jsonl data/instances_train2024.jsonl --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 annotations/instances_val2024.jsonl data/instances_val2024.jsonl --repo-type=dataset --commit-message="Upload val labels to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 classes.txt data/classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"
```

### 5. Download HuggingFace dataset

#### 5.2. Lifebuoy dataset

Download dataset
```bash
huggingface-cli download ARG-NCTU/Lifebuoy_dataset_2024 --include "*.jsonl" --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
huggingface-cli download ARG-NCTU/Lifebuoy_dataset_2024 data/classes.txt --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
```

#### 5.3. Lifebuoy underwater dataset

Download dataset
```bash
huggingface-cli download ARG-NCTU/Lifebuoy_underwater_dataset_2024 --include "*.jsonl" --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
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
huggingface-cli download ARG-NCTU/Boat_dataset_2024 --include "*.jsonl" --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
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