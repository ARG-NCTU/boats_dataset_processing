#!/bin/bash

# Stop if any command fails
set -e

# === Config with Defaults ===
DATASET_HF_REPO="${1:-ARG-NCTU/TW_Marine_2cls_dataset}"
DATASET_COCO_REPO="${2:-ARG-NCTU/TW_Marine_2cls_dataset_coco}"
DATASET_DIR="${3:-TW_Marine_2cls_dataset}"
DATASET_HF_DIR="${4:-TW_Marine_2cls_dataset_hf}"

echo "🛠 Using configuration:"
echo "  DATASET_HF_REPO   = $DATASET_HF_REPO"
echo "  DATASET_COCO_REPO = $DATASET_COCO_REPO"
echo "  DATASET_DIR       = $DATASET_DIR"
echo "  DATASET_HF_DIR    = $DATASET_HF_DIR"

# === Login ===
echo "🔑 Logging into Hugging Face..."
huggingface-cli login --token "$HUGGINGFACE_TOKEN"

# === Upload HF parquet dataset ===
echo "⬆️ Uploading Parquet-based dataset to $DATASET_HF_REPO..."
cd ~/boats_dataset_processing/"$DATASET_HF_DIR"

huggingface-cli upload "$DATASET_HF_REPO" annotations/instances_train2024.parquet data/instances_train2024.parquet \
  --repo-type=dataset \
  --commit-message="Upload training labels to hub"

huggingface-cli upload "$DATASET_HF_REPO" annotations/instances_val2024.parquet data/instances_val2024.parquet \
  --repo-type=dataset \
  --commit-message="Upload val labels to hub"

huggingface-cli upload "$DATASET_HF_REPO" annotations/instances_test2024.parquet data/instances_test2024.parquet \
  --repo-type=dataset \
  --commit-message="Upload test labels to hub"

# === Upload COCO dataset ===
echo "⬆️ Uploading COCO-format dataset to $DATASET_COCO_REPO..."
cd ~/boats_dataset_processing/"$DATASET_DIR"

huggingface-cli upload "$DATASET_COCO_REPO" ./annotations ./ \
  --repo-type=dataset \
  --commit-message="Upload training and val labels and classes to hub"

huggingface-cli upload "$DATASET_HF_REPO" ./annotations/classes.txt ./data/classes.txt \
  --repo-type=dataset \
  --commit-message="Upload classes to hub"

# === Zip and upload images ===
cd ~/boats_dataset_processing/"$DATASET_DIR"
echo "📦 Zipping and uploading image folders..."
zip -r images.zip images/
huggingface-cli upload "$DATASET_HF_REPO" ./images.zip ./data/images.zip \
  --repo-type=dataset \
  --commit-message="Upload all images to hub"

zip -r train2024.zip train2024/
huggingface-cli upload "$DATASET_COCO_REPO" ./train2024.zip ./train2024.zip \
  --repo-type=dataset \
  --commit-message="Upload training images to hub"

zip -r val2024.zip val2024/
huggingface-cli upload "$DATASET_COCO_REPO" ./val2024.zip ./val2024.zip \
  --repo-type=dataset \
  --commit-message="Upload val images to hub"

zip -r test2024.zip test2024/
huggingface-cli upload "$DATASET_COCO_REPO" ./test2024.zip ./test2024.zip \
  --repo-type=dataset \
  --commit-message="Upload test images to hub"

echo "✅ All uploads complete."
cd ~/boats_dataset_processing
