#!/bin/bash

# Stop if any command fails
set -e

# === Login ===
echo "🔑 Logging into Hugging Face..."
huggingface-cli login --token "$HUGGINGFACE_TOKEN"

# === Config with Defaults ===
DATASET_COCO_REPO="${1:-ARG-NCTU/TW_Marine_2cls_dataset_coco}"
DATASET_DIR="${2:-TW_Marine_2cls_dataset}"

echo "📥 Downloading from Hugging Face dataset repo:"
echo "  DATASET_COCO_REPO = $DATASET_COCO_REPO"
echo "  DATASET_DIR       = $DATASET_DIR"

# === Create dataset directory ===
mkdir -p ~/boats_dataset_processing/"$DATASET_DIR"
cd ~/boats_dataset_processing/"$DATASET_DIR"

# === Download files ===
echo "⬇️ Downloading annotations and class labels..."
huggingface-cli download "$DATASET_COCO_REPO" instances_train2024.json --repo-type=dataset --local-dir=./annotations
huggingface-cli download "$DATASET_COCO_REPO" instances_val2024.json --repo-type=dataset --local-dir=./annotations
huggingface-cli download "$DATASET_COCO_REPO" classes.txt --repo-type=dataset --local-dir=./annotations
huggingface-cli download "$DATASET_COCO_REPO" train2024.zip --repo-type=dataset --local-dir=.
huggingface-cli download "$DATASET_COCO_REPO" val2024.zip --repo-type=dataset --local-dir=.

# === Unzip images ===
echo "📂 Unzipping image files..."
unzip -o train2024.zip
unzip -o val2024.zip

# === Cleanup ===
echo "🧹 Cleaning up zip files..."
rm train2024.zip val2024.zip

echo "✅ COCO-format dataset download complete."
cd ~/boats_dataset_processing
