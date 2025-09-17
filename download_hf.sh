#!/bin/bash

# Stop if any command fails
set -e

# === Config with Defaults ===
DATASET_HF_REPO="${1:-ARG-NCTU/TW_Marine_2cls_dataset}"
DATASET_COCO_REPO="${2:-ARG-NCTU/TW_Marine_2cls_dataset_coco}"
DATASET_DIR="${3:-TW_Marine_2cls_dataset}"
DATASET_HF_DIR="${4:-TW_Marine_2cls_dataset_hf}"
FORMAT="${5:-parquet}"   # default parquet，can change to jsonl

echo "🛠 Using configuration:"
echo "  DATASET_HF_REPO   = $DATASET_HF_REPO"
echo "  DATASET_COCO_REPO = $DATASET_COCO_REPO"
echo "  DATASET_DIR       = $DATASET_DIR"
echo "  DATASET_HF_DIR    = $DATASET_HF_DIR"
echo "  FORMAT            = $FORMAT"

# === Login ===
echo "🔑 Logging into Hugging Face..."
huggingface-cli login --token "$HUGGINGFACE_TOKEN"

# === Download HF dataset (labels) ===
echo "⬇️ Downloading $FORMAT-based dataset from $DATASET_HF_REPO..."
mkdir -p ~/boats_dataset_processing/"$DATASET_HF_DIR"
cd ~/boats_dataset_processing/"$DATASET_HF_DIR"

if [ "$FORMAT" = "jsonl" ]; then
  huggingface-cli download "$DATASET_HF_REPO" data/instances_train2024.jsonl --repo-type=dataset --local-dir .
  huggingface-cli download "$DATASET_HF_REPO" data/instances_val2024.jsonl   --repo-type=dataset --local-dir .
  huggingface-cli download "$DATASET_HF_REPO" data/instances_test2024.jsonl  --repo-type=dataset --local-dir .
else
  huggingface-cli download "$DATASET_HF_REPO" data/instances_train2024.parquet --repo-type=dataset --local-dir .
  huggingface-cli download "$DATASET_HF_REPO" data/instances_val2024.parquet   --repo-type=dataset --local-dir .
  huggingface-cli download "$DATASET_HF_REPO" data/instances_test2024.parquet  --repo-type=dataset --local-dir .
fi

# === Download COCO dataset ===
echo "⬇️ Downloading COCO-format dataset from $DATASET_COCO_REPO..."
mkdir -p ~/boats_dataset_processing/"$DATASET_DIR"/annotations
cd ~/boats_dataset_processing/"$DATASET_DIR"/annotations

huggingface-cli download "$DATASET_COCO_REPO" ./instances_train2024.json --repo-type=dataset --local-dir .
huggingface-cli download "$DATASET_COCO_REPO" ./instances_val2024.json   --repo-type=dataset --local-dir .
huggingface-cli download "$DATASET_COCO_REPO" ./instances_test2024.json  --repo-type=dataset --local-dir .
huggingface-cli download "$DATASET_COCO_REPO" ./classes.txt --repo-type=dataset --local-dir .

# === Download image zips ===
cd ~/boats_dataset_processing/"$DATASET_DIR"
echo "⬇️ Downloading image archives..."
huggingface-cli download "$DATASET_HF_REPO" data/images.zip --repo-type=dataset --local-dir .
huggingface-cli download "$DATASET_COCO_REPO" ./train2024.zip --repo-type=dataset --local-dir .
huggingface-cli download "$DATASET_COCO_REPO" ./val2024.zip   --repo-type=dataset --local-dir .
huggingface-cli download "$DATASET_COCO_REPO" ./test2024.zip  --repo-type=dataset --local-dir .

# === Unzip images if exist ===
for z in images.zip train2024.zip val2024.zip test2024.zip; do
  if [ -f "$z" ]; then
    echo "📦 Unzipping $z..."
    unzip -o "$z"
  fi
done

echo "✅ All downloads complete."
cd ~/boats_dataset_processing
