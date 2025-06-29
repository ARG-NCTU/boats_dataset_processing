import pandas as pd
import argparse
import os

def visualize_parquet_annotations(parquet_path):
    df = pd.read_parquet(parquet_path)
    print(df.head())

def main():
    parser = argparse.ArgumentParser(description="Visualize Parquet annotations.")
    parser.add_argument(
        "--parquet_folder",
        type=str,
        default="TW_Marine_5cls_dataset_hf/annotations",
        help="Path to folder containing the Parquet annotation file."
    )

    args = parser.parse_args()
    print("train dataset:")
    train_parquet_path = os.path.join(args.parquet_folder, "instances_train2024.parquet")
    visualize_parquet_annotations(train_parquet_path)
    print("\n\nval dataset:")
    val_parquet_path = os.path.join(args.parquet_folder, "instances_val2024.parquet")
    visualize_parquet_annotations(val_parquet_path)

if __name__ == "__main__":
    main()
