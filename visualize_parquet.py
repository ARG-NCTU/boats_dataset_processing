import pandas as pd
import argparse
import os

def visualize_parquet_annotations(parquet_path):
    df = pd.read_parquet(parquet_path)

    # 設定不截斷欄位
    pd.set_option("display.max_colwidth", None)

    # 取前 5 筆
    df_first5 = df.iloc[:5].copy()

    # 縮短顯示的 image / image_path
    df_first5["image"] = df_first5["image"].apply(lambda x: str(x)[:15] + "..." if len(str(x)) > 15 else str(x))
    df_first5["image_path"] = df_first5["image_path"].apply(lambda x: x[:15] + "..." if len(x) > 15 else x)

    # 印出 image_id, image, image_path, objects
    print(df_first5[["image_id", "image", "image_path", "objects"]])

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
