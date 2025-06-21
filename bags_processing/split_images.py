import os
import shutil

def split_png_into_five_folders(src_folder, dest_root, folder_names):
    # 取得所有 png 檔案並排序
    png_files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith('.png')])

    # 檢查是否有足夠的檔案
    if len(png_files) < 5:
        print("檔案數不足五份，請確認資料夾內容。")
        return

    # 建立目的資料夾
    os.makedirs(dest_root, exist_ok=True)
    for name in folder_names:
        os.makedirs(os.path.join(dest_root, name), exist_ok=True)

    # 平均分成五份
    chunk_size = len(png_files) // 5
    remainder = len(png_files) % 5

    start_idx = 0
    for i in range(5):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        chunk = png_files[start_idx:end_idx]
        dest_dir = os.path.join(dest_root, folder_names[i])
        for f in chunk:
            shutil.copy(os.path.join(src_folder, f), os.path.join(dest_dir, f))
        start_idx = end_idx

    print("已成功將 PNG 分成五份並儲存至對應資料夾。")

# 使用範例
src_folder = 'ball_images'  # 修改為你的 PNG 檔來源資料夾
dest_root = 'ball_images_split'  # 所有資料夾的根目錄
folder_names = ['Adam', 'Brian', 'Josh', 'Wilbur', 'Arthur']  # 自訂五個資料夾名稱

split_png_into_five_folders(src_folder, dest_root, folder_names)
