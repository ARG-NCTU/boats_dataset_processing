import shutil
import os

def copy_json_to_img_folder(json_dir, img_dir):
    json_files = os.listdir(json_dir)
    for json_file in json_files:
        shutil.copyfile(os.path.join(json_dir, json_file), os.path.join(img_dir, json_file))
    print(f"Copy {len(json_files)} json files to {img_dir}")

def main():
    for json_dir in os.listdir("label"):
        json_dir_path = os.path.join("label", json_dir)
        img_dir_path = os.path.join("d435_images", json_dir.replace("label_", ""))
        copy_json_to_img_folder(json_dir_path, img_dir_path)

if __name__ == "__main__":
    main()

