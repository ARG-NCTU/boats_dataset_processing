import os
from tqdm import tqdm
import shutil

def sort_files(file):
    return file.lower()  

def load_classes():
    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

def move_images_to_folders(rgb_thermal='rgb', dest_rgb_folder='rgb_images', dest_thermal_folder='thermal_images', dest_mask_folder='masked_images', boats_root_path='~/dataset/Images', boat_count_per_scene=3002):
    """
    Moves the images to folders based on the class names   
    Args:
        boats_root_path: Path to the root directory of the dataset
        boat_count_per_scene: Number of boats per scene
    """
    
    # Create destination folders if they do not exist
    shutil.rmtree(dest_rgb_folder, ignore_errors=True)
    os.makedirs(dest_rgb_folder, exist_ok=True)
    print(f"Created folder: {dest_rgb_folder}")
    
    shutil.rmtree(dest_thermal_folder, ignore_errors=True)
    os.makedirs(dest_thermal_folder, exist_ok=True)
    print(f"Created folder: {dest_thermal_folder}")

    # shutil.rmtree(dest_mask_folder, ignore_errors=True)
    # os.makedirs(dest_mask_folder, exist_ok=True)
    # print(f"Created folder: {dest_mask_folder}")

    # Data Structure
    # boats1-6
    #   - Scene1
    #       - *.png
    #       - *.json
    #   - Scene1_blur1
    #       - *.png
    #       - *.json
    #   - Scene1_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene1_blurN
    #       - *.png
    #       - *.json
    #   - Scene2
    #       - *.png
    #       - *.json
    #   - Scene2_blur1
    #       - *.png
    #       - *.json
    #   - Scene2_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene2_blurN
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene5
    #       - *.png
    #       - *.json
    #   - Scene5_blur1
    #       - *.png
    #       - *.json
    #   - Scene5_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene5_blurN
    #       - *.png
    #       - *.json
    # boats7-12
    #   - Scene1
    #       - *.png
    #       - *.json
    #   - Scene1_blur1
    #       - *.png
    #       - *.json
    #   - Scene1_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene1_blurN
    #       - *.png
    #       - *.json
    #   - Scene2
    #       - *.png
    #       - *.json
    #   - Scene2_blur1
    #       - *.png
    #       - *.json
    #   - Scene2_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene2_blurN
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene5
    #       - *.png
    #       - *.json
    #   - Scene5_blur1
    #       - *.png
    #       - *.json
    #   - Scene5_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene5_blurN
    #       - *.png
    #       - *.json
    # N >= 1, default N = 1



    for boats_dir in tqdm(os.listdir(boats_root_path), desc="Boats", leave=False):
        # print(f"Processing {boats_dir}")
        boats_dir_path = os.path.join(boats_root_path, boats_dir)
        for scene_dir in tqdm(os.listdir(boats_dir_path), desc="Scenes", leave=False):
            # print(f"Processing {scene_dir}")
            scene_dir_path = os.path.join(boats_dir_path, scene_dir)
            for i in tqdm(range(2, boat_count_per_scene - 1), desc="Images", leave=False):
                # RGB images: 1.png, 2.png, 3.png, ...
                # Thermal images: 1_thermal.png, 2_thermal.png, 3_thermal.png, ...
                # Mask images: 1_label.png, 2_label.png, 3_label.png, ...
                # JSON files: 1.main.json, 2.main.json, 3.main.json, ...

                rgb_image_path = os.path.join(scene_dir_path, f'{i}.png')
                thermal_image_path = os.path.join(scene_dir_path, f'{i}_thermal.png')
                mask_image_path = os.path.join(scene_dir_path, f'{i}_label.png')

                shutil.copy(rgb_image_path, dest_rgb_folder)
                os.rename(os.path.join(dest_rgb_folder, f'{i}.png'), os.path.join(dest_rgb_folder, f'{boats_dir}_{scene_dir}_{i}.png'))
                if 'Scene1' in scene_dir:
                    shutil.copy(thermal_image_path, dest_thermal_folder)
                    os.rename(os.path.join(dest_thermal_folder, f'{i}_thermal.png'), os.path.join(dest_thermal_folder, f'{boats_dir}_{scene_dir}_{i}_thermal.png'))
                # shutil.copy(mask_image_path, dest_mask_folder)
                # if rgb_thermal=='rgb':
                #     os.rename(os.path.join(dest_mask_folder, f'{i}_label.png'), os.path.join(dest_mask_folder, f'{boats_dir}_{scene_dir}_{i}.png'))
                # elif rgb_thermal=='thermal':
                #     os.rename(os.path.join(dest_mask_folder, f'{i}_label.png'), os.path.join(dest_mask_folder, f'{boats_dir}_{scene_dir}_{i}_thermal.png'))
                # else:
                #     raise ValueError("Invalid value for rgb_thermal")
                

move_images_to_folders(boats_root_path='Images', boat_count_per_scene=3002)