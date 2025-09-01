import os
from tqdm import tqdm
import shutil
import argparse

def sort_files(file):
    return file.lower()

def move_images_to_folders(
    rgb_thermal='rgb',
    dest_rgb_folder='rgb_images',
    dest_thermal_folder='thermal_images',
    dest_mask_folder='masked_images',
    boats_root_path='~/dataset/Images',
    boat_count_per_scene=3002
):
    """
    Moves the images to folders based on the class names
    Args:
        boats_root_path: Path to the root directory of the dataset
        boat_count_per_scene: Number of boats per scene
    """
    boats_root_path = os.path.expanduser(boats_root_path)

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

    # Walk boats_* / Scene* / images
    if not os.path.isdir(boats_root_path):
        raise FileNotFoundError(f"Root folder not found: {boats_root_path}")

    boats_dirs = sorted(
        [d for d in os.listdir(boats_root_path) if os.path.isdir(os.path.join(boats_root_path, d))],
        key=sort_files
    )

    for boats_dir in tqdm(boats_dirs, desc="Boats", leave=False):
        boats_dir_path = os.path.join(boats_root_path, boats_dir)
        scene_dirs = sorted(
            [d for d in os.listdir(boats_dir_path) if os.path.isdir(os.path.join(boats_dir_path, d))],
            key=sort_files
        )
        for scene_dir in tqdm(scene_dirs, desc="Scenes", leave=False):
            scene_dir_path = os.path.join(boats_dir_path, scene_dir)
            for i in tqdm(range(2, boat_count_per_scene - 1), desc="Images", leave=False):
                # Paths
                rgb_image_path = os.path.join(scene_dir_path, f'{i}.png')
                thermal_image_path = os.path.join(scene_dir_path, f'{i}_thermal.png')
                mask_image_path = os.path.join(scene_dir_path, f'{i}_label.png')

                if not os.path.exists(rgb_image_path):
                    print(f"File {rgb_image_path} does not exist. Skipping...")
                    continue

                # Copy RGB and rename to unique flat filename
                shutil.copy(rgb_image_path, dest_rgb_folder)
                src_in_dest = os.path.join(dest_rgb_folder, f'{i}.png')
                dst_name = f'{boats_dir}_{scene_dir}_{i}.png'
                os.rename(src_in_dest, os.path.join(dest_rgb_folder, dst_name))

                # If you later want thermal/mask, just uncomment your original lines below.
                # if 'Scene1' in scene_dir:
                #     shutil.copy(thermal_image_path, dest_thermal_folder)
                #     os.rename(os.path.join(dest_thermal_folder, f'{i}_thermal.png'),
                #               os.path.join(dest_thermal_folder, f'{boats_dir}_{scene_dir}_{i}_thermal.png'))
                #
                # shutil.copy(mask_image_path, dest_mask_folder)
                # if rgb_thermal == 'rgb':
                #     os.rename(os.path.join(dest_mask_folder, f'{i}_label.png'),
                #               os.path.join(dest_mask_folder, f'{boats_dir}_{scene_dir}_{i}.png'))
                # elif rgb_thermal == 'thermal':
                #     os.rename(os.path.join(dest_mask_folder, f'{i}_label.png'),
                #               os.path.join(dest_mask_folder, f'{boats_dir}_{scene_dir}_{i}_thermal.png'))
                # else:
                #     raise ValueError("Invalid value for rgb_thermal")

def parse_args():
    parser = argparse.ArgumentParser(description="Flatten boat dataset images into destination folders with renamed files.")
    parser.add_argument("--rgb_thermal", type=str, choices=["rgb", "thermal"], default="rgb",
                        help="Affects mask renaming if you later enable mask copy logic.")
    parser.add_argument("--dest_rgb_folder", type=str, default="Boats1-22/rgb_images", help="Destination folder for RGB images")
    parser.add_argument("--dest_thermal_folder", type=str, default="Boats1-22/thermal_images", help="Destination folder for thermal images")
    parser.add_argument("--dest_mask_folder", type=str, default="Boats1-22/masked_images", help="Destination folder for mask images")
    parser.add_argument("--boats_root_path", type=str, default="Images", help="Root dataset folder (supports ~)")
    parser.add_argument("--boat_count_per_scene", type=int, default=2002, help="Images per scene (upper bound index)")
    # optional: classes path (not used by core logic; kept for compatibility)
    parser.add_argument("--classes_path", type=str,
                        default="~/boats_dataset_processing/Boat_dataset_unity/Boats1-22/classes.txt",
                        help="Path to classes.txt (optional; not required by this script)")
    return parser.parse_args()

def main():
    args = parse_args()

    move_images_to_folders(
        rgb_thermal=args.rgb_thermal,
        dest_rgb_folder=args.dest_rgb_folder,
        dest_thermal_folder=args.dest_thermal_folder,
        dest_mask_folder=args.dest_mask_folder,
        boats_root_path=args.boats_root_path,
        boat_count_per_scene=args.boat_count_per_scene
    )
    print("✅ Done.")

if __name__ == "__main__":
    main()

# python3 copy_images.py --boats_root_path Images --dest_rgb_folder Boats1-22/rgb_images --boat_count_per_scene 2002