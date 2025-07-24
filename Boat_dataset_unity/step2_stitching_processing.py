import os
from tqdm import tqdm


input_root = "Images"
output_root = "stitched_results"
suffixes = [None, "_depth", "_thermal", "_seg"]
h1_path = "stitched_results/h1_h2/homography/H1_1.npy"
h2_path = "stitched_results/h1_h2/homography/H2_1.npy"

for boat_folder in tqdm(sorted(os.listdir(input_root)), desc="Boat Folder", leave=False):
    boat_folder_path = os.path.join(input_root, boat_folder)
    os.makedirs(os.path.join(output_root, boat_folder), exist_ok=True)
    for scene_folder in tqdm(sorted(os.listdir(boat_folder_path)), desc="Scene Folder", leave=False):
        scene_folder_path = os.path.join(boat_folder_path, scene_folder)
        os.makedirs(os.path.join(output_root, boat_folder, scene_folder), exist_ok=True)
        output_path = os.path.join(output_root, boat_folder, scene_folder)
        for suffix in tqdm(suffixes, desc="Suffix", leave=False):
            if suffix is None:
                os.system(f"python3 Stitcher.py --input_dir {scene_folder_path} --output_dir {output_path} --h1_path {h1_path} --h2_path {h2_path}")
            else:
                os.system(f"python3 Stitcher.py --input_dir {scene_folder_path} --suffix {suffix} --output_dir {output_path} --h1_path {h1_path} --h2_path {h2_path}")

        
# python3 step2_stitching_processing.py