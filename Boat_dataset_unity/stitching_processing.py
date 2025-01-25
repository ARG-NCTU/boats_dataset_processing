import os
from tqdm import tqdm

# h1 h2
# os.makedirs("stitched_results/h1_h2", exist_ok=True)
# os.system("python stiching.py --input_dir Images/boats14-22/Scene1 --output_dir stiched_results/h1_h2")

input_root = "Images"
output_root = "stiched_results"
suffixes = ["", "_depth", "_seg", "_thermal"]
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
            os.system(f"python stiching.py --input_dir {scene_folder_path} --suffix {suffix} --output_dir {output_path} --h1_path {h1_path} --h2_path {h2_path}")
        
        