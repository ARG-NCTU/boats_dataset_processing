import os
from tqdm import tqdm

# h1 h2
os.makedirs("stitched_results/h1_h2", exist_ok=True)
os.system("python3 Stitcher.py --input_dir Images/boats1-13/Scene1 --output_dir stitched_results/h1_h2")

# python3 step1_stitching_processing.py