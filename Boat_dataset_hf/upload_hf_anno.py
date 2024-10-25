import os

os.system('huggingface-cli upload ARG-NCTU/Boat_dataset_2024 classes.txt data/classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"')

for anno_files in os.listdir("annotations"):
    if anno_files.endswith(".jsonl"):
        os.system(f'huggingface-cli upload ARG-NCTU/Boat_dataset_2024 annotations/{anno_files} data/{anno_files} --repo-type=dataset --commit-message="Upload labels to hub"')
