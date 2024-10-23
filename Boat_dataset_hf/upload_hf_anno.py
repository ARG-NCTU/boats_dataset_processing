import os

os.system('huggingface-cli upload ARG-NCTU/Boat_dataset_2024 classes.txt data/classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"')

for anno_files in os.listdir("annotations"):
    if anno_files.endswith(".jsonl"):
        os.system(f'huggingface-cli upload ARG-NCTU/Boat_dataset_2024 annotations/{anno_files} data/{anno_files} --repo-type=dataset --commit-message="Upload labels to hub"')

# Make files with urls
URL = "https://huggingface.co/datasets/ARG-NCTU/Boat_dataset_2024/resolve/main"
with open("urls.txt", "w") as f:
    for anno_files in os.listdir("annotations"):
        if anno_files.endswith(".jsonl"):
            f.write(f"{URL}/data/{anno_files}\n")
    f.write(f"{URL}/data/classes.txt\n")
os.system('huggingface-cli upload ARG-NCTU/Boat_dataset_2024 urls.txt data/urls.txt --repo-type=dataset --commit-message="Upload urls to hub"')
