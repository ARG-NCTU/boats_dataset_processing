import os
import json
import shutil
import random
from copy import deepcopy
from tqdm import tqdm
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def check_exist(data, image_dir):
    # Filter out entries for non-existing images
    existing_images = []
    existing_annotations = []
    not_existing_images = []
    for image in tqdm(data['images'], desc="Filtering images", leave=False):
        if os.path.exists(os.path.join(image_dir, image['file_name'])):
            existing_images.append(image)
            # Find and append corresponding annotations
            image_annotations = [anno for anno in data['annotations'] if anno['image_id'] == image['id']]
            existing_annotations.extend(image_annotations)
        else:
            print(f"Image {image['file_name']} not found. Skipping...")
            not_existing_images.append(image)

    print(f"Found {len(existing_images)} existing images and {len(not_existing_images)} non-existing images.")

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data['images'])} images to {file_path}")


def load_classes():
    class_list = []
    with open("Boat_dataset_unity/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

def split_dataset(dataset, val_ratio=0.2, seed=42):
    random.seed(seed)

    # Create a dictionary for fast image_id lookup
    image_id_to_index = {image['id']: idx for idx, image in enumerate(dataset['images'])}

    # Get all image indices and shuffle them
    indices = np.arange(len(dataset['images']))
    np.random.shuffle(indices)

    # Determine the number of validation images
    num_val = int(len(indices) * val_ratio)

    # Split the indices into training and validation
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    # Using tqdm to show progress during image and annotation filtering
    train_images = [dataset['images'][i] for i in tqdm(train_indices, desc="Processing training images", leave=False)]
    val_images = [dataset['images'][i] for i in tqdm(val_indices, desc="Processing validation images", leave=False)]

    # Filter annotations using tqdm progress bar
    train_annotations = [
        anno for anno in tqdm(dataset['annotations'], desc="Processing training annotations", leave=False)
        if anno['image_id'] in image_id_to_index and image_id_to_index[anno['image_id']] in train_indices
    ]
    val_annotations = [
        anno for anno in tqdm(dataset['annotations'], desc="Processing validation annotations", leave=False)
        if anno['image_id'] in image_id_to_index and image_id_to_index[anno['image_id']] in val_indices
    ]

    # Build the datasets
    train_dataset = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': dataset['categories']
    }

    val_dataset = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': dataset['categories']
    }

    return train_dataset, val_dataset

def merge_datasets(virtual_dataset, real_dataset):
    merged_dataset = {
        'images': virtual_dataset['images'] + real_dataset['images'],
        'annotations': virtual_dataset['annotations'] + real_dataset['annotations'],
        'categories': virtual_dataset['categories']  # Assumes categories are the same
    }
    return merged_dataset

def copy_images(image_list, source_dirs, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for i, img in tqdm(enumerate(image_list), desc="Copying images", leave=False):
        file_name = img['file_name']
        for source_dir in source_dirs:
            source_path = os.path.join(source_dir, file_name)
            if os.path.exists(source_path):
                dest_path = os.path.join(dest_dir, file_name)
                shutil.copy2(source_path, dest_path)
                break
    print(f"Copied {i+1} images.")


# Load datasets
virtual_dataset = load_json('Boat_dataset_unity/coco_formatted_unity_rgb_data.json')
real_train_dataset = load_json('real_dataset/annotations/instances_train2023r.json')
real_val_dataset = load_json('real_dataset/annotations/instances_val2023r.json')


# Split datasets for virtual dataset
virtual_train_dataset, virtual_val_dataset = split_dataset(virtual_dataset, val_ratio=0.2, seed=42)

print('splitted train num: ', len(virtual_train_dataset['images']))
print('splitted val num: ', len(virtual_val_dataset['images']))

# Merge datasets
merged_train_dataset = merge_datasets(virtual_train_dataset, real_train_dataset)
merged_val_dataset = merge_datasets(virtual_val_dataset, real_val_dataset)

# Copy images to the common directory (assuming paths are valid and images exist)
copy_images(merged_train_dataset['images'], ['Boat_dataset_unity/rgb_images/', 'real_dataset/images/'], 'Boat_dataset/images/')
copy_images(merged_train_dataset['images'], ['Boat_dataset_unity/rgb_images/', 'real_dataset/images/'], 'Boat_dataset/train2024/')
copy_images(merged_val_dataset['images'], ['Boat_dataset_unity/rgb_images/', 'real_dataset/images/'], 'Boat_dataset/images/')
copy_images(merged_val_dataset['images'], ['Boat_dataset_unity/rgb_images/', 'real_dataset/images/'], 'Boat_dataset/val2024/')

print('merged train num: ', len(merged_train_dataset['images']))
print('merged val num: ', len(merged_val_dataset['images']))

os.makedirs('Boat_dataset/annotations', exist_ok=True)
# Save merged datasets
save_json(merged_train_dataset, 'Boat_dataset/annotations/instances_train2024.json')
save_json(merged_val_dataset, 'Boat_dataset/annotations/instances_val2024.json')

save_json(real_train_dataset, 'Boat_dataset/annotations/instances_train2023r.json')
save_json(real_val_dataset, 'Boat_dataset/annotations/instances_val2023r.json')

save_json(virtual_train_dataset, 'Boat_dataset/annotations/instances_train2024v.json')
save_json(virtual_val_dataset, 'Boat_dataset/annotations/instances_val2024v.json')

# Check if all images exist
check = False
if check:
    check_exist(virtual_train_dataset, 'Boat_dataset_unity/rgb_images/')
    check_exist(virtual_val_dataset, 'Boat_dataset_unity/rgb_images/')
    check_exist(merged_train_dataset, 'Boat_dataset/train2024/')
    check_exist(merged_val_dataset, 'Boat_dataset/val2024/')

