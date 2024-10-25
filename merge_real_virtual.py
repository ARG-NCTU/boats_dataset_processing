import argparse
import os
import json
import shutil
import random
from tqdm import tqdm
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data['images'])} images info and {len(data['annotations'])} annotations to {file_path}")

def remove_invalid_annotations(dataset, batch_size=10000):
    valid_annotations = []
    invalid_annotations = []
    
    annotations = dataset['annotations']
    
    for i in tqdm(range(0, len(annotations), batch_size), desc="Validating annotations in batches"):
        batch = annotations[i:i + batch_size]
        for annotation in batch:
            bbox = annotation['bbox']  # bbox is [x, y, width, height]
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            annotation['bbox'] = [x, y, w, h]

            # Update condition: x, y >= 0 and w, h > 0
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                valid_annotations.append(annotation)
            else:
                invalid_annotations.append(annotation)
    
    # Update dataset with only valid annotations
    dataset['annotations'] = valid_annotations
    
    print(f"Removed {len(invalid_annotations)} invalid annotations. Remaining valid annotations: {len(valid_annotations)}.")
    
    return dataset, invalid_annotations

# Function to shift the IDs of the images and the image_id in the annotations
def image_id_shift(dataset, image_shift):
    for image in dataset['images']:
        image['id'] += image_shift
    for annotation in dataset['annotations']:
        annotation['image_id'] += image_shift
    return dataset

# Function to shift the annotation IDs
def anno_id_shift(dataset, anno_shift):
    for annotation in dataset['annotations']:
        annotation['id'] += anno_shift
    return dataset

def split_dataset(dataset, val_ratio=0.2, batch_size=10000, seed=1234):
    random.seed(seed)

    # Create a dictionary for fast image_id lookup
    image_id_to_index = {image['id']: idx for idx, image in enumerate(dataset['images'])}

    # Get all image indices and shuffle them
    indices = np.arange(len(dataset['images']))
    np.random.shuffle(indices)

    # Determine the number of validation images
    num_val = int(len(indices) * val_ratio)

    # Split the indices into training and validation
    val_indices = set(indices[:num_val])
    train_indices = set(indices[num_val:])

    # Filter images in batches
    train_images = []
    val_images = []

    for i in tqdm(range(0, len(dataset['images']), batch_size), desc="Processing images in batches"):
        batch = dataset['images'][i:i + batch_size]
        for idx, image in enumerate(batch):
            if idx in train_indices:
                train_images.append(image)
            elif idx in val_indices:
                val_images.append(image)

    # Filter annotations in batches
    train_annotations = []
    val_annotations = []

    for i in tqdm(range(0, len(dataset['annotations']), batch_size), desc="Processing annotations in batches"):
        batch = dataset['annotations'][i:i + batch_size]
        for anno in batch:
            image_id = anno['image_id']
            if image_id in image_id_to_index:
                image_idx = image_id_to_index[image_id]
                if image_idx in train_indices:
                    train_annotations.append(anno)
                elif image_idx in val_indices:
                    val_annotations.append(anno)

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

def merge_datasets(dataset1, dataset2):
    merged_dataset = {
        'images': dataset1['images'] + dataset2['images'],
        'annotations': dataset1['annotations'] + dataset2['annotations'],
        'categories': dataset1['categories']
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

def process_datasets(unity_dataset_dir, real_dataset_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/annotations', exist_ok=True)
    if os.path.exists(f'{output_dir}/images'):
        shutil.rmtree(f'{output_dir}/images')
    if os.path.exists(f'{output_dir}/train2024'):
        shutil.rmtree(f'{output_dir}/train2024')
    if os.path.exists(f'{output_dir}/val2024'):
        shutil.rmtree(f'{output_dir}/val2024')
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    os.makedirs(f'{output_dir}/train2024', exist_ok=True)
    os.makedirs(f'{output_dir}/val2024', exist_ok=True)
    
    ############################## Load and clean datasets ##############################
    # Load and clean the virtual RGB dataset
    virtual_dataset_rgb = load_json(f'{unity_dataset_dir}/coco_formatted_unity_rgb_data.json')
    virtual_dataset_rgb, invalid_rgb_annotations = remove_invalid_annotations(virtual_dataset_rgb)
    virtual_dataset_rgb_anno_len = len(virtual_dataset_rgb['annotations'])
    virtual_dataset_rgb_image_len = len(virtual_dataset_rgb['images'])

    # Load and clean the virtual thermal dataset
    virtual_dataset_thermal = load_json(f'{unity_dataset_dir}/coco_formatted_unity_thermal_data.json')
    virtual_dataset_thermal, invalid_thermal_annotations = remove_invalid_annotations(virtual_dataset_thermal)
    virtual_dataset_thermal = anno_id_shift(virtual_dataset_thermal, virtual_dataset_rgb_anno_len)
    virtual_dataset_thermal = image_id_shift(virtual_dataset_thermal, virtual_dataset_rgb_image_len)
    virtual_dataset_thermal_anno_len = len(virtual_dataset_thermal['annotations'])
    virtual_dataset_thermal_image_len = len(virtual_dataset_thermal['images'])

    # Merge cleaned RGB and thermal datasets (virtual dataset)
    virtual_rgb_thermal_dataset = merge_datasets(virtual_dataset_rgb, virtual_dataset_thermal)
    virtual_dataset_anno_len = virtual_dataset_rgb_anno_len + virtual_dataset_thermal_anno_len
    virtual_dataset_image_len = virtual_dataset_rgb_image_len + virtual_dataset_thermal_image_len

    # Load and clean the real train dataset
    real_train_dataset = load_json(f'{real_dataset_dir}/annotations/instances_train2023r.json')
    real_train_dataset, invalid_train_annotations = remove_invalid_annotations(real_train_dataset)
    real_train_dataset_len = len(real_train_dataset['annotations'])
    real_train_dataset_image_len = len(real_train_dataset['images'])
    real_train_dataset = anno_id_shift(real_train_dataset, virtual_dataset_anno_len)
    real_train_dataset = image_id_shift(real_train_dataset, virtual_dataset_image_len)

    # Load and clean the real validation dataset
    real_val_dataset = load_json(f'{real_dataset_dir}/annotations/instances_val2023r.json')
    real_val_dataset, invalid_val_annotations = remove_invalid_annotations(real_val_dataset)
    real_val_dataset = anno_id_shift(real_val_dataset, virtual_dataset_anno_len + real_train_dataset_len)
    real_val_dataset = image_id_shift(real_val_dataset, virtual_dataset_image_len + real_train_dataset_image_len)

    ############################## Split datasets ##############################
    # Split virtual dataset into train and validation sets
    virtual_train_rtv, virtual_val_rtv = split_dataset(virtual_rgb_thermal_dataset, val_ratio=0.2)

    print('rtv splitted train num: ', len(virtual_train_rtv['images']))
    print('rtv splitted val num: ', len(virtual_val_rtv['images']))

    virtual_train_rv, virtual_val_rv = split_dataset(virtual_dataset_rgb, val_ratio=0.2)
    print('rv splitted train num: ', len(virtual_train_rv['images']))
    print('rv splitted val num: ', len(virtual_val_rv['images']))

    virtual_train_tv, virtual_val_tv = split_dataset(virtual_dataset_thermal, val_ratio=0.2)
    print('tv splitted train num: ', len(virtual_train_tv['images']))
    print('tv splitted val num: ', len(virtual_val_tv['images']))

    ############################## Merge and save datasets ##############################
    # Merge only virtual RGB and real datasets (suffix: 'rvrr')
    merged_train_rvrr = merge_datasets(virtual_train_rv, real_train_dataset)
    merged_val_rvrr = merge_datasets(virtual_val_rv, real_val_dataset)
    save_json(merged_train_rvrr, f'{output_dir}/annotations/instances_train2024_rvrr.json')
    save_json(merged_val_rvrr, f'{output_dir}/annotations/instances_val2024_rvrr.json')

    # Merge virtual RGB + thermal and real datasets (suffix: 'rtvrr')
    merged_train_rtvrr = merge_datasets(virtual_train_rtv, real_train_dataset)
    merged_val_rtvrr = merge_datasets(virtual_val_rtv, real_val_dataset)
    save_json(merged_train_rtvrr, f'{output_dir}/annotations/instances_train2024_rtvrr.json')
    save_json(merged_val_rtvrr, f'{output_dir}/annotations/instances_val2024_rtvrr.json')

    # Merge virtual RGB + thermal (suffix: 'rtv')
    save_json(virtual_train_rtv, f'{output_dir}/annotations/instances_train2024_rtv.json')
    save_json(virtual_val_rtv, f'{output_dir}/annotations/instances_val2024_rtv.json')

    # Save only virtual RGB dataset (suffix: 'rv')
    save_json(virtual_train_rv, f'{output_dir}/annotations/instances_train2024_rv.json')
    save_json(virtual_val_rv, f'{output_dir}/annotations/instances_val2024_rv.json')

    # Save only virtual Thermal dataset (suffix: 'tv')
    save_json(virtual_train_tv, f'{output_dir}/annotations/instances_train2024_tv.json')
    save_json(virtual_val_tv, f'{output_dir}/annotations/instances_val2024_tv.json')

    # Save only real datasets (suffix: 'rr')
    save_json(real_train_dataset, f'{output_dir}/annotations/instances_train2024_rr.json')
    save_json(real_val_dataset, f'{output_dir}/annotations/instances_val2024_rr.json')

    # Merge all datasets
    merged_all_dataset = merge_datasets(merged_train_rtvrr, merged_val_rtvrr)

    ############################## Copy images to output directories ##############################

    # Copy images for both merged datasets to appropriate directories
    copy_images(merged_all_dataset['images'], [f'{unity_dataset_dir}/rgb_images/', f'{real_dataset_dir}/images/'], f'{output_dir}/images/')
    copy_images(merged_train_rtvrr['images'], [f'{unity_dataset_dir}/rgb_images/', f'{real_dataset_dir}/images/'], f'{output_dir}/train2024/')
    copy_images(merged_val_rtvrr['images'], [f'{unity_dataset_dir}/rgb_images/', f'{real_dataset_dir}/images/'], f'{output_dir}/val2024/')

def process_unity_rgb_dataset(unity_dataset_dir, output_dir):
    ############################## Load and clean datasets ##############################
    # Load and clean the virtual RGB dataset
    virtual_dataset_rgb = load_json(f'{unity_dataset_dir}/coco_formatted_unity_rgb_data.json')
    virtual_dataset_rgb, invalid_rgb_annotations = remove_invalid_annotations(virtual_dataset_rgb)
    virtual_dataset_rgb_anno_len = len(virtual_dataset_rgb['annotations'])
    virtual_dataset_rgb_image_len = len(virtual_dataset_rgb['images'])

    ############################## Split datasets ##############################
    # Split virtual dataset into train and validation sets
    virtual_train_rv, virtual_val_rv = split_dataset(virtual_dataset_rgb, val_ratio=0.2)

    print('rv splitted train num: ', len(virtual_train_rv['images']))
    print('rv splitted val num: ', len(virtual_val_rv['images']))

    ############################## Merge and save datasets ##############################
    # Save only virtual RGB dataset
    save_json(virtual_train_rv, f'{output_dir}/annotations/instances_train2024.json')
    save_json(virtual_val_rv, f'{output_dir}/annotations/instances_val2024.json')

    ############################## Copy images to output directories ##############################
    # Copy images for both merged datasets to appropriate directories
    copy_images(virtual_train_rv['images'], [f'{unity_dataset_dir}/rgb_images/'], f'{output_dir}/images/')
    copy_images(virtual_val_rv['images'], [f'{unity_dataset_dir}/rgb_images/'], f'{output_dir}/train2024/')
    copy_images(virtual_val_rv['images'], [f'{unity_dataset_dir}/rgb_images/'], f'{output_dir}/val2024/')

# Main function using argparse for command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Process and merge virtual and real datasets.")
    
    # Define arguments
    parser.add_argument('--unity_dataset_dir', default='Boat_dataset_unity', help="Path to the Unity dataset directory.")
    parser.add_argument('--real_dataset_dir', default='real_dataset', help="Path to the real dataset directory.")
    parser.add_argument('--output_dir', default='Boat_dataset', help="Path to the output directory.")
    parser.add_argument('--unity_rgb_only', action='store_true', help="Process only the Unity RGB dataset.")
    
    # Parse arguments
    args = parser.parse_args()

    if args.unity_rgb_only:
        # Call process_unity_rgb_dataset function with parsed arguments
        process_unity_rgb_dataset(args.unity_dataset_dir, args.output_dir)
    else:
        # Call process_datasets function with parsed arguments
        process_datasets(args.unity_dataset_dir, args.real_dataset_dir, args.output_dir)

if __name__ == '__main__':
    main()

# python3 merge_real_virtual.py --unity_dataset_dir Boat_dataset_unity/Boats --real_dataset_dir real_dataset --output_dir Boat_dataset
# python3 merge_real_virtual.py --unity_rgb_only --unity_dataset_dir Boat_dataset_unity/Lifebuoy --output_dir Lifebuoy_dataset
