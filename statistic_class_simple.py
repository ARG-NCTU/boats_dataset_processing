import os
import json
import csv
import argparse

def load_classes(classes_path):
    class_list = []
    with open(classes_path, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

def statistic_classes(label_path, class_list):
    class_statistic = {}
    
    with open(label_path, 'r') as f:
        data = json.load(f)
        for annotation in data['annotations']:
            class_id = annotation['category_id']
            class_name = class_list[class_id]
            class_statistic[class_name] = class_statistic.get(class_name, 0) + 1

    return class_statistic

def total_images(label_path):
    with open(label_path, 'r') as f:
        data = json.load(f)
        return len(data['images'])

def main(args):
    # Load classes from file
    class_list = load_classes(args.classes)

    # Generate class statistics
    train_statistic = statistic_classes(args.train_json, class_list)
    val_statistic = statistic_classes(args.val_json, class_list)

    # Calculate totals and percentages
    sum_train_images = total_images(args.train_json)
    sum_val_images = total_images(args.val_json)
    sum_images = sum_train_images + sum_val_images
    train_percent = (sum_train_images / sum_images) * 100
    val_percent = (sum_val_images / sum_images) * 100

    sum_train_annotations = sum(train_statistic.values())
    sum_val_annotations = sum(val_statistic.values())
    sum_annotations = sum_train_annotations + sum_val_annotations
    train_anno_percent = (sum_train_annotations / sum_annotations) * 100
    val_anno_percent = (sum_val_annotations / sum_annotations) * 100

    # Display summary
    print('-' * 50)
    print('Total number of classes:', len(class_list))
    print('Total number of images:', sum_images)
    print(f'Training images: {sum_train_images} ({train_percent:.2f}%)')
    print(f'Validation images: {sum_val_images} ({val_percent:.2f}%)')
    print('Total number of annotations:', sum_annotations)
    print(f'Training annotations: {sum_train_annotations} ({train_anno_percent:.2f}%)')
    print(f'Validation annotations: {sum_val_annotations} ({val_anno_percent:.2f}%)')
    print('Average annotations per training image:', sum_train_annotations / sum_train_images)
    print('Average annotations per validation image:', sum_val_annotations / sum_val_images)
    
    # Display class statistics
    print('-' * 50)
    print('Class Name            Train Annotations    Validation Annotations')
    print('-' * 50)
    for class_name in class_list:
        train_count = train_statistic.get(class_name, 0)
        val_count = val_statistic.get(class_name, 0)
        print(f'{class_name:<20} {train_count:<20} {val_count}')

    # Save statistics to CSV
    with open('class_statistics.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Class Name', 'Train Annotations', 'Validation Annotations'])
        for class_name in class_list:
            train_count = train_statistic.get(class_name, 0)
            val_count = val_statistic.get(class_name, 0)
            writer.writerow([class_name, train_count, val_count])

    print('Class statistics saved to class_statistics.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset paths and generate class statistics.")
    parser.add_argument('--classes', type=str, default="Boat_dataset_unity/Boats/classes.txt", help="Path to the classes file.")
    parser.add_argument('--train_json', type=str, default="Boat_dataset/annotations/instances_train2024.json", help="Path to the training JSON annotation file.")
    parser.add_argument('--val_json', type=str, default="Boat_dataset/annotations/instances_val2024.json", help="Path to the validation JSON annotation file.")
    
    args = parser.parse_args()
    main(args)

# python3 statistic_class_simple.py --classes Kaohsiung_Port_dataset/annotations/classes.txt --train_json Kaohsiung_Port_dataset/annotations/instances_train2024.json --val_json Kaohsiung_Port_dataset/annotations/instances_val2024.json
