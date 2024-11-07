import os
import json
import csv
import argparse
from tqdm import tqdm

def sort_files(file):
    return file.lower()

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

            # Update class statistic
            if class_name not in class_statistic:
                class_statistic[class_name] = 1
            else:
                class_statistic[class_name] += 1

    return class_statistic

def total_images(label_path):
    total_images = 0

    with open(label_path, 'r') as f:
        data = json.load(f)
        total_images = len(data['images'])

    return total_images
    


def main(args):

    # Load classes from files
    class_list = load_classes(args.classes)
    # print('Classes:', class_list)

    # Generate class statistics
    virtual_train_rgb_statistic = statistic_classes(args.virtual_train_rgb_json, class_list)
    virtual_val_rgb_statistic = statistic_classes(args.virtual_val_rgb_json, class_list)
    virtual_train_thermal_statistic = statistic_classes(args.virtual_train_thermal_json, class_list)
    virtual_val_thermal_statistic = statistic_classes(args.virtual_val_thermal_json, class_list)
    real_train_rgb_statistic = statistic_classes(args.real_train_json, class_list)
    real_val_rgb_statistic = statistic_classes(args.real_val_json, class_list)

    # Calculate class statistic percent
    print('-' * 50)
    print('Totol number of classes:', len(class_list))

    # Calculate class statistic percent
    sum_train_images = total_images(args.virtual_train_rgb_json) + total_images(args.virtual_train_thermal_json) + total_images(args.real_train_json)
    sum_val_images = total_images(args.virtual_val_rgb_json) + total_images(args.virtual_val_thermal_json) + total_images(args.real_val_json)
    sum_images = sum_train_images + sum_val_images
    sum_train_percent = (sum_train_images / sum_images) * 100
    sum_val_percent = (sum_val_images / sum_images) * 100
    sum_train_anno = sum(virtual_train_rgb_statistic.values()) + sum(virtual_train_thermal_statistic.values()) + sum(real_train_rgb_statistic.values())
    sum_val_anno = sum(virtual_val_rgb_statistic.values()) + sum(virtual_val_thermal_statistic.values()) + sum(real_val_rgb_statistic.values())
    sum_anno = sum_train_anno + sum_val_anno
    sum_train_anno_percent = (sum_train_anno / sum_anno) * 100
    sum_val_anno_percent = (sum_val_anno / sum_anno) * 100
    print('-' * 50)
    print('Total number of images:', sum_images)
    print('Total number / percentage of images in training set:', sum_train_images, f' / ({int(sum_train_percent)}%)')
    print('Total number / percentage of images in validation set:', sum_val_images, f' / ({int(sum_val_percent)}%)')
    
    print('-' * 50)
    print('Total number of annotations:', sum_anno)
    print('Total number / percentage of annotations in training set:', sum_train_anno, f' / ({int(sum_train_anno_percent)}%)')
    print('Total number / percentage of annotations in validation set:', sum_val_anno, f' / ({int(sum_val_anno_percent)}%)')

    print('-' * 50)
    print('Average number of annotations per image in training set:', sum_train_anno / sum_train_images)

    # Calculate row and column sums
    row_sums = {class_name: virtual_train_rgb_statistic.get(class_name, 0) + virtual_val_rgb_statistic.get(class_name, 0) + virtual_train_thermal_statistic.get(class_name, 0) + virtual_val_thermal_statistic.get(class_name, 0) + real_train_rgb_statistic.get(class_name, 0) + real_val_rgb_statistic.get(class_name, 0) for class_name in class_list}
    col_sums = {
        'Virtual Train RGB': sum(virtual_train_rgb_statistic.values()),
        'Virtual Val RGB': sum(virtual_val_rgb_statistic.values()),
        'Virtual Train Thermal': sum(virtual_train_thermal_statistic.values()),
        'Virtual Val Thermal': sum(virtual_val_thermal_statistic.values()),
        'Real Train': sum(real_train_rgb_statistic.values()),
        'Real Val': sum(real_val_rgb_statistic.values())
    }

    print('-' * 50)
    print('Dataset class & type statistics:')
    header = ['Class Name', 'Virtual Train RGB', 'Virtual Val RGB', 'Virtual Train Thermal', 'Virtual Val Thermal', 'Real Train RGB', 'Real Val RGB', '(100% / Avg Row%)']
    print(f'{header[0]:<20} {header[1]:<20} {header[2]:<20} {header[3]:<20} {header[4]:<20} {header[5]:<20} {header[6]:<20} {header[7]:<20}')
    print('-' * 160)
    for class_name in class_list:
        combined_statistic = {
            'Virtual Train RGB': f'{virtual_train_rgb_statistic.get(class_name, 0)} ({int((virtual_train_rgb_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((virtual_train_rgb_statistic.get(class_name, 0) / col_sums["Virtual Train RGB"]) * 100) if col_sums["Virtual Train RGB"] > 0 else 0}%)',
            'Virtual Val RGB': f'{virtual_val_rgb_statistic.get(class_name, 0)} ({int((virtual_val_rgb_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((virtual_val_rgb_statistic.get(class_name, 0) / col_sums["Virtual Val RGB"]) * 100) if col_sums["Virtual Val RGB"] > 0 else 0}%)',
            'Virtual Train Thermal': f'{virtual_train_thermal_statistic.get(class_name, 0)} ({int((virtual_train_thermal_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((virtual_train_thermal_statistic.get(class_name, 0) / col_sums["Virtual Train Thermal"]) * 100) if col_sums["Virtual Train Thermal"] > 0 else 0}%)',
            'Virtual Val Thermal': f'{virtual_val_thermal_statistic.get(class_name, 0)} ({int((virtual_val_thermal_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((virtual_val_thermal_statistic.get(class_name, 0) / col_sums["Virtual Val Thermal"]) * 100) if col_sums["Virtual Val Thermal"] > 0 else 0}%)',
            'Real Train': f'{real_train_rgb_statistic.get(class_name, 0)} ({int((real_train_rgb_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((real_train_rgb_statistic.get(class_name, 0) / col_sums["Real Train"]) * 100) if col_sums["Real Train"] > 0 else 0}%)',
            'Real Val': f'{real_val_rgb_statistic.get(class_name, 0)} ({int((real_val_rgb_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((real_val_rgb_statistic.get(class_name, 0) / col_sums["Real Val"]) * 100) if col_sums["Real Val"] > 0 else 0}%)'
        }
        avg_row_percent = sum([int(combined_statistic[key].split(' ')[1].split('%')[0].replace('(', '')) for key in combined_statistic]) / len(combined_statistic)
        print(f'{class_name:<20} '
              f'{combined_statistic["Virtual Train RGB"]:<20} '
              f'{combined_statistic["Virtual Val RGB"]:<20} '
              f'{combined_statistic["Virtual Train Thermal"]:<20} '
              f'{combined_statistic["Virtual Val Thermal"]:<20} '
              f'{combined_statistic["Real Train"]:<20} '
              f'{combined_statistic["Real Val"]:<20} '
              f'(100% / {int(avg_row_percent)}%)')

    avg_col_percent = {key: sum([int(combined_statistic[key].split(' ')[1].split('%')[0].replace('(', '')) for class_name in class_list]) / len(class_list) for key in col_sums.keys()}
    print(f'{"(Avg Col% / 100%)":<20} '
          f'({int(avg_col_percent["Virtual Train RGB"]):<2}% / 100%){"":<8} '
          f'({int(avg_col_percent["Virtual Val RGB"]):<2}% / 100%){"":<8} '
          f'({int(avg_col_percent["Virtual Train Thermal"]):<2}% / 100%){"":<8} '
          f'({int(avg_col_percent["Virtual Val Thermal"]):<2}% / 100%){"":<8} '
          f'({int(avg_col_percent["Real Train"]):<2}% / 100%){"":<8} '
          f'({int(avg_col_percent["Real Val"]):<2}% / 100%){"":<8}')
    print('-' * 160)
    print('meaning of the items in the table:')
    print('1. The first number is the number of annotations in the class.')
    print('2. The second number in the bracket is the percentage of the class in the row.')
    print('3. The third number in the bracket is the percentage of the class in the column.')
    print('-' * 160)

    # Save class statistics to CSV
    with open('class_statistics.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Class Name', 'Virtual Train RGB', 'Virtual Val RGB', 'Virtual Train Thermal', 'Virtual Val Thermal', 'Real Train', 'Real Val'])
        # Number and percentage of annotations in each class
        for class_name in class_list:
            combined_statistic = {
                'Virtual Train RGB': f'{virtual_train_rgb_statistic.get(class_name, 0)} ({int((virtual_train_rgb_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((virtual_train_rgb_statistic.get(class_name, 0) / col_sums["Virtual Train RGB"]) * 100) if col_sums["Virtual Train RGB"] > 0 else 0}%)',
                'Virtual Val RGB': f'{virtual_val_rgb_statistic.get(class_name, 0)} ({int((virtual_val_rgb_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((virtual_val_rgb_statistic.get(class_name, 0) / col_sums["Virtual Val RGB"]) * 100) if col_sums["Virtual Val RGB"] > 0 else 0}%)',
                'Virtual Train Thermal': f'{virtual_train_thermal_statistic.get(class_name, 0)} ({int((virtual_train_thermal_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((virtual_train_thermal_statistic.get(class_name, 0) / col_sums["Virtual Train Thermal"]) * 100) if col_sums["Virtual Train Thermal"] > 0 else 0}%)',
                'Virtual Val Thermal': f'{virtual_val_thermal_statistic.get(class_name, 0)} ({int((virtual_val_thermal_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((virtual_val_thermal_statistic.get(class_name, 0) / col_sums["Virtual Val Thermal"]) * 100) if col_sums["Virtual Val Thermal"] > 0 else 0}%)',
                'Real Train': f'{real_train_rgb_statistic.get(class_name, 0)} ({int((real_train_rgb_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((real_train_rgb_statistic.get(class_name, 0) / col_sums["Real Train"]) * 100) if col_sums["Real Train"] > 0 else 0}%)',
                'Real Val': f'{real_val_rgb_statistic.get(class_name, 0)} ({int((real_val_rgb_statistic.get(class_name, 0) / row_sums[class_name]) * 100) if row_sums[class_name] > 0 else 0}% / {int((real_val_rgb_statistic.get(class_name, 0) / col_sums["Real Val"]) * 100) if col_sums["Real Val"] > 0 else 0}%)',
            }
            writer.writerow([class_name, combined_statistic['Virtual Train RGB'], combined_statistic['Virtual Val RGB'], combined_statistic['Virtual Train Thermal'], combined_statistic['Virtual Val Thermal'], combined_statistic['Real Train'], combined_statistic['Real Val']])

    print('Class statistics saved to class_statistics.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset paths and generate class statistics.")
    parser.add_argument('--classes', type=str, default="Boat_dataset_unity/Boats/classes.txt", help="Path to the classes file.")
    parser.add_argument('--virtual_train_rgb_json', type=str, default="Boat_dataset/annotations/instances_train2024_rv.json", help="Path to the virtual rgb training JSON annotation file.")
    parser.add_argument('--virtual_val_rgb_json', type=str, default="Boat_dataset/annotations/instances_val2024_rv.json", help="Path to the virtual rgb validation JSON annotation file.")
    parser.add_argument('--virtual_train_thermal_json', type=str, default="Boat_dataset/annotations/instances_train2024_tv.json", help="Path to the virtual thermal training JSON annotation file.")
    parser.add_argument('--virtual_val_thermal_json', type=str, default="Boat_dataset/annotations/instances_val2024_tv.json", help="Path to the virtual thermal validation JSON annotation file.")
    parser.add_argument('--real_train_json', type=str, default="Boat_dataset/annotations/instances_train2024_rr.json", help="Path to the real rgb training JSON annotation file.")
    parser.add_argument('--real_val_json', type=str, default="Boat_dataset/annotations/instances_val2024_rr.json", help="Path to the real rgb validation JSON annotation file.")
    
    args = parser.parse_args()
    main(args)