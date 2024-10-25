import os
import json
from tqdm import tqdm
import csv

def sort_files(file):
    return file.lower()

def load_classes():
    class_list = []
    with open("~/boats_dataset_processing/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()
class_list.append('real_WAM_V')

def is_virtual_image(filepath, class_list):
    for class_name in class_list:
        if filepath.lower().find(class_name.lower()) != -1:
            return True
    return False

def statistic_classes(label_path):
    class_statistic = {}
    
    with open(label_path, 'r') as f:
        data = json.load(f)
        for image, annotation in zip(data['images'], data['annotations']):
            filename = image['file_name']
            if is_virtual_image(filename, class_list):
                class_id = annotation['category_id']
            else:
                class_id = class_list.index('real_WAM_V')

            class_name = class_list[class_id]
            if class_name not in class_statistic:
                class_statistic[class_name] = 1
            else:
                class_statistic[class_name] += 1

    return class_statistic

train_label_path = 'Boat_dataset/annotations/instances_train2024.json'
val_label_path = 'Boat_dataset/annotations/instances_val2024.json'
train_class_statistic = statistic_classes(train_label_path)
val_class_statistic = statistic_classes(val_label_path)

# Train & val class percents
# train + val = 100%
class_statistic = {}
for class_name in class_list:
    class_statistic[class_name] = train_class_statistic.get(class_name, 0) + val_class_statistic.get(class_name, 0)

class_statistic = dict(sorted(class_statistic.items(), key=lambda item: class_list.index(item[0])))

train_class_statistic_percent = {}
val_class_statistic_percent = {}
for class_name in class_list:
    total_count = class_statistic[class_name]
    train_count = train_class_statistic.get(class_name, 0)
    val_count = val_class_statistic.get(class_name, 0)
    train_class_statistic_percent[class_name] = f"{int(train_count / total_count * 100)}%"
    val_class_statistic_percent[class_name] = f"{int(val_count / total_count * 100)}%"

print('Train / (Train + Val) class statistic:', train_class_statistic)
print()
print('Train class statistic percent:', train_class_statistic_percent)

print('-' * 50)
print('Val / (Train + Val) class statistic:', val_class_statistic)
print()
print('Val class statistic percent:', val_class_statistic_percent)

# Make a CSV file to save the class statistic
with open('class_statistic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Train', 'Train percent', 'Val', 'Val percent'])
    for class_name in class_list:
        writer.writerow([class_name, train_class_statistic.get(class_name, 0), train_class_statistic_percent[class_name], val_class_statistic.get(class_name, 0), val_class_statistic_percent[class_name]])
