import os
import cv2
import json
from tqdm import tqdm
import numpy as np

def sort_files(file):
    return file.lower()  

def load_classes():
    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

def merge_json_files(rgb_thermal='rgb', bbox_seg='bbox', boats_root_path='~/dataset_boat12/Images', boat_count_per_scene=3002, vis=False, vis_dir='Visualization'):
    """
    Merges the JSON files and the images to create a COCO formatted JSON file
    Args:
        rgb_thermal: rgb or thermal
        bbox_seg: bbox or both
        boats_root_path: Path to the root directory of the dataset
        boat_count_per_scene: Number of boats per scene
    Returns:
        A dictionary containing the COCO formatted data
    """
    
    images = []
    annotations = []
    image_id = 1  # Starting image ID
    
    image_counter = 0  # Counter for the number of images processed
    annotation_id = 1  # Starting annotation ID

    # Data Structure
    # boats1-6
    #   - Scene1
    #       - *.png
    #       - *.json
    #   - Scene1_blur1
    #       - *.png
    #       - *.json
    #   - Scene1_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene1_blurN
    #       - *.png
    #       - *.json
    #   - Scene2
    #       - *.png
    #       - *.json
    #   - Scene2_blur1
    #       - *.png
    #       - *.json
    #   - Scene2_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene2_blurN
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene5
    #       - *.png
    #       - *.json
    #   - Scene5_blur1
    #       - *.png
    #       - *.json
    #   - Scene5_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene5_blurN
    #       - *.png
    #       - *.json
    # boats7-12
    #   - Scene1
    #       - *.png
    #       - *.json
    #   - Scene1_blur1
    #       - *.png
    #       - *.json
    #   - Scene1_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene1_blurN
    #       - *.png
    #       - *.json
    #   - Scene2
    #       - *.png
    #       - *.json
    #   - Scene2_blur1
    #       - *.png
    #       - *.json
    #   - Scene2_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene2_blurN
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene5
    #       - *.png
    #       - *.json
    #   - Scene5_blur1
    #       - *.png
    #       - *.json
    #   - Scene5_blur2
    #       - *.png
    #       - *.json
    #   ...
    #   - Scene5_blurN
    #       - *.png
    #       - *.json
    # N >= 1, default N = 1

    # Load the mask ids json file to dictionary
    with open("mask_ids.json", "r") as f:
        mask_ids = json.load(f)
    # print(mask_ids.keys())
    obscured_num = {cname: 0 for cname in class_list}

    for boats_dir in tqdm(os.listdir(boats_root_path), desc="Boats", leave=False):
        # print(f"Processing {boats_dir}")
        boats_dir_path = os.path.join(boats_root_path, boats_dir)
        for i, scene_dir in tqdm(enumerate(os.listdir(boats_dir_path)), desc="Scenes", leave=False):
            # print(f"Processing {scene_dir}")
            # Only Scene1*
            if rgb_thermal == 'thermal' and 'Scene1' not in scene_dir:
                continue
            scene_dir_path = os.path.join(boats_dir_path, scene_dir)
            for i in tqdm(range(2, boat_count_per_scene - 1), desc="Images", leave=False):
                # RGB images: 1.png, 2.png, 3.png, ...
                # Thermal images: 1_thermal.png, 2_thermal.png, 3_thermal.png, ...
                # Mask images: 1_seg.png, 2_seg.png, 3_seg.png, ...
                # JSON files: 1.main.json, 2.main.json, 3.main.json, ...

                if rgb_thermal == 'rgb':
                    image_path = os.path.join(scene_dir_path, f'{i}.png')
                elif rgb_thermal == 'thermal':
                    image_path = os.path.join(scene_dir_path, f'{i}_thermal.png')
                else:
                    raise ValueError("Invalid value for rgb_thermal")
                
                # print(f"Processing {image_path}")
                image = cv2.imread(image_path)
                image_width = image.shape[1]
                image_height = image.shape[0]

                file_name = f'{boats_dir}_{scene_dir}_{i}.png' if rgb_thermal == 'rgb' else f'{boats_dir}_{scene_dir}_{i}_thermal.png'

                images.append({
                    "id": image_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": file_name,
                })

                label_path = os.path.join(scene_dir_path, f'{i}.main.json')

                with open(label_path, 'r') as f:
                    data = json.load(f)
                    
                    ##############################
                    
                    mask_path = os.path.join(scene_dir_path, f'{i}_seg.png')
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = np.array(mask_img)
                    source_img = image.copy()
                    for obj in data['objects']:
                        # if float(obj['visibility']) < 0.5:
                        #     continue
                        ##### code here #####
                        current_visible_area = float('inf')
                        if boats_dir in mask_ids.keys():
                            mask_id = int(mask_ids[boats_dir][obj['class']])
                            current_obj_mask = np.where(mask == mask_id, 255, 0).astype(np.uint8)
                            # Chooose the largest connected component
                            contours, _ = cv2.findContours(current_obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                current_obj_mask = np.zeros_like(current_obj_mask)
                                cv2.drawContours(current_obj_mask, [largest_contour], -1, 255, cv2.FILLED)
                                if bbox_seg == 'both':
                                    polygon = largest_contour.flatten().tolist()
                                    # Ensure that the polygon has even number of elements (x, y pairs)
                                    if len(polygon) % 2 == 1:
                                        polygon = polygon[:-1]
                                else:
                                    polygon = []
                            # cv2.imwrite('mask.png', current_obj_mask)
                            # mask to bbox area: (xmax-xmin) * (ymax-ymin) 
                            x, y, w, h = cv2.boundingRect(current_obj_mask)
                            x, y, w, h = int(x), int(y), int(w), int(h)
                            current_visible_bbox = [x, y, w, h]
                            current_visible_area = w * h

                        ##############################

                        class_name = obj['class']
                        if class_name in class_list:
                            class_id = class_list.index(class_name)
                            
                            if bbox_seg in ['bbox', 'both']:
                                point = obj['bounding_box']
                                x1, y1 = point["top_left"]
                                x2, y2 = point["bottom_right"]
                                w, h = x2 - x1, y2 - y1
                                bbox = [x1, y1, w, h]
                                bbox = [round(x, 3) for x in bbox]
                                bbox[0], bbox[1], bbox[2], bbox[3] = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                                
                            
                            if boat_count_per_scene > 100:
                                count_on =  i % 100 == 0
                            else:
                                count_on = True
                            
                            if vis and count_on:
                                # draw current_visible_bbox and bbox
                                if bbox_seg in ['bbox', 'both']:
                                    cv2.rectangle(source_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
                                if boats_dir in mask_ids.keys():
                                    if current_visible_area < 0.3 * w * h:
                                        # use pink color to draw the bbox
                                        cv2.rectangle(source_img, (current_visible_bbox[0], current_visible_bbox[1]), (current_visible_bbox[0]+current_visible_bbox[2], current_visible_bbox[1]+current_visible_bbox[3]), (255, 0, 255), 2)
                                    else:
                                        # use red color to draw the bbox
                                        cv2.rectangle(source_img, (current_visible_bbox[0], current_visible_bbox[1]), (current_visible_bbox[0]+current_visible_bbox[2], current_visible_bbox[1]+current_visible_bbox[3]), (0, 0, 255), 2)
                                
                                dir = os.path.join(vis_dir, boats_dir, scene_dir)
                                os.makedirs(dir, exist_ok=True)
                                vis_img_path = os.path.join(dir, f'{i}.png') if rgb_thermal == 'rgb' else os.path.join(dir, f'{i}_thermal.png')
                                cv2.imwrite(vis_img_path, source_img)

                            if current_visible_area < 0.3 * w * h:
                                obscured_num[obj['class']] += 1
                                continue
                            


                            area = w * h if boats_dir in mask_ids.keys() else current_visible_area
                            anno = {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id,
                                "area": area,
                                "iscrowd": 0
                            }

                            if bbox_seg in ['bbox', 'both']:
                                if boats_dir in mask_ids.keys():
                                    anno["bbox"] = current_visible_bbox
                                else:
                                    anno["bbox"] = bbox
                            
                            if bbox_seg == 'both':
                                if polygon:
                                    anno["segmentation"] = [polygon]
                                else:
                                    anno["segmentation"] = [[]]

                            # print(anno)
                            annotations.append(anno)

                            annotation_id += 1

                image_id += 1

    blur_level = 2
    images_num_theory = len(os.listdir(boats_root_path)) * (len(os.listdir(boats_dir_path)) // (blur_level + 1)) * (blur_level + 1) * (boat_count_per_scene - 1)
    obscured_rate = {cname: str(int(round(obscured_num[cname] / len(annotations), 2) * 100))+'%' for cname in class_list}

    info = f"""
    Data processing completed.\n
    RGB or Thermal: {rgb_thermal}\n
    Bbox or Both: {bbox_seg}\n
    Boats root path: {boats_root_path}\n
    Boats dirs * Scenes * (Source + Blur) * Images = {len(os.listdir(boats_root_path))} * {len(os.listdir(boats_dir_path)) // (blur_level + 1)} * {blur_level + 1} * {boat_count_per_scene - 1} = {images_num_theory}\n
    Actual number of images: {image_id-1}\n
    Number of annotations: {len(annotations)}\n
    Number of obscured objects: {obscured_num}\n
    Obscured rate (Number of obscured objects / Number of annotations): {obscured_rate}\n\n\n
    """
    print(info)
    # create the info.txt file if it does not exist
    if not os.path.exists('process_info.txt'):
        with open('process_info.txt', 'w') as f:
            f.write(info)
    else:
        with open('process_info.txt', 'a') as f:
            f.write(info)

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_list)]
    }     

coco_formatted_rgb_data = merge_json_files(rgb_thermal='rgb', bbox_seg='bbox', boats_root_path='Images', boat_count_per_scene=3002, vis=True, vis_dir='Visualization') # 3002
with open('coco_formatted_unity_rgb_data.json', 'w') as f:
    json.dump(coco_formatted_rgb_data, f, indent=4)

coco_formatted_thermal_data = merge_json_files(rgb_thermal='thermal', bbox_seg='bbox', boats_root_path='Images', boat_count_per_scene=3002, vis=True, vis_dir='Visualization') # 3002
with open('coco_formatted_unity_thermal_data.json', 'w') as f:
    json.dump(coco_formatted_thermal_data, f, indent=4)