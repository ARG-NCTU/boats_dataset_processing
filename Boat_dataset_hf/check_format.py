import os
import json

# Define the directory containing the JSONL annotation files
annotations_dir = './annotations'

def validate_annotations(annotations_file):
    """
    Check if any annotations contain invalid types for bbox, area, and category fields.
    """
    errors = 0
    with open(annotations_file, 'r') as file:
        for line_num, line in enumerate(file, start=1):
            try:
                annotation = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON at line {line_num} in {annotations_file}: {e}")
                continue

            # Check objects for invalid types
            objects = annotation.get("objects", {})
            if isinstance(objects, dict):
                bbox = objects.get("bbox", [])
                area = objects.get("area", [])
                category = objects.get("category", [])

                # Ensure bbox is a list of floats or ints
                if not all(isinstance(b, (float, int)) for bb in bbox for b in bb):
                    print(f"Invalid bbox at line {line_num} in {annotations_file}: {bbox}")
                    errors += 1

                # Ensure area is a float or int
                if not all(isinstance(a, (float, int)) for a in area):
                    print(f"Invalid area at line {line_num} in {annotations_file}: {area}")
                    errors += 1

                # Ensure category is an int
                if not all(isinstance(c, int) for c in category):
                    print(f"Invalid category at line {line_num} in {annotations_file}: {category}")
                    errors += 1

    if errors == 0:
        print(f"All annotations in {annotations_file} are valid.")
    else:
        print(f"Found {errors} invalid annotations in {annotations_file}.")

# Iterate over all JSONL files in the annotations directory
for filename in os.listdir(annotations_dir):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(annotations_dir, filename)
        print(f"Checking file: {file_path}")
        validate_annotations(file_path)
