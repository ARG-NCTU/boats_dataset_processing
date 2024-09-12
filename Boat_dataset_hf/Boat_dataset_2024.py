# Source: https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py

import csv
import json
import os

import datasets

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {Boat dataset},
author={Tzu-Chi Chen, Inc.},
year={2024}
}
"""

_DESCRIPTION = """\
This dataset is designed to solve an object detection task with images of boats.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ARG-NCTU/Boat_dataset_2024/resolve/main"

_LICENSE = ""

_URLS = {
    "classes": f"{_HOMEPAGE}/data/classes.txt",
    "train": f"{_HOMEPAGE}/data/instances_train2024.jsonl",
    "val": f"{_HOMEPAGE}/data/instances_val2024.jsonl",
    # "test": f"{_HOMEPAGE}/data/instances_val2024r.jsonl"
}

class BoatDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.1.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="Boat_dataset_2024", version=VERSION, description="Dataset for detecting boats in aerial images."),
    ]

    DEFAULT_CONFIG_NAME = "Boat_dataset_2024"  # Provide a default configuration

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'image_id': datasets.Value('int32'),
                'image_path': datasets.Value('string'),
                'width': datasets.Value('int32'),
                'height': datasets.Value('int32'),
                'objects': datasets.Features({
                    'id': datasets.Sequence(datasets.Value('int32')),
                    'area': datasets.Sequence(datasets.Value('float32')),
                    'bbox': datasets.Sequence(datasets.Sequence(datasets.Value('float32'), length=4)),  # [x, y, width, height]
                    'category': datasets.Sequence(datasets.Value('int32'))
                }),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Download all files and extract them
        downloaded_files = dl_manager.download_and_extract(_URLS)

        # Load class labels from the classes file
        with open('classes.txt', 'r') as file:
            classes = [line.strip() for line in file.readlines()]
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotations_file": downloaded_files["train"],
                    "classes": classes,
                    "split": "train",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotations_file": downloaded_files["val"],
                    "classes": classes,
                    "split": "val",
                }
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "annotations_file": downloaded_files["test"],
            #         "classes": classes,
            #         "split": "val_real",
            #     }
            # ),
        ]

    def _generate_examples(self, annotations_file, classes, split):
        # Process annotations
        with open(annotations_file, encoding="utf-8") as f:
            for key, row in enumerate(f):
                try:
                    data = json.loads(row.strip())
                    yield key, {
                        "image_id": data["image_id"],
                        "image_path": data["image_path"],
                        "width": data["width"],
                        "height": data["height"],
                        "objects": data["objects"],
                    }
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at line {key + 1}: {row}")
                    continue
