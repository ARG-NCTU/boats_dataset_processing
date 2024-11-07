from datasets import load_dataset
from datasets import Features, Value, Sequence

# Specify the correct schema for your dataset
features = Features({
    'image_id': Value('int32'),
    'image_path': Value('string'),
    'width': Value('int32'),
    'height': Value('int32'),
    'objects': {
        'id': Sequence(Value('int32')),
        'area': Sequence(Value('float32')),  # Ensure area is treated as float
        'bbox': Sequence(Sequence(Value('float32'), length=4)),  # Ensure bbox is a list of 4 floats
        'category': Sequence(Value('int32'))
    }
})

# Load the dataset with the correct features
dataset_rtvrr = load_dataset(
    'json', 
    data_files={'train': 'annotations/instances_train2024_rtvrr.jsonl', 
                'validation': 'annotations/instances_val2024_rtvrr.jsonl'},
    features=features  # Explicitly specify the schema
)

dataset_rvrr = load_dataset(
    'json', 
    data_files={'train': 'annotations/instances_train2024_rvrr.jsonl', 
                'validation': 'annotations/instances_val2024_rvrr.jsonl'},
    features=features  # Apply the same schema
)

dataset_rtv = load_dataset(
    'json', 
    data_files={'train': 'annotations/instances_train2024_rtv.jsonl', 
                'validation': 'annotations/instances_val2024_rtv.jsonl'},
    features=features  # Apply the same schema
)

dataset_rv = load_dataset(
    'json', 
    data_files={'train': 'annotations/instances_train2024_rv.jsonl', 
                'validation': 'annotations/instances_val2024_rv.jsonl'},
    features=features  # Apply the same schema
)

dataset_tv = load_dataset(
    'json', 
    data_files={'train': 'annotations/instances_train2024_tv.jsonl', 
                'validation': 'annotations/instances_val2024_tv.jsonl'},
    features=features  # Apply the same schema
)

dataset_rr = load_dataset(
    'json', 
    data_files={'train': 'annotations/instances_train2024_rr.jsonl', 
                'validation': 'annotations/instances_val2024_rr.jsonl'},
    features=features  # Apply the same schema
)

# Display the dataset info
print(dataset_rtvrr)
print(dataset_rvrr)
print(dataset_rtv)
print(dataset_rv)
print(dataset_tv)
print(dataset_rr)

# print the first example in the training set with pretty print
print(dataset_rtvrr['train'][0])