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
dataset = load_dataset(
    'json', 
    data_files={'train': 'annotations/instances_train2024.jsonl', 
                'validation': 'annotations/instances_val2024.jsonl'},
    features=features  # Explicitly specify the schema
)

# Display the dataset info
print(dataset)