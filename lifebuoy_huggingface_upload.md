```bash
cd Lifebuoy_dataset_hf/
huggingface-cli login

huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 annotations/instances_train2024.jsonl data/instances_train2024.jsonl --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_dataset_2024 annotations/instances_val2024.jsonl data/instances_val2024.jsonl --repo-type=dataset --commit-message="Upload val labels to hub"


huggingface-cli download ARG-NCTU/Lifebuoy_dataset_2024 --include "*.jsonl" --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
huggingface-cli download ARG-NCTU/Lifebuoy_dataset_2024 data/classes.txt --repo-type dataset --local-dir ~/huggingface-notebooks/transformers_doc/en/pytorch
```


```bash
huggingface-cli login

huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 Lifebuoy_underwater_dataset_2024.py --repo-type=dataset --commit-message="Update script to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 README.md --repo-type=dataset --commit-message="Update README to hub"

huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 annotations/instances_train2024.jsonl data/instances_train2024.jsonl --repo-type=dataset --commit-message="Upload training labels to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 annotations/instances_val2024.jsonl data/instances_val2024.jsonl --repo-type=dataset --commit-message="Upload val labels to hub"
huggingface-cli upload ARG-NCTU/Lifebuoy_underwater_dataset_2024 classes.txt data/classes.txt --repo-type=dataset --commit-message="Upload classes list to hub"
```