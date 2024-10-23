from datasets import load_dataset

dataset_rr = load_dataset("ARG-NCTU/Boat_dataset_2024", name="rr", use_auth_token=True, trust_remote_code=True)

dataset_rtvrr = load_dataset("ARG-NCTU/Boat_dataset_2024", name="rtvrr", use_auth_token=True, trust_remote_code=True)

dataset_rvrr = load_dataset("ARG-NCTU/Boat_dataset_2024", name="rvrr", use_auth_token=True, trust_remote_code=True)

dataset_rtv = load_dataset("ARG-NCTU/Boat_dataset_2024", name="rtv", use_auth_token=True, trust_remote_code=True)

dataset_rv = load_dataset("ARG-NCTU/Boat_dataset_2024", name="rv", use_auth_token=True, trust_remote_code=True)

dataset_tv = load_dataset("ARG-NCTU/Boat_dataset_2024", name="tv", use_auth_token=True, trust_remote_code=True)

# from datasets import load_dataset

# # Load the local dataset files
# dataset_rr = load_dataset('json', data_files={'train': 'annotations/instances_train2024_rr.jsonl', 
#                                               'validation': 'annotations/instances_val2024_rr.jsonl'})

# dataset_rtvrr = load_dataset('json', data_files={'train': 'annotations/instances_train2024_rtvrr.jsonl', 
#                                                  'validation': 'annotations/instances_val2024_rtvrr.jsonl'})

# dataset_rvrr = load_dataset('json', data_files={'train': 'annotations/instances_train2024_rvrr.jsonl', 
#                                                 'validation': 'annotations/instances_val2024_rvrr.jsonl'})

# dataset_rtv = load_dataset('json', data_files={'train': 'annotations/instances_train2024_rtv.jsonl', 
#                                                'validation': 'annotations/instances_val2024_rtv.jsonl'})

# dataset_rv = load_dataset('json', data_files={'train': 'annotations/instances_train2024_rv.jsonl', 
#                                               'validation': 'annotations/instances_val2024_rv.jsonl'})

# dataset_tv = load_dataset('json', data_files={'train': 'annotations/instances_train2024_tv.jsonl', 
#                                               'validation': 'annotations/instances_val2024_tv.jsonl'})
