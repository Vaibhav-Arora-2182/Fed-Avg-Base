import torch 
import torchvision
import json
import os


def dataset_normalization_values(dataset): 
    data = torch.stack([item[0] for item in dataset], dim=0)
    mean = data.mean(dim=(0, 2, 3))
    std = data.std(dim=(0, 2, 3))
    return [mean, std]


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def is_file_path(value):
        return isinstance(value, str) and os.path.exists(value)


def parse_json_recursively(json_data: dict, prefix=""):
    """Recursively extract JSON key-value pairs, loading nested JSON files when encountered."""
    result = {}

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if is_file_path(value):
                nested_data = load_json(value)
                result.update(parse_json_recursively(nested_data, prefix + key + "-"))
            else:
                result.update(parse_json_recursively(value, prefix + key + "-"))
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            result.update(parse_json_recursively(item, prefix + str(i) + "-"))
    else:
        result[prefix.rstrip("-")] = json_data 
    return result
