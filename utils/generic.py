import torch 
import torchvision
import json



def dataset_normalization_values(dataset):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    data = torch.stack([item[0] for item in dataset], dim=0)
    mean = data.mean(dim=(0, 2, 3))
    std = data.std(dim=(0, 2, 3))
    return [mean, std]

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data