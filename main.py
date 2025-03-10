import numpy as np
import pandas as pd
import random
import os
import argparse

import torch
import torchvision

from torchvision.datasets import CIFAR10, MNIST, CIFAR100
from torchvision import transforms 
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from utils import load_json, dataset_normalization_values
from data_dist import distribute_data_iid, distribute_data_dirichilet
from models import Resnet18_model

#paths
cwd = os.getcwd()
data_dir = f'{cwd}/data'
main_cfg_path = f'{cwd}/configs/experiment_config.json'



main_cfg = load_json(main_cfg_path)
print(main_cfg)

# loading exp variables
dataset_name = main_cfg['dataset']
seed = main_cfg['seed']
distribution  = main_cfg['dist']
shuffle = main_cfg['shuffle']
model_name = main_cfg['model']
device=main_cfg['device']
save = main_cfg['save']
ratio = main_cfg['ratio']
training_config_file_path = main_cfg['training']
num_clients = main_cfg['clients']


# exp_name = '-'.join([f'{key}-{value}' for key, value in vars(args).items()])

torch.manual_seed(seed)
np.random.seed(seed)


# torch.manual_seed(seed)
# np.random.seed(seed)

if dataset_name == 'cifar10' : 
    dataset = torchvision.datasets.CIFAR10(
        root=f'{data_dir}/CIFAR10',
        train=True,
        download=True,
        transform=transforms.ToTensor())
    
elif dataset_name == 'mnist' :
    dataset = torchvision.datasets.MNIST(
        root=f'{data_dir}/MNIST',
        train=True,
        download=True,
        transform=transforms.ToTensor())
    
elif dataset_name == 'cifar100' :
    dataset = torchvision.datasets.CIFAR100(
        root=f'{data_dir}/CIFAR100',
        train=True,
        download=True,
        transform=transforms.ToTensor())    


mean, std = dataset_normalization_values(dataset=dataset)


dataset.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])



num_classes = len(dataset.classes)

if distribution == 'dirichilet' :
    
    alphas = [0.5 for _ in range(num_clients)]  # change this for testing with various alphas
    
    datasets = distribute_data_dirichilet(
        dataset=dataset, 
        num_clients=num_clients, 
        alphas=alphas, 
        seed=seed,
        shuffle=shuffle
        )

elif distribution == 'iid' : 
    datasets = distribute_data_iid(
        dataset=dataset,
        num_clients=num_clients,
        seed=seed,
        shuffle=shuffle
    )
print('Data Distribution is done')



models = {
    i : Resnet18_model(in_channels=dataset.data.shape[-1], 
                       num_classes=num_classes
                       ) for i in range(num_clients)
}
print('Models Initialized')


