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


from utils.generic import load_json, dataset_normalization_values
from datafactory.dirichilet_dist import distribute_data_dirichilet
from datafactory.iid_dist import distribute_data_iid

from models.resnet18 import Resnet18_model
from models.cnn import CNN
from clients.fedavg import FedAvgClient
from servers.fedavg import FedAvgServer

#paths
cwd = os.getcwd()
data_dir = f'{cwd}/data'
main_cfg_path = f'{cwd}/configs/experiment_config.json'

#made a cfg to be given to the server for better usage....as complexity grows, keep adding cfg paths here
all_cfgs = {
    'main' : main_cfg_path
}



main_cfg = load_json(main_cfg_path)
# print(main_cfg)

dataset_name = main_cfg.get("dataset", "cifar10")
seed = main_cfg.get("seed", 69)
distribution = main_cfg.get("dist", "iid")
shuffle = main_cfg.get("shuffle", True)
model_name = main_cfg.get("model", "resnet")
device = main_cfg.get("device", "cuda")
save = main_cfg.get("save", False)
ratio = main_cfg.get("ratio", 0.9)
training_config_file_path = main_cfg.get("training", "")
num_clients = main_cfg.get("clients", 5)

all_cfgs['train'] = training_config_file_path

# exp_name = '-'.join([f'{key}-{value}' for key, value in vars(args).items()])

torch.manual_seed(seed)
np.random.seed(seed)


# torch.manual_seed(seed)
# np.random.seed(seed)

client_id_name_map = {f'C{i}' : f'Client-{i}' for i in range(num_clients)} # this is redundant for now...will use it for better purposes later

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
    
    alphas = [0.1 for _ in range(num_clients)]  # change this for testing with various alphas
    
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
    i : CNN(in_channels=dataset.data.shape[-1], 
                       num_classes=num_classes
                       ) for i in range(num_clients)
}

print('Models Initialized')


clients = {client_id : FedAvgClient(name=client_id_name_map[client_id],
                                    id=client_id,
                                    dataset=datasets[ind],
                                    device=device,
                                    model=models[ind],
                                    ratio=ratio,
                                    shuffle=shuffle,
                                    train_config=training_config_file_path
                                    ) for ind, client_id in enumerate(client_id_name_map.keys())
           }


print('Clients have been initalized')


server = FedAvgServer(clients=clients,
                      configs=all_cfgs,
                      model_sample=list(models.values())[-1],
                      data=None, 
                      device=device
                      )
train_config = load_json(training_config_file_path)

server.aggregate(
    num_global_epochs=train_config.get('num_global_epochs', 10),
    num_local_epochs=train_config.get('num_local_epochs', 20),
    save=save,
    save_step=1
)


