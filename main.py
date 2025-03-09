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


from utils import dataset_normalization_values
from data_dist import distribute_data_iid, distribute_data_dirichilet


cwd = os.getcwd()
parser = argparse.ArgumentParser()

data_dir = f'{cwd}/data'


parser.add_argument("--dataset", type=str , default="cifar10", help="Dataset : mnist/cirfar10/cifar100")
parser.add_argument("--seed", type=int , default=69, help="Seed for random number generators")
parser.add_argument("--lepochs", type=int, default=20, help="Number of local epochs on each client before sending models/gradients to the aggregator")
parser.add_argument("--gepochs", type=int, default=10, help="Number of global epochs")
parser.add_argument("--batch-size", type=int , default=128, help="Batch size for dataloaders")
parser.add_argument("--dist", type=str , default='dirichilet', help="Distrinution for dataset splitting : iid/dirichilet")
parser.add_argument("--lr", type=float , default=0.01, help="Learning rate")
parser.add_argument("--agg", type=str , default='fedavg', help="Type of aggregation strategy : fedavg/optics")
parser.add_argument("--clients", type=int , default=5, help="Number of clients participating")
parser.add_argument('--shuffle', type=bool, default=True,action=argparse.BooleanOptionalAction, help="Whether to shuffle the dataset or not")
parser.add_argument('--model', type=str, default='resnet',help="The model to be used on the dataset resnet18/custom")
parser.add_argument('--mom', type = float, default=0.9, help = 'Momentum for optimizer')
parser.add_argument('--wd', type = float, default = 5e-4, help = 'Weight decay for the optimizer')
parser.add_argument('--ratio', type=float, default = 0.9, help='Train:Total Data ratio (between 0 and 1)')
parser.add_argument('--optim', type=str, default='sgd', help = 'Optimizer for training sgd/adam')
parser.add_argument('--device', type=str, default='cuda', help='Device to run the experiment on gpu/cpu')
parser.add_argument('--save', default=False,action=argparse.BooleanOptionalAction, help="Whether to save the global model or not"), 


args = parser.parse_args()


dataset_name = args.dataset
seed = args.seed
gepochs = args.gepochs
lepochs = args.lepochs
batch_size = args.batch_size
learning_rate = args.lr
num_clients = args.clients
momentum = args.mom
weight_decay = args.wd
ratio = args.ratio
optimizer = args.optim
distribution  = args.dist
aggregation = args.agg
shuffle = args.shuffle
model_name = args.model
device=args.device
save = args.save

exp_name = '-'.join([f'{key}-{value}' for key, value in vars(args).items()])

torch.manual_seed(seed)
np.random.seed(seed)


torch.manual_seed(seed)
np.random.seed(seed)

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