import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
import json 
from abc import ABC
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter



class BaseClient(ABC):
    def __init__(self,
                 name : str,
                 id : str,
                 dataset : Dataset, 
                 model : nn.Module,
                 ratio : float,
                 train_config : dict
                 ) -> None:
        super().__init__()


        self.name = name
        self.id = id
        self.dataset = dataset
        self.ratio = ratio
        self.model = model
        self.train_config = train_config


    def train(self):
        pass

    def update(self):
        pass

    def eval(self):
        pass