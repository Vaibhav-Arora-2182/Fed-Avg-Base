import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
import json 
from abc import ABC
from typing import Tuple, Union

from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.optim import Adam, SGD
from tqdm import tqdm

from utils.generic import load_json




class BaseClient(ABC):
    def __init__(self,
                 name : str,
                 id : str,
                 dataset : Dataset,
                 device : str,
                 model : nn.Module,
                 ratio : float,
                 shuffle : Union[torch.device, bool],
                 train_config : str,
                 ) -> None:
        super().__init__()


        self.name = name
        self.id = id
        self.dataset = dataset
        self.ratio = ratio
        self.model = model
        self.train_config = train_config
        self.shuffle=shuffle
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.metrics = {
            'train_accuracies' : [],
            'train_losses'  : [],
            
            'test_accuracies' : [],
            'test_losses'  : [],
            
        }

        self._parse_train_config()
        self.train_loader, self.test_loader = self._split_data()
        # self.local_train(5)
        self.logger = None



    def _split_data(self) -> Tuple[DataLoader, DataLoader] :
        print("Implement data spliting logic here")
        return

    def _parse_train_config(self) -> None:
        print('Implement this function to read from the train config and parse the appropriate values')
        return
    
    def local_train(self):
        print("Implement client training here")
        return

    def update_model(self, 
                     model : Union[nn.Module, dict] 
                     ):
        
        if  isinstance(model, nn.Module): self.model.load_state_dict(model.state_dict())
        elif isinstance(model, dict) : self.model.load_state_dict(model) 
        # most model formats should work here..modify if necessary, to take additional measures, u can check if the values are of type tensor
        else : raise TypeError(f"Model of type {type(model)}")
        
        
        return
    
    def update_with_grad(self, gradient):
        print("Implement model update from gradient here ")
        return

    def eval(self, dataset : Union[None, Dataset]=None):
        # Implement dataset eval here...if no dataset is provided, then eval on local test data

        self.model.eval()
        self.model.to(self.device)

        if dataset : dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        else : dataloader = self.test_loader

        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad(): 
            for inputs, labels in dataloader : 
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.shape[0]
                test_correct += predicted.eq(labels).sum().item()


        return test_loss, 100*(test_correct/test_total)
    
