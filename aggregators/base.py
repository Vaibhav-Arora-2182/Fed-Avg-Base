import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy
from typing import Dict, List, Tuple, Iterable

from clients.fedavg import BaseClient
from tqdm import tqdm
from time import time

from abc import ABC
from typing import Union


class BaseAggregator(ABC):
    def __init__(self,
                 clients : Iterable[BaseClient],
                 device : Union[str, torch.device]
                 ):
        super().__init__()

        self.clients = clients
        self.device = device
        self.start_time = time()
        self.logger = None

    def aggregate(self):
        print('Write Aggregation logic here ')
        return
    
    def _gloabl_eval(self):
        metrics = {
            'Accuracy' : 0,
            'Loss' : 0
        }
        
        for client in self.clients :
            Loss, Accuracy = client.eval()
            metrics['Accuracy'] += Accuracy/len(self.clients)
            metrics['Loss'] += Loss/len(self.clients)

            train_loss, train_accuracy = client.eval(client.train_data)
            test_loss, test_accuracy = client.eval()

            
            client.metrics['train_accuracies'] += [train_accuracy]
            client.metrics['train_losses']  += [train_loss]
            client.metrics['test_accuracies'] += [test_accuracy]
            client.metrics['test_losses']  += [test_loss]



        return metrics