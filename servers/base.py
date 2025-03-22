import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import random
from copy import deepcopy

from abc import ABC
from tqdm import tqdm
from utils.generic import load_json
from typing import Dict, Union
from clients.fedavg import BaseClient, FedAvgClient


class BaseServer(ABC):
    def __init__(self,
                 clients : Dict[str, BaseClient],
                 configs : Dict[str, str],
                 model_sample : torch.nn.Module,
                 data : Union[torch.utils.data.Dataset, None],
                 device : Union[str, torch.device]
                ):
        super().__init__()
        
        self.clients = clients
        self.configs = configs
        self.model = deepcopy(model_sample)
        self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        self.exp_name = ''
        self.data = data
        self.device = device

        self.set_exp_name()

    def train_clients(self):
        print("Write the process for training clients here using client functions")
        return

    def aggregate(self):
        print("Write the code for calling the aggregator and updating the aggregated weights. \nThis will be complicated for clustered FL but keep it simple otherwise.")
        return

    def distribute_models(self):
        print('Write the model distributing logic')
        return
    
    def init_logger(self):
        print('Use the logger modules to obtain the desired results from clients')
        return
    
    def set_exp_name(self) -> str:
        print("Write a small function to create create identifier for individual experiments")
        return ''
    