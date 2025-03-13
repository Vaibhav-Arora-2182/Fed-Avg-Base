import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import random
from copy import deepcopy

from abc import ABC
from tqdm import tqdm
from utils import load_json
from typing import Dict
from client import BaseClient, FedAvgClient


class BaseServer(ABC):
    def __init__(self,
                 aggregation_startegy : str,
                 clients : Dict[str, BaseClient],
                 configs : Dict[str, str],
                 model : torch.nn.Module
                ):
        
        self.aggregation_strategy = aggregation_startegy
        self.clients = clients
        self.configs = configs
        self.model = deepcopy(model)
        self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    def train_clients(self):
        print("Write the process for training clients here using client functions")
        return

    def aggregate(self):
        print("Write the code for calling the aggregator and updating the aggregated weights. \nThis will be complicated for clustered FL but keep it simple otherwise.")
        return

    def distribute_models(self):
        print('Write the model distributing logic')
        return
    
    def logger(self):
        print('Use the logger modules to obtain the desired results from clients')
        return
    




class FedAvgServer(BaseServer):
    def __init__(self, aggregation_startegy, clients, configs, model):
        super().__init__(aggregation_startegy, clients, configs, model)