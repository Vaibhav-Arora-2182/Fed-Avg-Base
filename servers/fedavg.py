import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm

import random
from copy import deepcopy
from abc import ABC
from typing import Dict, Union, List
import os

from utils.generic import load_json, parse_json_recursively
from clients.fedavg import BaseClient, FedAvgClient
from servers.base import BaseServer
from aggregators.fedavg import FedAvgAggregator
from logger.logger import Logger
        
class FedAvgServer(BaseServer):
    def __init__(self,
                 clients : Dict[str, FedAvgClient],
                 configs : Dict[str, str],
                 model : torch.nn.Module,
                 data : Union[torch.utils.data.Dataset, None],
                 device : Union[str, torch.device]
                 ):
        super().__init__(clients, configs, model, data, device)

        self.aggregator = FedAvgAggregator(clients=self.clients.values(),
                                           server=self,
                                           device=self.device)
        self.logger()

        

    def set_exp_name(self):
        main_cfg = self.configs['main']
        main_cfg_data = load_json(main_cfg)
        exp_name = parse_json_recursively(main_cfg_data)

        self.exp_name = exp_name

    def train_clients(self, local_epochs) : 
        for client in self.clients.values() : client.local_train(local_epochs)
    
    def distribute_models(self, model : Union[torch.nn.Module, dict]) -> None:
        for client in self.clients.values() : client.update_model(model=model)
        return 
    
    def logger(self):
        self.logger = Logger(exp_name=self.exp_name)
        self.aggregator.logger = self.logger
        for client in self.clients.values() : client.logger =  self.logger 
        # Add metrics here

    def aggregate(self,
                  num_global_epochs : int,
                  num_local_epochs : int,
                  save : bool = False,
                  save_step : int = 1
                  ):
        progress_bar = tqdm(range(num_global_epochs+1), desc=f'Number of Global rounds compeleted : 0/{num_global_epochs}')
        for i in progress_bar :
            torch.cuda.empty_cache()
            self.train_clients(local_epochs=num_local_epochs)
            metrics, model = self.aggregator.aggregate()
        if save and ((i%save_step) == 1) : 
            os.makedirs(f'saved/models{self.exp_name}_time_{self.aggregator.start_time}', exist_ok=True)
            torch.save(model, f'saved/models/{self.exp_name}/global_model_{i}.pth')
        progress_bar.set_description(f'Number of Global rounds compeleted : {i}/{num_global_epochs}')

        





        
        

    