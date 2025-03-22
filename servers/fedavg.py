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
                 model_sample : torch.nn.Module,
                 data : Union[torch.utils.data.Dataset, None],
                 device : Union[str, torch.device]
                 ):
        super().__init__(clients, configs, model_sample, data, device)

        self.aggregator = FedAvgAggregator(clients=self.clients.values(),
                                           device=self.device)
        self.init_logger()

        

    
    def set_exp_name(self):

        main_cfg = self.configs['main']
        main_cfg_data = load_json(main_cfg)
        exp_name = parse_json_recursively(main_cfg_data)

        print(exp_name)

        dataset = exp_name.get('dataset', '')
        seed = exp_name.get('seed', '')
        dist = exp_name.get('dist', '')
        clients = exp_name.get('clients', '')
        model = exp_name.get('model', '')
        
        lr = exp_name.get('training-learning_rate', '')
        optimizer = exp_name.get('training-optimizer', '')
        batch_size = exp_name.get('training-batch_size', '')

        self.exp_name = f'{dataset}_{model}_seed{seed}_clients{clients}_dist{dist}_lr{lr}_opt{optimizer}_bs{batch_size}'
        

    def train_clients(self, local_epochs) : 
        for client in self.clients.values() : client.local_train(local_epochs)
    
    def distribute_models(self, model : Union[torch.nn.Module, dict]) -> None:
        for client in self.clients.values() : client.update_model(model=model)
        return 
    
    def init_logger(self):
        self.logger = Logger(exp_name=self.exp_name)
        self.aggregator.logger = self.logger
        for client in self.clients.values() : client.logger =  self.logger 
        self.logger.add_metrics(["global_avg_accuracy", "global_avg_loss"])
        self.logger.add_metrics([f'{client.name}_loss' for client in self.clients.values()] + [f'{client.name}_accuracy' for client in self.clients.values()])

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


            client_train_accuracies = {client.name: client.metrics['train_accuracies'][-num_local_epochs-1:] for client in self.clients.values()}
            client_test_accuracies = {client.name: client.metrics['test_accuracies'][-num_local_epochs-1:] for client in self.clients.values()}
            client_train_loss = {client.name: client.metrics['train_losses'][-num_local_epochs-1:] for client in self.clients.values()}
            client_test_loss = {client.name: client.metrics['test_losses'][-num_local_epochs-1:] for client in self.clients.values()}
            
            self.logger.update_scaler('global_avg_accuracy', [metrics['Accuracy'][-1]])
            self.logger.update_scaler('global_avg_loss', [metrics['Loss'][-1]])
            self.logger.update_scalers("Client_train_Accuracies", client_train_accuracies)
            self.logger.update_scalers("Client_test_Accuracies", client_test_accuracies)
            self.logger.update_scalers("Client_train_losses", client_train_loss)
            self.logger.update_scalers("Client_test_losses", client_test_loss)
            torch.cuda.empty_cache()

        





        
        

    