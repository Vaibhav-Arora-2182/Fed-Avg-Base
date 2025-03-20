import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy
from typing import Dict, List, Tuple, Iterable
from servers.fedavg import BaseServer, FedAvgServer
from clients.fedavg import BaseClient, FedAvgClient
from aggregators.base import BaseAggregator
from tqdm import tqdm
import os, time

from abc import ABC
from typing import Union

class FedAvgAggregator(BaseAggregator):
    def __init__(self,
                 clients : Iterable[FedAvgClient],
                 server : BaseServer,
                 device : Union[str, torch.device]
                 
                 ):
        super().__init__(clients, server, device)

    def aggregate(self, 
                  global_rounds : int, 
                  local_rounds : int,
                  save_global_model : bool = False,
                  save_step : int = 1
                  ):
        
        progress_bar = tqdm(range(global_rounds+1), desc=f'Number of Global rounds compeleted : 0/{global_rounds}') 

        metrics = {
            'Loss' : [],
            'Accuracy' : []
        }

        for i in progress_bar:
            torch.cuda.empty_cache()
            
            #local train step 
            for client in self.clients: client.local_train(local_rounds)
            
            client_models = [client.model for client in self.clients]
            
            # creating model with zero weights
            global_model_state_dict = {key: torch.zeros_like(val) for key, val in client_models[-1].state_dict().items()}

            for client in self.clients :
                for key in global_model_state_dict.keys(): 
                    global_model_state_dict[key] = global_model_state_dict[key].to(self.device)
                    global_model_state_dict[key] += (client.model.state_dict()[key]/len(self.clients)).to(global_model_state_dict[key].dtype)

            #updating client models 
            for client in self.clients : client.update_model(global_model_state_dict)

            if save_global_model and ((i%save_step) == 1) : 
                os.makedirs(f'saved/models{self.server.exp_name}_time_{self.start_time}', exist_ok=True)
                torch.save(global_model_state_dict, f'saved/models/{self.server.exp_name}/global_model_{i}.pth')


                
            epoch_metrics = self._gloabl_eval()
            
            metrics["Loss"] += [epoch_metrics["Loss"]]
            metrics["Accuracy"] += [epoch_metrics["Accuracy"]]
            progress_bar.set_description(f'Number of Global rounds compeleted : {i}/{global_rounds}')

        return metrics      