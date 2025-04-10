import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy
from typing import Dict, List, Tuple, Iterable
from clients.rfl_self import RFL_SelfClient
from aggregators.base import BaseAggregator
from tqdm import tqdm
import os, time

from abc import ABC
from typing import Union

class RFL_SelfAggregator(BaseAggregator):
    def __init__(self,
                 clients : Iterable[RFL_SelfClient],
                 device : Union[str, torch.device]
                  
                 ):
        super().__init__(clients, device)

    def aggregate(self):

        metrics = {
            'Loss' : [],
            'Accuracy' : []
        }

        torch.cuda.empty_cache()
        client_models = [client.model for client in self.clients]
        
        # creating model with zero weights
        global_model_state_dict = {key: torch.zeros_like(val) for key, val in client_models[-1].state_dict().items()}

        for client in self.clients :
            for key in global_model_state_dict.keys(): 
                global_model_state_dict[key] = global_model_state_dict[key].to(self.device)
                global_model_state_dict[key] += (client.model.state_dict()[key]/len(self.clients)).to(global_model_state_dict[key].dtype)

        #updating client models 
        for client in self.clients : client.update_model(global_model_state_dict)   
        epoch_metrics = self._gloabl_eval()

        
        metrics["Loss"] += [epoch_metrics["Loss"]]
        metrics["Accuracy"] += [epoch_metrics["Accuracy"]]
        

        return metrics, global_model_state_dict