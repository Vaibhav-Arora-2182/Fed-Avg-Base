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

from abc import ABC

class FedAvgAggregator(BaseAggregator):
    def __init__(self, 
                 clients : Iterable[FedAvgClient],
                 server : FedAvgServer
                ):
        super().__init__(clients, server)

    def aggregate(self, 
                  global_rounds : int, 
                  local_rounds : int,
                  save_global_model : bool = False,
                  save_step : int = 1
                  ):
        
        progress_bar = tqdm(range(global_rounds+1), desc=f'Number of Global rounds compeleted : 0/{global_rounds}') 

        for i in progress_bar:
            torch.cuda.empty_cache()
            
            client_models = [client.model for client in self.clients]
            

        

        return 