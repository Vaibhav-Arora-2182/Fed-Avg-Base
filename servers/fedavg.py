import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import random
from copy import deepcopy

from abc import ABC
from tqdm import tqdm
from typing import Dict, Union

from utils.generic import load_json, parse_json_recursively
from clients.fedavg import BaseClient, FedAvgClient
from servers.base import BaseServer
from aggregators.fedavg import FedAvgAggregator

class FedAvgServer(BaseServer):
    def __init__(self,
                 clients : Dict[str, BaseClient],
                 configs : Dict[str, str],
                 model : torch.nn.Module,
                 data : Union[torch.utils.data.Dataset, None],
                 device : Union[str, torch.device]
                 ):
        super().__init__(clients, configs, model, data, device)

        self.aggregator = FedAvgAggregator(clients=self.clients.values(),
                                           server=self,
                                           device=self.device,)

        

    def set_exp_name(self):
        main_cfg = self.configs['main']
        main_cfg_data = load_json(main_cfg)
        exp_name = parse_json_recursively(main_cfg_data)

        self.exp_name = exp_name

        
        

    