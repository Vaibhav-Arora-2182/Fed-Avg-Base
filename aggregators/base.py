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
from tqdm import tqdm
from time import time

from abc import ABC
from typing import Union


class BaseAggregator(ABC):
    def __init__(self,
                 clients : Iterable[FedAvgClient],
                 server : BaseServer,
                 device : Union[str, torch.device]
                 ):
        super().__init__()

        self.clients = clients
        self.server = server
        self.device = device
        self.start_time = time()

    def aggregate(self):
        print('Write Aggregation logic here ')
        return
    
    def gloabl_eval(self):
        metrics = {
            'Accuracy' : 0,
            'Loss' : 0
        }
        
        for client in self.clients :
            Loss, Accuracy = client.eval()
            metrics['Accuracy'] += Accuracy/len(self.clients)
            metrics['Loss'] += Loss/len(self.clients)

        return metrics