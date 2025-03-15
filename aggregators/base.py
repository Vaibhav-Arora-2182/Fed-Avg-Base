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

from abc import ABC


class BaseAggregator(ABC):
    def __init__(self,
                 clients : Iterable[FedAvgClient],
                 server : BaseServer
                 ):
        super().__init__()

        self.clients = clients
        self.server = server

    def aggregate(self):
        print('Write Aggregation logic here ')
        return