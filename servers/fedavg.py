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
from typing import Dict
from clients.fedavg import BaseClient, FedAvgClient
from servers.base import BaseServer


class FedAvgServer(BaseServer):
    def __init__(self, aggregation_startegy, clients, configs, model):
        super().__init__(aggregation_startegy, clients, configs, model)