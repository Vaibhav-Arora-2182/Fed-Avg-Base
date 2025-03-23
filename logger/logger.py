import threading
import time
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable, Dict, Any
import numpy as np
import torch

class Logger:
    def __init__(self, exp_name):
        self.writer = SummaryWriter(log_dir=f'tensorboard/{exp_name}')
        self.monitored_metrics_ctr = {}

    def update_scaler(self,
                      key : str,
                      values : Iterable
                      ):
        
        for ind, value in enumerate(values): self.writer.add_scalar(key, value, self.monitored_metrics_ctr[key] + ind)
        self.monitored_metrics_ctr[key] += len(values)


    def update_scalers(self, key, values : Dict[str, Iterable]):
        inside = True
        same_length = True
        for v in values.keys() : 
            if v not in self.monitored_metrics_ctr.keys() : inside = False
            if len(values[v]) != len(list(values.values())[0])  : same_length = False
            if not (isinstance(values[v], np.ndarray) or isinstance(values[v]), torch.Tensor) : values[v] = np.array(values[v])

        assert(inside and same_length, "Value not in monitored metrics or length issue")

        for ind in range(1, len(values.values())+1) : 
            transform_dict = {}
            for key in values.keys():
                transform_dict[key] = values[key][ind - 1]
            self.writer.add_scalars(key, transform_dict, global_step=self.monitored_metrics_ctr[list(values.keys())[0]] + ind)

        for key in self.monitored_metrics_ctr.keys(): self.monitored_metrics_ctr[key] += len(values[v])

    def add_metric(self, key : str)  -> None:
        if key not in self.monitored_metrics_ctr.keys() : self.monitored_metrics_ctr[key] = 0
        return
    
    def add_metrics(self, keys : Iterable[str]):
        for key in keys : 
            if key not in self.monitored_metrics_ctr.keys() : self.monitored_metrics_ctr[key] = 0
        return


    