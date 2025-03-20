import threading
import time
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable, Dict, Any

class Logger:
    def __init__(self, exp_name):
        self.writer = SummaryWriter(log_dir=exp_name)
        self.monitored_metrics_ctr = {}

    def update_scaler(self,
                      key : str,
                      values : Iterable
                      ):
        
        for ind, value in enumerate(values): self.writer.add_scalar(key, value, self.monitored_metrics_ctr[key] + ind)
        # increment self.monitored metrics counter


    def update_scalers(self, key, values : Dict[str, Any]):
        inside = True
        same_length = True
        for v in values.keys() : 
            if v not in self.monitored_metrics_ctr.keys() : inside = False
            if len(values[v]) != len(list(values.values())[0])  : same_length = False

        assert(inside and same_length, "Value not in monitored metrics or length issue")

        for ind in range(1, len(values.values())+1) : 
            self.writer.add_scalars(key, values, global_step=self.monitored_metrics_ctr[list(values.values())[0]] + ind)

        # increment self.monitoreed metrics counter

    