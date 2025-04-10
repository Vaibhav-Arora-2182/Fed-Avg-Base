import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
import json 
from abc import ABC
from typing import Tuple, Union

from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.optim import Adam, SGD
from tqdm import tqdm

from utils.generic import load_json
from clients.base import BaseClient


class FedAvgClient(BaseClient):

    def __init__(self, name, id, dataset, device, model, ratio, shuffle, train_config):
        super().__init__(name, id, dataset, device, model, ratio, shuffle, train_config)
        # print(self.train_loader, self.test_loader, self.name)
        # print(f"🔍 {self.name}: Train loader batches: {len(self.train_loader)}, Test loader batches: {len(self.test_loader)}")

        # for batch in self.train_loader:
        #     print(f"📝 {self.name}: Train batch size: {batch[0].shape}, Labels: {batch[1].shape}")
        #     break  # Just print the first batch
        self.progress_bar = tqdm(range(self.gepochs*self.lepochs), leave=False)


    
    def _split_data(self):
        train_size = int(self.ratio* len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_data, self.test_data = random_split(dataset=self.dataset, lengths=[train_size, test_size])
        
        while self.batch_size > 1 and train_size // self.batch_size == 0: self.batch_size //= 2 

        train_loader = DataLoader(dataset=self.train_data,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle,
                                  drop_last=True 
                                )
        test_loader = DataLoader(
                                dataset=self.test_data,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle
        )

        return train_loader, test_loader
    
    def _parse_train_config(self):
        train_config = load_json(self.train_config)


        self.learning_rate = train_config.get('learning_rate', 0.01) 
        optimizer_name = train_config.get('optimizer', 'Adam')  
        self.batch_size = train_config.get('batch_size', 64)  
        self.optimizer_params = train_config.get('optimizer_params', {})
        self.gepochs = train_config.get('num_global_epochs', 10)
        self.lepochs = train_config.get('num_local_epochs', 20)
        loss_name = train_config.get('loss', 'CE')
        self.local_epochs_completed = 0
        self.global_epochs_completed = 0

        
        if optimizer_name.lower() == "adam" :
            self.optimizer = Adam(self.model.parameters(),
                                lr=self.learning_rate,
                                betas=(self.optimizer_params.get('b1', 0.9),
                                        self.optimizer_params.get('b2', 0.999)),
                                eps=self.optimizer_params.get('eps', 1e-8),
                                weight_decay=self.optimizer_params.get('weight_decay', 0),
                                amsgrad=self.optimizer_params.get('amsgrad', False)
                                )

            
        elif optimizer_name.lower() == "sgd" :
            self.optimizer = SGD(self.model.parameters(),
                            lr=self.learning_rate,
                            momentum=self.optimizer_params.get('momentum', 0),
                            dampening=self.optimizer_params.get('dampening', 0),
                            weight_decay=self.optimizer_params.get('weight_decay', 0),
                            nesterov=self.optimizer_params.get('nesterov', False)
                            )
            
        else :
            raise ValueError('Optimizer not implemented')
        
        if loss_name == 'CE' :
            self.loss = nn.CrossEntropyLoss()
        else :
            raise ValueError('Loss not passed correctly')
        


        return 
    
    def local_train(self, num_epochs, logger) -> None:
        # progress_bar = tqdm(range(num_epochs), desc=f'{self.name} in its 0/{num_epochs} epoch in Global Epoch {self.global_epochs_completed}/{self.gepochs}')
        torch.cuda.empty_cache()

        for i in range(num_epochs):
            self.progress_bar.set_description(f'{self.name} in its {self.local_epochs_completed + i+1}/{self.gepochs*self.lepochs} epoch in Global Epoch {self.global_epochs_completed}/{self.gepochs}')
            self.model = self.model.to(self.device)

            self.model.train()
            train_loss, train_correct, train_total =  0, 0, 0


            # training 
            for inputs, labels in self.train_loader : 
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                # print(inputs.device, next(self.model.parameters()).device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.shape[0]
                train_correct += predicted.eq(labels).sum().item()
                # print(train_correct, train_total, labels.shape)

            train_accuracy = (train_correct/ train_total)*100
            logger.logger.log({f'{self.name}_train_accuracy': train_accuracy})
            logger.logger.log({f'{self.name}_train_loss': train_loss})
            
            #testing
            test_loss, test_correct, test_total = 0, 0, 0
            self.model.eval()
            with torch.no_grad(): 
                for inputs, labels in self.test_loader : 
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.shape[0]
                    test_correct += predicted.eq(labels).sum().item()
                    torch.cuda.empty_cache()

            train_accuracy = (train_correct/ train_total)*100
            self.metrics['train_accuracies'] += [train_accuracy]
            self.metrics['train_losses'] += [train_loss]

            test_accuracy = (test_correct/test_total)*100
            
            logger.logger.log({f'{self.name}_test_accuracy': test_accuracy})
            logger.logger.log({f'{self.name}_test_loss': test_loss})
            self.metrics['test_accuracies'] += [test_accuracy]
            self.metrics['test_losses'] += [test_loss]
            self.local_epochs_completed += 1
        
        self.global_epochs_completed += 1

        torch.cuda.empty_cache()
        return 
