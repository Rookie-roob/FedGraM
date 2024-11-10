import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from User.User import Userbase
from Server.sgd import SGD
import copy


# Implementation for FedAvg clients


class UserAVG(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs,localstep):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs,localstep)

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.local_server_model = copy.deepcopy(list(self.model.parameters()))
        self.dataiter=iter(self.trainloader)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train_root(self, epochs):
        LOSS = 0

        self.model.train()
        self.model.to(self.device)
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batchidx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs,embedding = self.model(inputs)
                loss = self.loss(outputs, targets)
                loss.backward()

                # LOSS+= loss.item()
                self.optimizer.step()

    def train(self, epochs):
        LOSS = 0

        self.model.train()
        self.model.to(self.device)
        i=0
        while True:
            for inputs,targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs,embedding = self.model(inputs)
                loss = self.loss(outputs, targets)
                loss.backward()
                # LOSS+= loss.item()
                self.optimizer.step()
                i+=1
                if i==self.localstep:
                    break
            if i==self.localstep:
                break