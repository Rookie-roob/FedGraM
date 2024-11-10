import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from User.User import Userbase


# Implementation for FedAvg clients


class UserFakeNIID1(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, localstep, n_attackers,fakeNIID_decrease_ratio):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs, localstep)
        self.n_attackers = n_attackers
        self.loss = nn.CrossEntropyLoss()
        self.fakeNIID_decrease_ratio=fakeNIID_decrease_ratio
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self, epochs):
        LOSS = 0

        self.model.train()
        self.model.to(self.device)
        i = 0
        while True:
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                print(inputs.length)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                loss.backward()
                # LOSS+= loss.item()
                self.optimizer.step()
                i += 1
                if i == self.localstep:
                    break
            if i == self.localstep:
                break
