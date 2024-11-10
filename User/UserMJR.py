import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from User.User import Userbase
from Server.sgd import SGD
import copy


# Implementation for FedAvg clients


class UserMJR(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs,localstep,lambda_JR):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs,localstep)

        self.loss = nn.CrossEntropyLoss()
        self.lambda_JR=lambda_JR
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



    def train(self, epochs):
        LOSS = 0

        self.model.train()
        self.model.to(self.device)
        i=0
        while True:
            for inputs,targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Calculating Model Jacobian Regularization
                B, C = outputs.shape  # 10*100
                flat_outputs = outputs.reshape(-1)
                v = torch.randn(B, C)
                arxilirary_zero = torch.zeros(B, C)
                vnorm = torch.norm(v, 2, 1, True)
                v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)  # (vnorm/v + arxilirary_zero)
                flat_v = v.reshape(-1)
                flat_v = flat_v.cuda()
                flat_outputs.backward(gradient=flat_v, retain_graph=True, create_graph=True)
                model_grad = []
                for param in self.model.parameters():
                    model_grad = param.grad.view(-1) if not len(model_grad) else torch.cat(
                        (model_grad, param.grad.view(-1)))
                for param in self.model.parameters():
                    param.grad.data = param.grad.data - param.grad.data
                loss_JR = C * torch.norm(model_grad) ** 2 / B
                loss_JR = loss_JR * 0.5

                loss_Super = self.loss(outputs, targets)
                loss = loss_Super + self.lambda_JR * loss_JR
                loss.backward()
                # LOSS+= loss.item()
                self.optimizer.step()
                # LOSS+= loss.item()
                self.optimizer.step()
                i+=1
                if i==self.localstep:
                    break
            if i==self.localstep:
                break



