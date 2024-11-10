import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader

from User.UserAVG import UserAVG
from User.UserLIE import UserLIE
from User.UserFang_new import UserFang
from User.UserNDSS import UserNDSS
from User.UserMinMax import UserMinMax
from User.UserMinSum import UserMinSum
from User.UserLF import UserLF
from User.UserDLF import UserDLF


from Server.Server import Server
from Server.sgd import SGD
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

# Implementation for Trimean Server

class Trimean(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, num_glob_iters, local_epochs,
                 n_attackers, attacker_type, iid, select_users, datadir, beta, partition, numusers, localstep):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate,
                         num_glob_iters, local_epochs, num_users=numusers, times=1, select_users_num=select_users,
                         datadir=datadir, beta=beta, partition=partition)

        # Initialize data for all  users
        # data = read_data(dataset,iid)
        total_users = self.num_users
        self.n_attackers = n_attackers
        self.benign_users = []
        self.attackers = []
        self.optimizer = SGD(self.model.parameters(), lr=1)
        self.criterion = nn.CrossEntropyLoss()
        self.attacker_type = attacker_type
        self.local_server_model = copy.deepcopy(list(self.model.parameters()))
        self.iid = iid
        self.local_epochs = local_epochs
        self.total_train_samples = 0

        if total_users - n_attackers > 0:
            for i in range(0, total_users - n_attackers):
                id = i
                train = self.users_trainset[i]
                # test=self.users_testset[i]
                # id, train,test  = read_user_data(i, data, dataset)
                user = UserAVG(device, id, train, model, batch_size, learning_rate, local_epochs, localstep)
                # user.settest(test)
                self.benign_users.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if n_attackers > 0:
            for i in range(total_users - n_attackers, total_users):
                if self.attacker_type == "LIE":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserLIE(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                   n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "Fang":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserFang(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                   n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "NDSS":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserNDSS(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                   n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "MinMax":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserMinMax(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                   n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "MinSum":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserMinSum(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                   n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "LabelFlip":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    classnum=10
                    if self.dataset == "cifar100":
                        classnum=100
                    user = UserLF(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,classnum)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "DynamicLabelFlip":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    classnum=10
                    if self.dataset == "cifar100":
                        classnum=100
                    user = UserDLF(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,classnum)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples

    def aggregate_grads_Trimean_old(self):
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        self.optimizer.zero_grad()
        #if(self.num_users = self.to)
        user_grads=[]
        final_grad=[]
        for user in self.benign_users:
            param_grad=user.get_grads()
            user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)
        if self.n_attackers>0:
            if self.attacker_type=="Fang"or self.attacker_type=="SH":
                for user in self.attackers:
                    param_grad=user.generated_gradients(user_grads)
                    user_grads=torch.cat((param_grad,user_grads),0)
                    break
            if self.attacker_type=="LIE":
                for user in self.attackers:
                    param_grad=user.generated_gradients(user_grads)
                    for i in range(self.n_attackers):
                        user_grads=torch.cat((user_grads,param_grad[None,:]), 0)
                    break

        final_grad=self.tr_mean(user_grads, 10)
        #print(final_grad)
        start_idx=0
        model_grads=[]
        for param in self.model.parameters():
            param_=final_grad[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
            start_idx=start_idx+len(param.data.view(-1))
            param_=param_.to(self.device)
            model_grads.append(param_)
        self.optimizer.step(model_grads)

    def aggregate_grads_Trimean(self):

        users_param = []
        for user in self.selected_users:
            param = user.get_params()
            users_param = param[None, :] if len(users_param) == 0 else torch.cat((users_param, param[None, :]),
                                                                               0)
        global_param = self.tr_mean(users_param,int(self.select_users_num*0.3))
        local_param = self.model.state_dict()
        start_idx = 0
        for key in local_param:
            local_param[key] = global_param[start_idx:start_idx + len(local_param[key].data.view(-1))].reshape(
                local_param[key].data.shape)
            start_idx = start_idx + len(local_param[key].data.view(-1))
        self.model.load_state_dict(local_param)

    def tr_mean(self,all_updates, n_attackers):
        sorted_updates,indice = torch.sort(all_updates, 0)
        out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
        return out


    def train(self):
        loss = []

        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.send_parameters()
            # Evaluate model each interation
            self.evaluate()
            self.select_users()
            selected_benign_users = []
            for user in self.selected_users:
                if user in self.benign_users:
                    selected_benign_users.append(user)
                    user.train(self.local_epochs)  # * user.train_samples
            '''
            for user in self.selected_users:
                if user in self.attackers:
                    user.set_param_new(selected_benign_users)
            '''
            for user in self.selected_users:
                if user in self.attackers:
                    malicious_state = user.get_malicious_model_state(selected_benign_users)
                    user.load_malicious_state(malicious_state)
            '''
            for user in self.selected_users:
                if user in self.attackers:
                    user.load_malicious_state(malicious_state)
            '''
            self.aggregate_grads_Trimean()
        print("Highest accuracy")
        print(max(self.rs_glob_acc))
        self.save_results()
        self.save_model()

    
